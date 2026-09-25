#include "postgres.h"

#include <stdlib.h>

#include "catalog/storage.h"
#include "utils/memutils.h"

#include "evoke_am_build.h"
#include "evoke_am_accelerator.h"
#include "evoke_am_meta.h"
#include "evoke_segment_pages.h"

#define EVOKE_AM_REBUILD_MEMORY_ESTIMATE_MULTIPLIER 6
#define EVOKE_AM_COMPACT_REBUILD_MEMORY_ESTIMATE_MULTIPLIER 4
#define EVOKE_AM_SPILL_REBUILD_MEMORY_ESTIMATE_MULTIPLIER 2
#define EVOKE_AM_STANDARD_REBUILD_BUDGET_HEADROOM_NUM 3
#define EVOKE_AM_STANDARD_REBUILD_BUDGET_HEADROOM_DEN 5
#define EVOKE_AM_COMPACT_REBUILD_BUDGET_HEADROOM_NUM 3
#define EVOKE_AM_COMPACT_REBUILD_BUDGET_HEADROOM_DEN 4
#define EVOKE_AM_STANDARD_REBUILD_MAX_PAYLOAD_BYTES \
    (UINT64_C(1024) * 1024 * 1024)
#define EVOKE_AM_COMPACT_REBUILD_MAX_PAYLOAD_BYTES \
    (UINT64_C(512) * 1024 * 1024)
#define EVOKE_AM_SEMANTIC_BUILD_FIXED_WORKSPACE_BYTES \
    (UINT64_C(32) * 1024 * 1024)
#define EVOKE_AM_REBUILD_ID_SHIFT 11U

typedef struct evoke_am_semantic_posting_file_reader
{
    BufFile *file;
    size_t posting_count;
    size_t cursor;
} evoke_am_semantic_posting_file_reader;

typedef struct evoke_am_initial_fold_publish_context
{
    evoke_initial_fold_stream *stream;
    evoke_document_cow_record *document_records;
    size_t document_record_count;
    evoke_am_rebuild_output *replacement;
    bool source_released;
} evoke_am_initial_fold_publish_context;

static evoke_status
evoke_am_semantic_posting_file_read(
    void *context,
    evoke_segment_semantic_posting *posting_out
)
{
    evoke_am_semantic_posting_file_reader *reader = context;
    size_t read_bytes;

    if (reader == NULL || posting_out == NULL ||
        reader->file == NULL || reader->cursor >= reader->posting_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    read_bytes = BufFileRead(reader->file, posting_out, sizeof(*posting_out));
    if (read_bytes != sizeof(*posting_out))
    {
        return EVOKE_ERR_FORMAT;
    }
    reader->cursor++;
    return EVOKE_OK;
}

static evoke_status
evoke_am_semantic_posting_file_rewind(void *context)
{
    evoke_am_semantic_posting_file_reader *reader = context;

    if (reader == NULL || reader->file == NULL ||
        BufFileSeek(reader->file, 0, 0, SEEK_SET) != 0)
    {
        return EVOKE_ERR_INVALID;
    }
    reader->cursor = 0;
    return EVOKE_OK;
}

static void
evoke_am_rebuild_output_release_materialized_sources(
    evoke_am_rebuild_output *output
)
{
    if (output == NULL)
    {
        return;
    }

    if (output->doc_tids != NULL)
    {
        pfree(output->doc_tids);
        output->doc_tids = NULL;
    }
    if (output->index_bytes != NULL)
    {
        free(output->index_bytes);
        output->index_bytes = NULL;
        output->index_bytes_len = 0;
    }
    if (output->segment_semantic_postings != NULL)
    {
        BufFileClose(output->segment_semantic_postings);
        output->segment_semantic_postings = NULL;
        output->segment_semantic_posting_count = 0;
    }
    if (output->segment_semantic_input_fingerprints != NULL)
    {
        pfree(output->segment_semantic_input_fingerprints);
        output->segment_semantic_input_fingerprints = NULL;
        output->segment_semantic_input_fingerprint_count = 0;
    }
    if (output->index_valid)
    {
        evoke_index_free(&output->index);
        output->index_valid = false;
    }
}

static evoke_status
evoke_am_initial_fold_publish_next(
    void *context,
    evoke_term_fold_bundle *bundle_out,
    bool *done_out
)
{
    evoke_am_initial_fold_publish_context *publish = context;
    evoke_status status;

    if (publish == NULL || publish->stream == NULL ||
        publish->replacement == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_initial_fold_stream_next(
        publish->stream,
        bundle_out,
        done_out
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (*done_out)
    {
        evoke_initial_fold_stream_free(publish->stream);
        publish->stream = NULL;
        evoke_am_rebuild_output_release_materialized_sources(
            publish->replacement
        );
        publish->source_released = true;
        return EVOKE_OK;
    }
    for (uint32 run_index = 0;
         run_index < bundle_out->run_count;
         run_index++)
    {
        const evoke_term_fold_run *run = &bundle_out->runs[run_index];
        uint64 posting_end = run->posting_offset + run->posting_count;

        for (uint64 posting_index = run->posting_offset;
             posting_index < posting_end;
             posting_index++)
        {
            uint32 document_slot =
                bundle_out->document_slots[posting_index];
            uint64 *residency;

            if (document_slot >= publish->document_record_count)
            {
                return EVOKE_ERR_FORMAT;
            }
            residency = run->kind ==
                EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT
                ? &publish->document_records[document_slot]
                    .semantic_residency
                : &publish->document_records[document_slot]
                    .lexical_residency;
            if (*residency == UINT64_MAX)
            {
                return EVOKE_ERR_RANGE;
            }
            (*residency)++;
        }
    }
    return EVOKE_OK;
}

void
evoke_am_rebuild_output_release(evoke_am_rebuild_output *output)
{
    if (output == NULL)
    {
        return;
    }

    evoke_am_rebuild_output_release_materialized_sources(output);
    memset(output, 0, sizeof(*output));
}

/* Prepare MAIN for one caller-serialized replacement publication. */
uint64
evoke_am_prepare_replacement_relation(Relation indexRelation)
{
    evoke_am_meta_page empty_meta;
    uint64 rebuild_count;

    memset(&empty_meta, 0, sizeof(empty_meta));
    empty_meta.magic = EVOKE_AM_MAGIC;
    empty_meta.version = EVOKE_AM_VERSION;
    empty_meta.page_kind = EVOKE_AM_PAGE_META;

    rebuild_count = evoke_am_next_rebuild_count(indexRelation);
    RelationTruncate(indexRelation, 0);
    evoke_am_write_new_page(
        indexRelation,
        &empty_meta,
        sizeof(empty_meta)
    );
    return rebuild_count;
}

/* Publish one checked rebuild bundle after caller-owned fork setup. */
void
evoke_am_publish_replacement_segments(
    Relation indexRelation,
    ForkNumber fork_number,
    evoke_am_rebuild_output *replacement,
    const uint8 contract_hash[EVOKE_SEGMENT_CONTRACT_HASH_BYTES],
    uint16 cache_epoch,
    uint16 flags,
    uint64 rebuild_count
)
{
    struct evoke_am_segment_publish_cleanup
    {
        evoke_segment_manifest manifest;
        evoke_segment_query_contract query_contract;
        evoke_document_cow_record *document_records;
        evoke_am_initial_fold_publish_context fold_publish;
    } *cleanup;
    evoke_segment_read_root root;
    evoke_am_semantic_posting_file_reader semantic_reader = {0};
    uint64 manifest_id;
    uint64 first_segment_id;
    uint64 active_l0_segment_id;
    uint64 total_document_length = 0;
    evoke_status status;
    size_t document_index;

    if (replacement == NULL || !replacement->index_valid ||
        replacement->index.num_docs != replacement->num_docs ||
        (replacement->num_docs > 0 && replacement->doc_tids == NULL) ||
        (replacement->segment_sae &&
         replacement->segment_semantic_input_fingerprint_count !=
             replacement->num_docs) ||
        (replacement->segment_sae &&
         replacement->num_docs > 0 &&
         replacement->segment_semantic_input_fingerprints == NULL) ||
        (replacement->segment_semantic_posting_count > 0 &&
         replacement->segment_semantic_postings == NULL) ||
        (!replacement->segment_sae &&
         (replacement->segment_semantic_postings != NULL ||
          replacement->segment_semantic_posting_count != 0 ||
          replacement->segment_semantic_input_fingerprints != NULL ||
          replacement->segment_semantic_input_fingerprint_count != 0)) ||
        replacement->num_docs > UINT32_MAX ||
        rebuild_count == 0 ||
        rebuild_count > (UINT64_MAX >> EVOKE_AM_REBUILD_ID_SHIFT))
    {
        ereport(ERROR, (errmsg("invalid evoke convergent replacement")));
    }

    manifest_id =
        (rebuild_count << EVOKE_AM_REBUILD_ID_SHIFT) | UINT64_C(1);
    first_segment_id = manifest_id + 1;
    active_l0_segment_id = first_segment_id;
    cleanup = palloc0(sizeof(*cleanup));
    evoke_segment_manifest_init(&cleanup->manifest);
    evoke_segment_query_contract_init(&cleanup->query_contract);
    cleanup->fold_publish.replacement = replacement;
    memset(&root, 0, sizeof(root));

    PG_TRY();
    {
        evoke_segment_manifest *manifest = &cleanup->manifest;
        const evoke_index *index = &replacement->index;

        manifest->manifest_id = manifest_id;
        manifest->max_sequence = replacement->num_docs;
        manifest->statistics_epoch = manifest_id;
        manifest->visible_document_count = replacement->num_docs;
        manifest->document_slot_count = replacement->num_docs;
        manifest->vocab_size = index->vocab_size;
        if (replacement->segment_sae)
        {
            manifest->flags |= EVOKE_SEGMENT_MANIFEST_FLAG_SAE;
        }
        memcpy(
            manifest->contract_hash,
            contract_hash,
            sizeof(manifest->contract_hash)
        );
        if (manifest->vocab_size > 0)
        {
            manifest->doc_frequencies = calloc(
                manifest->vocab_size,
                sizeof(*manifest->doc_frequencies)
            );
            if (manifest->doc_frequencies == NULL)
            {
                ereport(ERROR, (errmsg("out of memory")));
            }
            /*
             * An empty numeric index can retain its synthetic empty-token
             * vocabulary while omitting exact-stat arrays. Its document
             * frequencies are all zero, which calloc already materialized.
             */
            if (index->doc_frequencies != NULL)
            {
                memcpy(
                    manifest->doc_frequencies,
                    index->doc_frequencies,
                    (size_t) manifest->vocab_size *
                        sizeof(*manifest->doc_frequencies)
                );
            }
            semantic_reader.file = replacement->segment_semantic_postings;
            semantic_reader.posting_count =
                replacement->segment_semantic_posting_count;
            if (semantic_reader.posting_count > 0 &&
                evoke_am_semantic_posting_file_rewind(&semantic_reader) !=
                    EVOKE_OK)
            {
                ereport(
                    ERROR,
                    (errmsg("invalid evoke semantic posting stream"))
                );
            }
            for (size_t posting_index = 0;
                 posting_index < semantic_reader.posting_count;
                 posting_index++)
            {
                evoke_segment_semantic_posting posting;

                if (evoke_am_semantic_posting_file_read(
                        &semantic_reader,
                        &posting) != EVOKE_OK ||
                    posting.term_id >= manifest->vocab_size ||
                    posting.document_slot >= replacement->num_docs ||
                    manifest->doc_frequencies[posting.term_id] ==
                        UINT32_MAX)
                {
                    ereport(
                        ERROR,
                        (errmsg(
                            "invalid evoke replacement semantic posting"
                        ))
                    );
                }
                manifest->doc_frequencies[posting.term_id]++;
            }
        }
        if (replacement->num_docs > 0)
        {
            if (replacement->num_docs >
                SIZE_MAX / sizeof(*cleanup->document_records))
            {
                ereport(ERROR, (errmsg("evoke document directory is too large")));
            }
            cleanup->document_records = calloc(
                replacement->num_docs,
                sizeof(*cleanup->document_records)
            );
            if (cleanup->document_records == NULL)
            {
                ereport(ERROR, (errmsg("out of memory")));
            }
            cleanup->fold_publish.document_records =
                cleanup->document_records;
            cleanup->fold_publish.document_record_count =
                replacement->num_docs;

            for (document_index = 0;
                 document_index < replacement->num_docs;
                 document_index++)
            {
                const ItemPointerData *tid =
                    &replacement->doc_tids[document_index];
                evoke_document_version_record *version =
                    &cleanup->document_records[document_index].version;

                if (!ItemPointerIsValid(tid))
                {
                    ereport(
                        ERROR,
                        (errmsg("invalid evoke replacement tuple identity"))
                    );
                }
                version->document_slot = document_index;
                version->born_sequence = document_index + 1;
                version->heap_block = ItemPointerGetBlockNumber(tid);
                version->heap_offset = ItemPointerGetOffsetNumber(tid);
                version->document_length = index->doc_lengths[document_index];
                version->flags = EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID;
                if (replacement->segment_sae)
                {
                    version->flags |=
                        EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_COMPLETE;
                    memcpy(
                        version->semantic_input_fingerprint,
                        &replacement->
                            segment_semantic_input_fingerprints[
                                document_index *
                                EVOKE_DOCUMENT_FINGERPRINT_BYTES
                            ],
                        sizeof(
                            version->semantic_input_fingerprint
                        )
                    );
                    if (evoke_document_fingerprint_is_zero(
                            version->semantic_input_fingerprint))
                    {
                        ereport(
                            ERROR,
                            (
                                errmsg(
                                    "invalid evoke semantic input "
                                    "fingerprint"
                                ),
                                errdetail(
                                    "Document slot %zu has an all-zero "
                                    "fingerprint.",
                                    document_index
                                )
                            )
                        );
                    }
                }
                if (UINT64_MAX - total_document_length <
                    version->document_length)
                {
                    ereport(
                        ERROR,
                        (errmsg(
                            "evoke replacement document length overflow"
                        ))
                    );
                }
                total_document_length += version->document_length;
            }

            semantic_reader.file = replacement->segment_semantic_postings;
            semantic_reader.posting_count =
                replacement->segment_semantic_posting_count;
            status = evoke_initial_fold_stream_create(
                index,
                replacement->segment_semantic_posting_count,
                evoke_am_semantic_posting_file_read,
                evoke_am_semantic_posting_file_rewind,
                &semantic_reader,
                manifest->manifest_id,
                manifest->max_sequence,
                EVOKE_INITIAL_FOLD_TARGET_BYTES,
                &cleanup->fold_publish.stream
            );
            if (status != EVOKE_OK)
            {
                ereport(
                    ERROR,
                    (
                        errmsg("failed to stream evoke initial folds"),
                        errdetail(
                            "Validation failed: %s.",
                            evoke_strerror(status)
                        )
                    )
                );
            }
            active_l0_segment_id = first_segment_id;
        }
        manifest->total_document_length = total_document_length;
        status = evoke_segment_query_contract_build(
            index,
            manifest,
            &cleanup->query_contract
        );
        if (status != EVOKE_OK)
        {
            ereport(
                ERROR,
                (
                    errmsg("failed to build evoke segment query contract"),
                    errdetail(
                        "Validation failed: %s.",
                        evoke_strerror(status)
                    )
                )
            );
        }
        if (replacement->num_docs == 0)
        {
            evoke_am_rebuild_output_release_materialized_sources(replacement);
            cleanup->fold_publish.source_released = true;
            evoke_segment_pages_write_sealed_bundle_fork(
                indexRelation,
                fork_number,
                manifest,
                &cleanup->query_contract,
                NULL,
                0,
                active_l0_segment_id,
                manifest->max_sequence + 1,
                &root
            );
        }
        else
        {
            evoke_segment_pages_write_streamed_initial_folded_bundle_fork(
                indexRelation,
                fork_number,
                manifest,
                &cleanup->query_contract,
                evoke_am_initial_fold_publish_next,
                &cleanup->fold_publish,
                cleanup->document_records,
                replacement->num_docs,
                active_l0_segment_id,
                manifest->max_sequence + 1,
                &root
            );
        }
        evoke_am_publish_rebuild_meta(
            indexRelation,
            fork_number,
            &root,
            replacement->source_type,
            (uint32) replacement->num_docs,
            cache_epoch,
            flags,
            rebuild_count
        );
    }
    PG_FINALLY();
    {
        evoke_initial_fold_stream_free(cleanup->fold_publish.stream);
        if (!cleanup->fold_publish.source_released)
        {
            evoke_am_rebuild_output_release_materialized_sources(replacement);
        }
        free(cleanup->document_records);
        evoke_segment_query_contract_free(&cleanup->query_contract);
        evoke_segment_manifest_free(&cleanup->manifest);
        pfree(cleanup);
    }
    PG_END_TRY();
}

static uint64
evoke_am_budget_headroom_limit(
    uint64 budget_bytes,
    uint64 numerator,
    uint64 denominator
)
{
    if (denominator == 0)
    {
        return 0;
    }
    if (budget_bytes > UINT64_MAX / numerator)
    {
        return budget_bytes;
    }
    return (budget_bytes * numerator) / denominator;
}

const char *
evoke_am_rebuild_builder_name(evoke_am_rebuild_builder builder)
{
    switch (builder)
    {
        case EVOKE_AM_REBUILD_BUILDER_COMPACT:
            return "compact";
        case EVOKE_AM_REBUILD_BUILDER_SPILL:
            return "spill";
        case EVOKE_AM_REBUILD_BUILDER_SEMANTIC_STREAM:
            return "semantic_stream";
        case EVOKE_AM_REBUILD_BUILDER_STANDARD:
        default:
            return "standard";
    }
}

bool
evoke_am_rebuild_memory_budget_choose(
    bool semantic_streaming,
    const evoke_am_rebuild_workload *workload,
    int rebuild_memory_budget_mb,
    int work_mem_kb,
    evoke_am_rebuild_builder *builder_out,
    uint64 *standard_estimated_bytes_out,
    uint64 *compact_estimated_bytes_out,
    uint64 *spill_estimated_bytes_out,
    uint64 *budget_bytes_out
)
{
    uint64 payload_bytes;
    uint64 standard_estimated_bytes;
    uint64 compact_estimated_bytes;
    uint64 spill_estimated_bytes;
    uint64 budget_bytes;
    uint64 standard_budget_limit;
    uint64 compact_budget_limit;
    bool standard_safe;
    bool compact_safe;

    if (builder_out != NULL)
    {
        *builder_out = semantic_streaming
            ? EVOKE_AM_REBUILD_BUILDER_SEMANTIC_STREAM
            : EVOKE_AM_REBUILD_BUILDER_STANDARD;
    }
    if (standard_estimated_bytes_out != NULL)
    {
        *standard_estimated_bytes_out = 0;
    }
    if (compact_estimated_bytes_out != NULL)
    {
        *compact_estimated_bytes_out = 0;
    }
    if (spill_estimated_bytes_out != NULL)
    {
        *spill_estimated_bytes_out = 0;
    }
    if (budget_bytes_out != NULL)
    {
        *budget_bytes_out = 0;
    }
    if (rebuild_memory_budget_mb <= 0 || workload == NULL)
    {
        return true;
    }

    budget_bytes = (uint64) rebuild_memory_budget_mb * 1024ULL * 1024ULL;
    if (budget_bytes_out != NULL)
    {
        *budget_bytes_out = budget_bytes;
    }
    if (semantic_streaming)
    {
        uint64 identity_workspace = evoke_u64_saturating_mul(
            workload->identity_source_bytes,
            8
        );
        uint64 delta_workspace = evoke_u64_saturating_mul(
            workload->mutation_bytes,
            4
        );
        uint64 sort_workspace = evoke_u64_saturating_mul(
            (uint64) Max(work_mem_kb, 0),
            1024
        );

        /*
         * Incremental unified compaction retains several O(documents) maps:
         * source descriptors, liveness flags, lengths, build TIDs,
         * replacement TIDs, and semantic starts. Eight times persisted TID
         * bytes conservatively covers those arrays and allocator growth.
         * Decoded linked-L0 rows transiently coexist with their durable
         * payload; reserve four times raw mutation bytes for atoms, weights,
         * and maps.
         */
        spill_estimated_bytes = evoke_u64_saturating_add(
            identity_workspace,
            delta_workspace
        );
        spill_estimated_bytes = evoke_u64_saturating_add(
            spill_estimated_bytes,
            sort_workspace
        );
        spill_estimated_bytes = evoke_u64_saturating_add(
            spill_estimated_bytes,
            EVOKE_AM_SEMANTIC_BUILD_FIXED_WORKSPACE_BYTES
        );
        if (spill_estimated_bytes_out != NULL)
        {
            *spill_estimated_bytes_out = spill_estimated_bytes;
        }
        return spill_estimated_bytes <= budget_bytes;
    }

    payload_bytes = workload->live_payload_bytes;
    /*
     * BM25 standard keeps raw docs, term stats, COO arrays, CSC arrays, and
     * serialized bytes live at overlapping points. Compact removes the
     * doc_terms/COO overlap. Spill moves term entries to PostgreSQL temp files
     * and streams publish, leaving final arrays as the main memory consumer.
     * SAE builds return above: posting pairs and impact-head starts spill for
     * every builder, so only document maps, work_mem, and bounded batch state
     * count toward backend memory admission.
     */
    standard_estimated_bytes = evoke_u64_saturating_mul(
        payload_bytes,
        EVOKE_AM_REBUILD_MEMORY_ESTIMATE_MULTIPLIER
    );
    compact_estimated_bytes = evoke_u64_saturating_mul(
        payload_bytes,
        EVOKE_AM_COMPACT_REBUILD_MEMORY_ESTIMATE_MULTIPLIER
    );
    spill_estimated_bytes = evoke_u64_saturating_mul(
        payload_bytes,
        EVOKE_AM_SPILL_REBUILD_MEMORY_ESTIMATE_MULTIPLIER
    );
    standard_safe =
        payload_bytes <= EVOKE_AM_STANDARD_REBUILD_MAX_PAYLOAD_BYTES;
    compact_safe =
        payload_bytes <= EVOKE_AM_COMPACT_REBUILD_MAX_PAYLOAD_BYTES;
    standard_budget_limit = evoke_am_budget_headroom_limit(
        budget_bytes,
        EVOKE_AM_STANDARD_REBUILD_BUDGET_HEADROOM_NUM,
        EVOKE_AM_STANDARD_REBUILD_BUDGET_HEADROOM_DEN
    );
    compact_budget_limit = evoke_am_budget_headroom_limit(
        budget_bytes,
        EVOKE_AM_COMPACT_REBUILD_BUDGET_HEADROOM_NUM,
        EVOKE_AM_COMPACT_REBUILD_BUDGET_HEADROOM_DEN
    );
    if (standard_estimated_bytes_out != NULL)
    {
        *standard_estimated_bytes_out = standard_estimated_bytes;
    }
    if (compact_estimated_bytes_out != NULL)
    {
        *compact_estimated_bytes_out = compact_estimated_bytes;
    }
    if (spill_estimated_bytes_out != NULL)
    {
        *spill_estimated_bytes_out = spill_estimated_bytes;
    }
    /*
     * Automatic rebuilds optimize for steady query service, not fastest build
     * time. The standard builder has the widest live-memory overlap, so it
     * needs substantial headroom below the configured budget. Compact is
     * safer but still keeps final arrays plus term entries live. Spill is the
     * only path allowed to use the full budget envelope.
     */
    if (standard_safe && standard_estimated_bytes <= standard_budget_limit)
    {
        return true;
    }
    /*
     * Compact still keeps its term-entry array in backend memory. Large
     * payloads can cross PostgreSQL's single-allocation cap even when the
     * coarse total-byte estimate appears to fit, so large rebuilds must go to
     * the spill builder.
     */
    if (compact_safe && compact_estimated_bytes <= compact_budget_limit)
    {
        if (builder_out != NULL)
        {
            *builder_out = EVOKE_AM_REBUILD_BUILDER_COMPACT;
        }
        return true;
    }
    if (spill_estimated_bytes <= budget_bytes)
    {
        if (builder_out != NULL)
        {
            *builder_out = EVOKE_AM_REBUILD_BUILDER_SPILL;
        }
        return true;
    }
    return false;
}
