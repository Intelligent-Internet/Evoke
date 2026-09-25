#include "postgres.h"

#include <stdlib.h>
#include <string.h>

#include "access/table.h"
#include "access/tableam.h"
#include "catalog/pg_collation_d.h"
#include "catalog/pg_type_d.h"
#include "executor/executor.h"
#include "executor/tuptable.h"
#include "funcapi.h"
#include "miscadmin.h"
#include "parser/parse_oper.h"
#include "utils/array.h"
#include "utils/backend_status.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/snapmgr.h"
#include "utils/tuplesort.h"

#include "evoke_scope.h"
#include "evoke_scope_pg.h"

#define EVOKE_SCOPE_SORT_MEMORY_MAX_KB (256 * 1024)
#define EVOKE_SCOPE_SNAPSHOT_BATCH_DOCUMENTS UINT32_C(256)

typedef struct evoke_scope_pg_column
{
    evoke_scope_column serialized;
    Oid output_function;
    bool output_is_varlena;
    int16 element_length;
    bool element_by_value;
    char element_alignment;
} evoke_scope_pg_column;

typedef struct evoke_scope_pg_value
{
    uint32 column_index;
    uint8 kind;
    uint8 *value;
    uint32 value_size;
    uint8 *gram_value;
    uint32 gram_value_size;
} evoke_scope_pg_value;

static void
evoke_scope_pg_value_free(evoke_scope_pg_value *value)
{
    if (value == NULL)
    {
        return;
    }
    if (value->value != NULL)
    {
        pfree(value->value);
    }
    if (value->gram_value != NULL)
    {
        pfree(value->gram_value);
    }
    memset(value, 0, sizeof(*value));
}

static void
evoke_scope_pg_columns_free(
    evoke_scope_pg_column *columns,
    uint32 column_count
)
{
    if (columns == NULL)
    {
        return;
    }
    for (uint32 column = 0; column < column_count; column++)
    {
        if (columns[column].serialized.name != NULL)
        {
            pfree((void *) columns[column].serialized.name);
        }
    }
    pfree(columns);
}

static evoke_status
evoke_scope_pg_put_value(
    Tuplesortstate *sort,
    TupleTableSlot *slot,
    uint32 column_index,
    uint8 kind,
    Oid output_function,
    Datum value,
    uint32 document_id
)
{
    char *output = OidOutputFunctionCall(output_function, value);
    text *text_value;

    if (output == NULL)
    {
        return EVOKE_ERR_FORMAT;
    }
    text_value = cstring_to_text(output);
    ExecClearTuple(slot);
    slot->tts_values[0] = Int32GetDatum((int32) column_index);
    slot->tts_values[1] = Int32GetDatum((int32) kind);
    slot->tts_values[2] = PointerGetDatum(text_value);
    slot->tts_values[3] = Int64GetDatum((int64) document_id);
    for (uint32 attribute = 0; attribute < 4; attribute++)
    {
        slot->tts_isnull[attribute] = false;
    }
    ExecStoreVirtualTuple(slot);
    tuplesort_puttupleslot(sort, slot);
    ExecClearTuple(slot);
    pfree(text_value);
    pfree(output);
    return EVOKE_OK;
}

static evoke_status
evoke_scope_pg_collect_document_batch(
    Relation heap_relation,
    const ItemPointerData *document_tids,
    uint32 first_document,
    uint32 document_count,
    evoke_scope_pg_column *columns,
    uint32 column_count,
    Tuplesortstate *sort,
    TupleTableSlot *input_slot,
    TupleTableSlot *heap_slot
)
{
    volatile bool snapshot_pushed = false;
    evoke_status status = EVOKE_OK;
    uint32 end_document = (uint32) Min(
        (uint64) document_count,
        (uint64) first_document + EVOKE_SCOPE_SNAPSHOT_BATCH_DOCUMENTS
    );

    PG_TRY();
    {
        /*
         * A current MVCC snapshot protects external TOAST values and rejects
         * root TIDs whose heap versions are no longer visible. Keeping it to
         * one bounded batch avoids pinning VACUUM for the corpus-sized scope
         * transpose.
         */
        PushActiveSnapshot(GetLatestSnapshot());
        snapshot_pushed = true;
        for (uint32 document = first_document;
             document < end_document;
             document++)
        {
            ItemPointerData tid;

            if (!ItemPointerIsValid(&document_tids[document]))
            {
                continue;
            }
            tid = document_tids[document];
            ExecClearTuple(heap_slot);
            if (!table_tuple_fetch_row_version(
                    heap_relation,
                    &tid,
                    GetActiveSnapshot(),
                    heap_slot))
            {
                continue;
            }
            for (uint32 column = 0; column < column_count; column++)
            {
                evoke_scope_pg_column *scope_column = &columns[column];
                bool is_null;
                Datum datum = slot_getattr(
                    heap_slot,
                    scope_column->serialized.heap_attribute,
                    &is_null
                );

                if (is_null)
                {
                    continue;
                }
                if (scope_column->serialized.element_type_oid == InvalidOid)
                {
                    status = evoke_scope_pg_put_value(
                        sort,
                        input_slot,
                        column,
                        EVOKE_SCOPE_VALUE_SCALAR,
                        scope_column->output_function,
                        datum,
                        document
                    );
                }
                else
                {
                    Datum *elements = NULL;
                    bool *nulls = NULL;
                    int element_count = 0;

                    deconstruct_array(
                        DatumGetArrayTypeP(datum),
                        scope_column->serialized.element_type_oid,
                        scope_column->element_length,
                        scope_column->element_by_value,
                        scope_column->element_alignment,
                        &elements,
                        &nulls,
                        &element_count
                    );
                    for (int element = 0;
                         status == EVOKE_OK && element < element_count;
                         element++)
                    {
                        if (!nulls[element])
                        {
                            status = evoke_scope_pg_put_value(
                                sort,
                                input_slot,
                                column,
                                EVOKE_SCOPE_VALUE_ARRAY_ELEMENT,
                                scope_column->output_function,
                                elements[element],
                                document
                            );
                        }
                    }
                    if (elements != NULL)
                    {
                        pfree(elements);
                    }
                    if (nulls != NULL)
                    {
                        pfree(nulls);
                    }
                }
                if (status != EVOKE_OK)
                {
                    break;
                }
            }
            if (status != EVOKE_OK)
            {
                break;
            }
            CHECK_FOR_INTERRUPTS();
        }
        PopActiveSnapshot();
        snapshot_pushed = false;
    }
    PG_FINALLY();
    {
        if (snapshot_pushed)
        {
            PopActiveSnapshot();
        }
    }
    PG_END_TRY();
    return status;
}

static bool
evoke_scope_pg_value_matches(
    const evoke_scope_pg_value *value,
    uint32 column_index,
    uint8 kind,
    const uint8 *bytes,
    uint32 size
)
{
    return value != NULL && value->value != NULL &&
        value->column_index == column_index && value->kind == kind &&
        value->value_size == size &&
        (size == 0 || memcmp(value->value, bytes, size) == 0);
}

static evoke_status
evoke_scope_pg_value_set(
    evoke_scope_pg_value *value,
    uint32 column_index,
    uint8 kind,
    const uint8 *bytes,
    uint32 size
)
{
    uint8 *copy;

    if (value == NULL || (size > 0 && bytes == NULL))
    {
        return EVOKE_ERR_INVALID;
    }
    copy = palloc(size == 0 ? 1U : size);
    if (size > 0)
    {
        memcpy(copy, bytes, size);
    }
    evoke_scope_pg_value_free(value);
    value->column_index = column_index;
    value->kind = kind;
    value->value = copy;
    value->value_size = size;
    return EVOKE_OK;
}

static bool
evoke_scope_pg_value_is_ascii(const evoke_scope_pg_value *value)
{
    for (uint32 offset = 0; offset < value->value_size; offset++)
    {
        if (value->value[offset] >= UINT8_C(0x80))
        {
            return false;
        }
    }
    return true;
}

static evoke_status
evoke_scope_pg_prepare_gram_value(
    const evoke_scope_pg_column *column,
    evoke_scope_pg_value *value,
    evoke_scope_value *serialized
)
{
    text *input;
    text *folded;
    Size folded_size;

    if (!OidIsValid(column->serialized.collation_oid))
    {
        return EVOKE_OK;
    }
    if (value->value_size > INT_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    if (evoke_scope_pg_value_is_ascii(value))
    {
        serialized->gram_value = value->value;
        serialized->gram_value_size = value->value_size;
        return EVOKE_OK;
    }
    input = cstring_to_text_with_len(
        (const char *) value->value,
        value->value_size
    );
    folded = DatumGetTextPP(DirectFunctionCall1Coll(
        lower,
        column->serialized.collation_oid,
        PointerGetDatum(input)
    ));
    folded_size = VARSIZE_ANY_EXHDR(folded);
    if (folded_size > UINT32_MAX)
    {
        if ((Pointer) folded != (Pointer) input)
        {
            pfree(folded);
        }
        pfree(input);
        return EVOKE_ERR_RANGE;
    }
    value->gram_value = palloc(folded_size == 0 ? 1U : folded_size);
    if (folded_size > 0)
    {
        memcpy(value->gram_value, VARDATA_ANY(folded), folded_size);
    }
    value->gram_value_size = (uint32) folded_size;
    serialized->gram_value = value->gram_value;
    serialized->gram_value_size = value->gram_value_size;
    if ((Pointer) folded != (Pointer) input)
    {
        pfree(folded);
    }
    pfree(input);
    return EVOKE_OK;
}

evoke_status
evoke_scope_build_for_index(
    Relation index_relation,
    const ItemPointerData *document_tids,
    uint32 document_count,
    uint64 source_authority_checksum,
    size_t maximum_size,
    size_t *required_size_out,
    uint8 **bytes_out,
    size_t *size_out
)
{
    AttrNumber sort_attnums[4] = {1, 2, 3, 4};
    Oid sort_operators[4];
    Oid sort_collations[4] = {
        InvalidOid,
        InvalidOid,
        C_COLLATION_OID,
        InvalidOid
    };
    bool nulls_first[4] = {false, false, false, false};
    Relation heap_relation = NULL;
    TupleDesc sort_desc = NULL;
    TupleTableSlot *heap_slot = NULL;
    TupleTableSlot *input_slot = NULL;
    TupleTableSlot *output_slot = NULL;
    Tuplesortstate *sort = NULL;
    evoke_scope_pg_column *columns = NULL;
    evoke_scope_column *serialized_columns = NULL;
    evoke_scope_pg_value current_value = {0};
    evoke_scope_writer *writer;
    size_t value_dictionary_size = 0;
    uint64 posting_count = 0;
    uint32 value_count = 0;
    uint32 previous_document = 0;
    bool have_current_value = false;
    uint32 key_count;
    uint32 total_count;
    uint32 column_count;
    evoke_status status = EVOKE_OK;

    if (index_relation == NULL || index_relation->rd_index == NULL ||
        document_tids == NULL || document_count == 0 ||
        source_authority_checksum == 0 || required_size_out == NULL ||
        bytes_out == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *bytes_out = NULL;
    *size_out = 0;
    *required_size_out = 0;
    key_count = index_relation->rd_index->indnkeyatts;
    total_count = index_relation->rd_index->indnatts;
    if (total_count < key_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    column_count = total_count - key_count;
    if (column_count == 0)
    {
        return EVOKE_OK;
    }
    writer = palloc0(sizeof(*writer));
    evoke_scope_writer_init(writer);
    columns = palloc0((size_t) column_count * sizeof(*columns));
    heap_relation = table_open(
        index_relation->rd_index->indrelid,
        AccessShareLock
    );
    for (uint32 column = 0; column < column_count; column++)
    {
        uint32 index_attribute = key_count + column;
        AttrNumber heap_attribute =
            index_relation->rd_index->indkey.values[index_attribute];
        Form_pg_attribute attribute;
        Oid output_type;
        char *name;

        if (heap_attribute <= 0 ||
            heap_attribute > RelationGetDescr(heap_relation)->natts)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        attribute = TupleDescAttr(
            RelationGetDescr(heap_relation),
            heap_attribute - 1
        );
        if (attribute->attisdropped)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        name = pstrdup(NameStr(attribute->attname));
        columns[column].serialized.index_attribute =
            (uint16) (index_attribute + 1U);
        columns[column].serialized.heap_attribute = heap_attribute;
        columns[column].serialized.type_oid = attribute->atttypid;
        columns[column].serialized.element_type_oid =
            get_element_type(attribute->atttypid);
        columns[column].serialized.collation_oid = attribute->attcollation;
        columns[column].serialized.type_modifier = attribute->atttypmod;
        columns[column].serialized.name = (const uint8 *) name;
        if (strlen(name) > UINT32_MAX)
        {
            pfree(name);
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        columns[column].serialized.name_size = (uint32) strlen(name);
        output_type = columns[column].serialized.element_type_oid == InvalidOid
            ? attribute->atttypid
            : columns[column].serialized.element_type_oid;
        getTypeOutputInfo(
            output_type,
            &columns[column].output_function,
            &columns[column].output_is_varlena
        );
        if (columns[column].serialized.element_type_oid != InvalidOid)
        {
            get_typlenbyvalalign(
                columns[column].serialized.element_type_oid,
                &columns[column].element_length,
                &columns[column].element_by_value,
                &columns[column].element_alignment
            );
        }
    }

    sort_desc = CreateTemplateTupleDesc(4);
    TupleDescInitEntry(sort_desc, 1, "scope_column", INT4OID, -1, 0);
    TupleDescInitEntry(sort_desc, 2, "scope_kind", INT4OID, -1, 0);
    TupleDescInitEntry(sort_desc, 3, "scope_value", TEXTOID, -1, 0);
    TupleDescInitEntry(sort_desc, 4, "document_id", INT8OID, -1, 0);
    BlessTupleDesc(sort_desc);
    input_slot = MakeSingleTupleTableSlot(sort_desc, &TTSOpsVirtual);
    output_slot = MakeSingleTupleTableSlot(sort_desc, &TTSOpsMinimalTuple);
    get_sort_group_operators(
        INT4OID,
        true,
        false,
        false,
        &sort_operators[0],
        NULL,
        NULL,
        NULL
    );
    sort_operators[1] = sort_operators[0];
    get_sort_group_operators(
        TEXTOID,
        true,
        false,
        false,
        &sort_operators[2],
        NULL,
        NULL,
        NULL
    );
    get_sort_group_operators(
        INT8OID,
        true,
        false,
        false,
        &sort_operators[3],
        NULL,
        NULL,
        NULL
    );
    sort = tuplesort_begin_heap(
        sort_desc,
        4,
        sort_attnums,
        sort_operators,
        sort_collations,
        nulls_first,
        Min(maintenance_work_mem, EVOKE_SCOPE_SORT_MEMORY_MAX_KB),
        NULL,
        TUPLESORT_RANDOMACCESS
    );

    /*
     * Scope describes the immutable index generation, not one worker-wide
     * transaction snapshot. Release catalog state before the relation-sized
     * pass, then use bounded snapshots so external TOAST remains valid without
     * retaining one VACUUM horizon for the complete transpose.
     */
    InvalidateCatalogSnapshot();
    pgstat_report_activity(
        STATE_RUNNING,
        "evoke maintenance: prepare semantic query accelerator scope"
    );
    heap_slot = table_slot_create(heap_relation, NULL);
    for (uint64 batch_document = 0;
         batch_document < document_count;
         batch_document += EVOKE_SCOPE_SNAPSHOT_BATCH_DOCUMENTS)
    {
        status = evoke_scope_pg_collect_document_batch(
            heap_relation,
            document_tids,
            (uint32) batch_document,
            document_count,
            columns,
            column_count,
            sort,
            input_slot,
            heap_slot
        );
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
    }

    tuplesort_performsort(sort);
    while (tuplesort_gettupleslot(
        sort,
        true,
        false,
        output_slot,
        NULL
    ))
    {
        bool is_null[4];
        uint32 column;
        uint8 kind;
        uint32 document;
        Datum value_datum;
        text *text_value;
        const uint8 *value_bytes;
        Size raw_value_size;
        uint32 value_size;
        bool new_value;

        column = (uint32) DatumGetInt32(
            slot_getattr(output_slot, 1, &is_null[0])
        );
        kind = (uint8) DatumGetInt32(
            slot_getattr(output_slot, 2, &is_null[1])
        );
        value_datum = slot_getattr(output_slot, 3, &is_null[2]);
        document = (uint32) DatumGetInt64(
            slot_getattr(output_slot, 4, &is_null[3])
        );
        if (is_null[0] || is_null[1] || is_null[2] || is_null[3] ||
            column >= column_count || document >= document_count)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        text_value = DatumGetTextPP(value_datum);
        value_bytes = (const uint8 *) VARDATA_ANY(text_value);
        raw_value_size = VARSIZE_ANY_EXHDR(text_value);
        if (raw_value_size > UINT32_MAX)
        {
            if ((Pointer) text_value != DatumGetPointer(value_datum))
            {
                pfree(text_value);
            }
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        value_size = (uint32) raw_value_size;
        new_value = !have_current_value ||
            !evoke_scope_pg_value_matches(
                &current_value,
                column,
                kind,
                value_bytes,
                value_size
            );
        if (new_value)
        {
            if (value_count == UINT32_MAX ||
                value_dictionary_size > SIZE_MAX - value_size ||
                posting_count == UINT64_MAX)
            {
                status = EVOKE_ERR_RANGE;
            }
            else
            {
                status = evoke_scope_pg_value_set(
                    &current_value,
                    column,
                    kind,
                    value_bytes,
                    value_size
                );
            }
            if (status == EVOKE_OK)
            {
                if (columns[column].serialized.value_count == 0)
                {
                    columns[column].serialized.first_value = value_count;
                }
                columns[column].serialized.value_count++;
                value_dictionary_size += value_size;
                value_count++;
                posting_count++;
                previous_document = document;
                have_current_value = true;
            }
        }
        else if (document != previous_document)
        {
            if (document < previous_document || posting_count == UINT64_MAX)
            {
                status = EVOKE_ERR_FORMAT;
            }
            else
            {
                posting_count++;
                previous_document = document;
            }
        }
        if ((Pointer) text_value != DatumGetPointer(value_datum))
        {
            pfree(text_value);
        }
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
        ExecClearTuple(output_slot);
    }
    {
        uint32 next_value = 0;

        for (uint32 column = 0; column < column_count; column++)
        {
            if (columns[column].serialized.value_count == 0)
            {
                columns[column].serialized.first_value = next_value;
            }
            else if (columns[column].serialized.first_value != next_value)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            next_value += columns[column].serialized.value_count;
        }
        if (next_value != value_count)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
    }
    serialized_columns = palloc0(
        (size_t) column_count * sizeof(*serialized_columns)
    );
    for (uint32 column = 0; column < column_count; column++)
    {
        serialized_columns[column] = columns[column].serialized;
    }
    status = evoke_scope_writer_begin(
        writer,
        source_authority_checksum,
        document_count,
        serialized_columns,
        column_count,
        value_count,
        value_dictionary_size,
        posting_count,
        maximum_size
    );
    *required_size_out = writer->total_size;
    if (status != EVOKE_OK)
    {
        goto cleanup;
    }
    PG_TRY();
    {
        evoke_scope_pg_value_free(&current_value);
        have_current_value = false;
        previous_document = 0;
        tuplesort_rescan(sort);
        while (tuplesort_gettupleslot(
            sort,
            true,
            false,
            output_slot,
            NULL
        ))
        {
            bool is_null[4];
            uint32 column;
            uint8 kind;
            uint32 document;
            Datum value_datum;
            text *text_value;
            const uint8 *value_bytes;
            Size raw_value_size;
            uint32 value_size;
            bool new_value;

            column = (uint32) DatumGetInt32(
                slot_getattr(output_slot, 1, &is_null[0])
            );
            kind = (uint8) DatumGetInt32(
                slot_getattr(output_slot, 2, &is_null[1])
            );
            value_datum = slot_getattr(output_slot, 3, &is_null[2]);
            document = (uint32) DatumGetInt64(
                slot_getattr(output_slot, 4, &is_null[3])
            );
            if (is_null[0] || is_null[1] || is_null[2] || is_null[3] ||
                column >= column_count || document >= document_count)
            {
                status = EVOKE_ERR_FORMAT;
                goto writer_done;
            }
            text_value = DatumGetTextPP(value_datum);
            value_bytes = (const uint8 *) VARDATA_ANY(text_value);
            raw_value_size = VARSIZE_ANY_EXHDR(text_value);
            if (raw_value_size > UINT32_MAX)
            {
                if ((Pointer) text_value != DatumGetPointer(value_datum))
                {
                    pfree(text_value);
                }
                status = EVOKE_ERR_RANGE;
                goto writer_done;
            }
            value_size = (uint32) raw_value_size;
            new_value = !have_current_value ||
                !evoke_scope_pg_value_matches(
                    &current_value,
                    column,
                    kind,
                    value_bytes,
                    value_size
                );
            if (new_value)
            {
                evoke_scope_value serialized = {0};

                status = evoke_scope_pg_value_set(
                    &current_value,
                    column,
                    kind,
                    value_bytes,
                    value_size
                );
                if (status == EVOKE_OK)
                {
                    serialized.column_index = column;
                    serialized.kind = kind;
                    serialized.value = current_value.value;
                    serialized.value_size = current_value.value_size;
                    status = evoke_scope_pg_prepare_gram_value(
                        &columns[column],
                        &current_value,
                        &serialized
                    );
                }
                if (status == EVOKE_OK)
                {
                    status = evoke_scope_writer_start_value(
                        writer,
                        &serialized
                    );
                }
                if (status == EVOKE_OK)
                {
                    previous_document = document;
                    have_current_value = true;
                }
            }
            if (status == EVOKE_OK &&
                (new_value || document != previous_document))
            {
                if (!new_value && document < previous_document)
                {
                    status = EVOKE_ERR_FORMAT;
                }
                else
                {
                    status = evoke_scope_writer_append_document(
                        writer,
                        document
                    );
                    previous_document = document;
                }
            }
            if ((Pointer) text_value != DatumGetPointer(value_datum))
            {
                pfree(text_value);
            }
            if (status != EVOKE_OK)
            {
                goto writer_done;
            }
            ExecClearTuple(output_slot);
        }
        status = evoke_scope_writer_finish(writer, bytes_out, size_out);
writer_done:
        ;
    }
    PG_FINALLY();
    {
        evoke_scope_writer_free(writer);
    }
    PG_END_TRY();
cleanup:
    if (sort != NULL)
    {
        tuplesort_end(sort);
    }
    if (input_slot != NULL)
    {
        ExecDropSingleTupleTableSlot(input_slot);
    }
    if (output_slot != NULL)
    {
        ExecDropSingleTupleTableSlot(output_slot);
    }
    if (heap_slot != NULL)
    {
        ExecDropSingleTupleTableSlot(heap_slot);
    }
    if (sort_desc != NULL)
    {
        FreeTupleDesc(sort_desc);
    }
    if (heap_relation != NULL)
    {
        table_close(heap_relation, AccessShareLock);
    }
    evoke_scope_writer_free(writer);
    pfree(writer);
    if (serialized_columns != NULL)
    {
        pfree(serialized_columns);
    }
    evoke_scope_pg_value_free(&current_value);
    evoke_scope_pg_columns_free(columns, column_count);
    if (status != EVOKE_OK)
    {
        free(*bytes_out);
        *bytes_out = NULL;
        *size_out = 0;
    }
    return status;
}
