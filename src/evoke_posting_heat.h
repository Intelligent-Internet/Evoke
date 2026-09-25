#ifndef EVOKE_POSTING_HEAT_H
#define EVOKE_POSTING_HEAT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define EVOKE_POSTING_HEAT_WAYS 4U

typedef struct evoke_posting_heat_observation
{
    uint32_t database_id;
    uint32_t index_id;
    uint32_t term_id;
    uint64_t root_id;
    uint64_t query_count;
    uint64_t root_query_count;
    uint64_t extent_transition_work;
    uint64_t posting_work;
    uint64_t block_work;
    uint64_t blocks_considered;
    uint64_t blocks_scored;
    uint64_t blocks_skipped;
    uint64_t postings_scored;
    uint64_t last_posting_count;
    uint32_t last_extent_count;
    uint32_t last_block_count;
    bool impact_specialization_ready;
    bool hot_cache_republish_ready;
    uint64_t observed_at_ms;
} evoke_posting_heat_observation;

typedef struct evoke_posting_heat_entry
{
    uint32_t database_id;
    uint32_t index_id;
    uint32_t term_id;
    uint32_t last_extent_count;
    uint32_t last_block_count;
    bool impact_specialization_ready;
    bool hot_cache_republish_ready;
    uint8_t reserved[2];
    uint64_t root_id;
    uint64_t query_heat;
    uint64_t root_heat;
    uint64_t extent_transition_work;
    uint64_t posting_work;
    uint64_t block_work;
    uint64_t blocks_considered;
    uint64_t blocks_scored;
    uint64_t blocks_skipped;
    uint64_t postings_scored;
    uint64_t last_posting_count;
    uint64_t last_seen_ms;
    uint64_t last_decay_ms;
    uint64_t last_attempt_root_id;
    uint64_t last_attempt_ms;
} evoke_posting_heat_entry;

typedef struct evoke_posting_heat_merge_result
{
    bool valid;
    bool evicted;
    bool became_candidate;
    uint64_t query_heat;
    uint64_t root_heat;
} evoke_posting_heat_merge_result;

typedef struct evoke_posting_heat_summary
{
    uint32_t entry_count;
    uint64_t total_query_heat;
    bool candidate_found;
    evoke_posting_heat_entry candidate;
} evoke_posting_heat_summary;

bool evoke_posting_heat_table_valid(size_t capacity);

evoke_posting_heat_merge_result evoke_posting_heat_merge(
    evoke_posting_heat_entry *entries,
    size_t capacity,
    const evoke_posting_heat_observation *observation,
    uint64_t half_life_ms,
    uint64_t candidate_heat,
    uint64_t candidate_root_heat
);

bool evoke_posting_heat_select(
    evoke_posting_heat_entry *entries,
    size_t capacity,
    uint32_t database_id,
    uint32_t index_id,
    uint64_t root_id,
    uint64_t now_ms,
    uint64_t half_life_ms,
    uint64_t minimum_heat,
    uint64_t minimum_root_heat,
    uint64_t retry_cooldown_ms,
    evoke_posting_heat_entry *candidate_out
);

bool evoke_posting_heat_note_attempt(
    evoke_posting_heat_entry *entries,
    size_t capacity,
    uint32_t database_id,
    uint32_t index_id,
    uint32_t term_id,
    uint64_t root_id,
    uint64_t attempted_at_ms
);

bool evoke_posting_heat_note_hot_cache_resident(
    evoke_posting_heat_entry *entries,
    size_t capacity,
    uint32_t database_id,
    uint32_t index_id,
    uint32_t term_id,
    uint64_t root_id
);

void evoke_posting_heat_summarize(
    evoke_posting_heat_entry *entries,
    size_t capacity,
    uint32_t database_id,
    uint32_t index_id,
    uint64_t root_id,
    uint64_t now_ms,
    uint64_t half_life_ms,
    uint64_t minimum_heat,
    uint64_t minimum_root_heat,
    uint64_t retry_cooldown_ms,
    evoke_posting_heat_summary *summary_out
);

#endif
