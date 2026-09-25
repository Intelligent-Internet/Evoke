#include "evoke_posting_heat.h"

#include <limits.h>
#include <string.h>

static uint64_t
evoke_posting_heat_add_saturating(uint64_t left, uint64_t right)
{
    return UINT64_MAX - left < right ? UINT64_MAX : left + right;
}

static uint64_t
evoke_posting_heat_hash(
    uint32_t database_id,
    uint32_t index_id,
    uint32_t term_id
)
{
    uint64_t key =
        ((uint64_t) database_id << 32) ^
        ((uint64_t) index_id << 1) ^
        (uint64_t) term_id;

    key ^= key >> 33;
    key *= UINT64_C(0xff51afd7ed558ccd);
    key ^= key >> 33;
    key *= UINT64_C(0xc4ceb9fe1a85ec53);
    key ^= key >> 33;
    return key;
}

static bool
evoke_posting_heat_entry_valid(const evoke_posting_heat_entry *entry)
{
    return entry != NULL &&
        entry->database_id != 0 &&
        entry->index_id != 0;
}

static bool
evoke_posting_heat_entry_matches(
    const evoke_posting_heat_entry *entry,
    uint32_t database_id,
    uint32_t index_id,
    uint32_t term_id
)
{
    return evoke_posting_heat_entry_valid(entry) &&
        entry->database_id == database_id &&
        entry->index_id == index_id &&
        entry->term_id == term_id;
}

static void
evoke_posting_heat_decay(
    evoke_posting_heat_entry *entry,
    uint64_t now_ms,
    uint64_t half_life_ms
)
{
    uint64_t elapsed_ms;
    uint64_t steps;
    unsigned int shift;

    if (!evoke_posting_heat_entry_valid(entry) ||
        half_life_ms == 0 ||
        now_ms <= entry->last_decay_ms)
    {
        return;
    }
    elapsed_ms = now_ms - entry->last_decay_ms;
    steps = elapsed_ms / half_life_ms;
    if (steps == 0)
    {
        return;
    }
    if (steps >= 64)
    {
        entry->query_heat = 0;
        entry->root_heat = 0;
        entry->extent_transition_work = 0;
        entry->posting_work = 0;
        entry->block_work = 0;
        entry->blocks_considered = 0;
        entry->blocks_scored = 0;
        entry->blocks_skipped = 0;
        entry->postings_scored = 0;
    }
    else
    {
        shift = (unsigned int) steps;
        entry->query_heat >>= shift;
        entry->root_heat >>= shift;
        entry->extent_transition_work >>= shift;
        entry->posting_work >>= shift;
        entry->block_work >>= shift;
        entry->blocks_considered >>= shift;
        entry->blocks_scored >>= shift;
        entry->blocks_skipped >>= shift;
        entry->postings_scored >>= shift;
    }
    entry->last_decay_ms = now_ms - (elapsed_ms % half_life_ms);
}

static size_t
evoke_posting_heat_bucket_start(
    size_t capacity,
    uint32_t database_id,
    uint32_t index_id,
    uint32_t term_id
)
{
    size_t bucket_count = capacity / EVOKE_POSTING_HEAT_WAYS;
    uint64_t hash = evoke_posting_heat_hash(
        database_id,
        index_id,
        term_id
    );

    return (size_t) (hash % bucket_count) * EVOKE_POSTING_HEAT_WAYS;
}

static evoke_posting_heat_entry *
evoke_posting_heat_find(
    evoke_posting_heat_entry *entries,
    size_t capacity,
    uint32_t database_id,
    uint32_t index_id,
    uint32_t term_id
)
{
    size_t bucket_start;
    size_t way;

    if (!evoke_posting_heat_table_valid(capacity) || entries == NULL)
    {
        return NULL;
    }
    bucket_start = evoke_posting_heat_bucket_start(
        capacity,
        database_id,
        index_id,
        term_id
    );
    for (way = 0; way < EVOKE_POSTING_HEAT_WAYS; way++)
    {
        evoke_posting_heat_entry *entry = &entries[bucket_start + way];

        if (evoke_posting_heat_entry_matches(
                entry,
                database_id,
                index_id,
                term_id))
        {
            return entry;
        }
    }
    return NULL;
}

bool
evoke_posting_heat_table_valid(size_t capacity)
{
    size_t bucket_count;

    if (capacity < EVOKE_POSTING_HEAT_WAYS ||
        capacity % EVOKE_POSTING_HEAT_WAYS != 0)
    {
        return false;
    }
    bucket_count = capacity / EVOKE_POSTING_HEAT_WAYS;
    return (bucket_count & (bucket_count - 1)) == 0;
}

evoke_posting_heat_merge_result
evoke_posting_heat_merge(
    evoke_posting_heat_entry *entries,
    size_t capacity,
    const evoke_posting_heat_observation *observation,
    uint64_t half_life_ms,
    uint64_t candidate_heat,
    uint64_t candidate_root_heat
)
{
    evoke_posting_heat_merge_result result = {0};
    evoke_posting_heat_entry *entry = NULL;
    evoke_posting_heat_entry *victim = NULL;
    size_t bucket_start;
    uint64_t prior_heat = 0;
    uint64_t prior_root_heat = 0;
    bool prior_hot_cache_republish_ready = false;
    bool observation_is_current = false;

    if (entries == NULL || observation == NULL ||
        !evoke_posting_heat_table_valid(capacity) ||
        observation->database_id == 0 ||
        observation->index_id == 0 ||
        observation->query_count == 0 ||
        observation->root_query_count > observation->query_count ||
        observation->observed_at_ms == 0)
    {
        return result;
    }

    bucket_start = evoke_posting_heat_bucket_start(
        capacity,
        observation->database_id,
        observation->index_id,
        observation->term_id
    );
    for (size_t way = 0; way < EVOKE_POSTING_HEAT_WAYS; way++)
    {
        evoke_posting_heat_entry *candidate =
            &entries[bucket_start + way];

        if (evoke_posting_heat_entry_matches(
                candidate,
                observation->database_id,
                observation->index_id,
                observation->term_id))
        {
            entry = candidate;
            break;
        }
        if (!evoke_posting_heat_entry_valid(candidate))
        {
            if (victim == NULL)
            {
                victim = candidate;
            }
            continue;
        }
        evoke_posting_heat_decay(
            candidate,
            observation->observed_at_ms,
            half_life_ms
        );
        if (victim == NULL ||
            candidate->query_heat < victim->query_heat ||
            (candidate->query_heat == victim->query_heat &&
             candidate->last_seen_ms < victim->last_seen_ms))
        {
            victim = candidate;
        }
    }

    if (entry == NULL)
    {
        entry = victim;
        if (entry == NULL)
        {
            return result;
        }
        result.evicted = evoke_posting_heat_entry_valid(entry);
        memset(entry, 0, sizeof(*entry));
        entry->database_id = observation->database_id;
        entry->index_id = observation->index_id;
        entry->term_id = observation->term_id;
        entry->last_decay_ms = observation->observed_at_ms;
    }
    else
    {
        evoke_posting_heat_decay(
            entry,
            observation->observed_at_ms,
            half_life_ms
        );
        prior_heat = entry->query_heat;
        prior_root_heat = entry->root_heat;
        prior_hot_cache_republish_ready =
            entry->hot_cache_republish_ready;
    }

    entry->query_heat = evoke_posting_heat_add_saturating(
        entry->query_heat,
        observation->query_count
    );

    if (observation->root_id > entry->root_id)
    {
        entry->root_id = observation->root_id;
        entry->root_heat = observation->root_query_count;
        entry->extent_transition_work = 0;
        entry->posting_work = 0;
        entry->block_work = 0;
        entry->blocks_considered = 0;
        entry->blocks_scored = 0;
        entry->blocks_skipped = 0;
        entry->postings_scored = 0;
        entry->impact_specialization_ready = false;
        entry->hot_cache_republish_ready = false;
        prior_root_heat = 0;
        prior_hot_cache_republish_ready = false;
        observation_is_current = true;
    }
    else if (observation->root_id == entry->root_id)
    {
        entry->root_heat = evoke_posting_heat_add_saturating(
            entry->root_heat,
            observation->root_query_count
        );
        observation_is_current = true;
    }
    if (observation_is_current)
    {
        entry->extent_transition_work =
            evoke_posting_heat_add_saturating(
                entry->extent_transition_work,
                observation->extent_transition_work
            );
        entry->posting_work = evoke_posting_heat_add_saturating(
            entry->posting_work,
            observation->posting_work
        );
        entry->block_work = evoke_posting_heat_add_saturating(
            entry->block_work,
            observation->block_work
        );
        entry->blocks_considered = evoke_posting_heat_add_saturating(
            entry->blocks_considered,
            observation->blocks_considered
        );
        entry->blocks_scored = evoke_posting_heat_add_saturating(
            entry->blocks_scored,
            observation->blocks_scored
        );
        entry->blocks_skipped = evoke_posting_heat_add_saturating(
            entry->blocks_skipped,
            observation->blocks_skipped
        );
        entry->postings_scored = evoke_posting_heat_add_saturating(
            entry->postings_scored,
            observation->postings_scored
        );
    }
    if (observation_is_current &&
        observation->observed_at_ms >= entry->last_seen_ms)
    {
        entry->last_extent_count = observation->last_extent_count;
        entry->last_block_count = observation->last_block_count;
        entry->last_posting_count = observation->last_posting_count;
        entry->impact_specialization_ready =
            observation->impact_specialization_ready;
        entry->hot_cache_republish_ready =
            observation->hot_cache_republish_ready;
        if (entry->hot_cache_republish_ready &&
            !prior_hot_cache_republish_ready)
        {
            entry->last_attempt_root_id = 0;
            entry->last_attempt_ms = 0;
        }
    }
    entry->last_seen_ms =
        observation->observed_at_ms > entry->last_seen_ms
            ? observation->observed_at_ms
            : entry->last_seen_ms;

    result.valid = true;
    result.query_heat = entry->query_heat;
    result.root_heat = entry->root_heat;
    result.became_candidate =
        (entry->last_extent_count > 1 ||
         entry->impact_specialization_ready ||
         entry->hot_cache_republish_ready) &&
        entry->query_heat >= candidate_heat &&
        entry->root_heat >= candidate_root_heat &&
        (prior_heat < candidate_heat ||
         prior_root_heat < candidate_root_heat);
    return result;
}

static bool
evoke_posting_heat_candidate_is_better(
    const evoke_posting_heat_entry *candidate,
    const evoke_posting_heat_entry *current
)
{
    uint64_t candidate_fragment_work;
    uint64_t candidate_scored_exposure;
    uint64_t current_fragment_work;
    uint64_t current_scored_exposure;

    if (current == NULL)
    {
        return true;
    }
    candidate_scored_exposure =
        candidate->blocks_considered >= candidate->blocks_skipped
            ? candidate->blocks_considered - candidate->blocks_skipped
            : 0;
    candidate_fragment_work = evoke_posting_heat_add_saturating(
        candidate->extent_transition_work,
        candidate_scored_exposure
    );
    current_scored_exposure =
        current->blocks_considered >= current->blocks_skipped
            ? current->blocks_considered - current->blocks_skipped
            : 0;
    current_fragment_work = evoke_posting_heat_add_saturating(
        current->extent_transition_work,
        current_scored_exposure
    );
    if (candidate_fragment_work != current_fragment_work)
    {
        return candidate_fragment_work > current_fragment_work;
    }
    if (candidate->root_heat != current->root_heat)
    {
        return candidate->root_heat > current->root_heat;
    }
    if (candidate->query_heat != current->query_heat)
    {
        return candidate->query_heat > current->query_heat;
    }
    return candidate->term_id < current->term_id;
}

bool
evoke_posting_heat_select(
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
)
{
    evoke_posting_heat_entry *best = NULL;

    if (entries == NULL || candidate_out == NULL ||
        !evoke_posting_heat_table_valid(capacity) ||
        database_id == 0 || index_id == 0 || root_id == 0 ||
        now_ms == 0)
    {
        return false;
    }
    memset(candidate_out, 0, sizeof(*candidate_out));
    for (size_t index = 0; index < capacity; index++)
    {
        evoke_posting_heat_entry *entry = &entries[index];

        if (!evoke_posting_heat_entry_valid(entry) ||
            entry->database_id != database_id ||
            entry->index_id != index_id)
        {
            continue;
        }
        evoke_posting_heat_decay(entry, now_ms, half_life_ms);
        if (entry->root_id != root_id ||
            entry->query_heat < minimum_heat ||
            entry->root_heat < minimum_root_heat ||
            (entry->last_extent_count < 2 &&
             !entry->impact_specialization_ready &&
             !entry->hot_cache_republish_ready))
        {
            continue;
        }
        if (entry->last_attempt_root_id == root_id &&
            now_ms >= entry->last_attempt_ms &&
            now_ms - entry->last_attempt_ms < retry_cooldown_ms)
        {
            continue;
        }
        if (evoke_posting_heat_candidate_is_better(entry, best))
        {
            best = entry;
        }
    }
    if (best == NULL)
    {
        return false;
    }
    *candidate_out = *best;
    return true;
}

bool
evoke_posting_heat_note_attempt(
    evoke_posting_heat_entry *entries,
    size_t capacity,
    uint32_t database_id,
    uint32_t index_id,
    uint32_t term_id,
    uint64_t root_id,
    uint64_t attempted_at_ms
)
{
    evoke_posting_heat_entry *entry = evoke_posting_heat_find(
        entries,
        capacity,
        database_id,
        index_id,
        term_id
    );

    if (entry == NULL || entry->root_id != root_id)
    {
        return false;
    }
    entry->last_attempt_root_id = root_id;
    entry->last_attempt_ms = attempted_at_ms;
    return true;
}

bool
evoke_posting_heat_note_hot_cache_resident(
    evoke_posting_heat_entry *entries,
    size_t capacity,
    uint32_t database_id,
    uint32_t index_id,
    uint32_t term_id,
    uint64_t root_id
)
{
    evoke_posting_heat_entry *entry = evoke_posting_heat_find(
        entries,
        capacity,
        database_id,
        index_id,
        term_id
    );

    if (entry == NULL || entry->root_id != root_id)
    {
        return false;
    }
    entry->hot_cache_republish_ready = false;
    return true;
}

void
evoke_posting_heat_summarize(
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
)
{
    if (summary_out == NULL)
    {
        return;
    }
    memset(summary_out, 0, sizeof(*summary_out));
    if (entries == NULL || !evoke_posting_heat_table_valid(capacity))
    {
        return;
    }
    for (size_t index = 0; index < capacity; index++)
    {
        evoke_posting_heat_entry *entry = &entries[index];

        if (!evoke_posting_heat_entry_valid(entry) ||
            entry->database_id != database_id ||
            entry->index_id != index_id)
        {
            continue;
        }
        evoke_posting_heat_decay(entry, now_ms, half_life_ms);
        summary_out->entry_count++;
        summary_out->total_query_heat = evoke_posting_heat_add_saturating(
            summary_out->total_query_heat,
            entry->query_heat
        );
    }
    summary_out->candidate_found = evoke_posting_heat_select(
        entries,
        capacity,
        database_id,
        index_id,
        root_id,
        now_ms,
        half_life_ms,
        minimum_heat,
        minimum_root_heat,
        retry_cooldown_ms,
        &summary_out->candidate
    );
}
