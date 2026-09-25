#ifndef EVOKE_WEIGHTED_SPACE_SAVING_H
#define EVOKE_WEIGHTED_SPACE_SAVING_H

#include "evoke_core.h"

#include <stddef.h>
#include <stdint.h>

typedef struct evoke_weighted_space_saving_counter
{
    uint32_t item;
    double estimate;
    double error;
    size_t heap_position;
} evoke_weighted_space_saving_counter;

typedef struct evoke_weighted_space_saving_hash_slot
{
    uint32_t item;
    size_t counter_index_plus_one;
} evoke_weighted_space_saving_hash_slot;

typedef struct evoke_weighted_space_saving
{
    evoke_weighted_space_saving_counter *counters;
    evoke_weighted_space_saving_hash_slot *hash_slots;
    size_t *heap;
    size_t capacity;
    size_t len;
    size_t hash_capacity;
    uint64_t replacements;
    double total_weight;
    double maximum_error;
} evoke_weighted_space_saving;

void evoke_weighted_space_saving_init_empty(
    evoke_weighted_space_saving *summary
);

evoke_status evoke_weighted_space_saving_init(
    evoke_weighted_space_saving *summary,
    size_t capacity
);

void evoke_weighted_space_saving_free(
    evoke_weighted_space_saving *summary
);

evoke_status evoke_weighted_space_saving_offer(
    evoke_weighted_space_saving *summary,
    uint32_t item,
    double weight
);

#endif
