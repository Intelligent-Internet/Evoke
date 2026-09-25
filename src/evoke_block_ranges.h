#ifndef EVOKE_BLOCK_RANGES_H
#define EVOKE_BLOCK_RANGES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "evoke_core.h"

typedef struct evoke_block_range
{
    uint32_t start_block;
    uint32_t block_count;
} evoke_block_range;

typedef struct evoke_block_range_inventory
{
    evoke_block_range *ranges;
    size_t range_count;
    size_t range_capacity;
    size_t observed_range_count;
    uint64_t reachable_block_count;
    uint64_t interior_unreachable_block_count;
    uint32_t highest_reachable_block_exclusive;
    uint32_t published_block_high_watermark;
    bool finalized;
} evoke_block_range_inventory;

typedef struct evoke_block_range_allocator
{
    evoke_block_range *ranges;
    size_t range_count;
    uint64_t available_block_count;
    uint64_t allocated_block_count;
    bool initialized;
} evoke_block_range_allocator;

void evoke_block_range_inventory_init(
    evoke_block_range_inventory *inventory
);

void evoke_block_range_inventory_free(
    evoke_block_range_inventory *inventory
);

evoke_status evoke_block_range_inventory_add(
    evoke_block_range_inventory *inventory,
    uint32_t start_block,
    uint32_t block_count
);

/*
 * Sort, deduplicate, and merge the exact live ranges below one checked root
 * high-water mark. Exact duplicate references are legal; partial overlap is
 * corruption because two distinct immutable objects may not own the same
 * physical page.
 */
evoke_status evoke_block_range_inventory_finalize(
    evoke_block_range_inventory *inventory,
    uint32_t published_block_high_watermark
);

void evoke_block_range_allocator_init(
    evoke_block_range_allocator *allocator
);

void evoke_block_range_allocator_free(
    evoke_block_range_allocator *allocator
);

/*
 * Build the exact complement of one finalized reachability inventory below
 * its checked high-water mark. Block zero must be reachable and is never
 * allocatable.
 */
evoke_status evoke_block_range_allocator_build(
    const evoke_block_range_inventory *inventory,
    evoke_block_range_allocator *allocator
);

/*
 * Build directly from canonical free ranges authenticated by one manifest.
 * The ranges must be sorted, non-adjacent, and lie below the checked root
 * high-water mark. Block zero is never reusable.
 */
evoke_status evoke_block_range_allocator_build_free(
    const evoke_block_range *ranges,
    size_t range_count,
    uint32_t published_block_high_watermark,
    evoke_block_range_allocator *allocator
);

/*
 * Consume the smallest free run that can hold block_count contiguous pages.
 * Equal-sized runs are selected in physical order.
 */
bool evoke_block_range_allocator_allocate_best_fit(
    evoke_block_range_allocator *allocator,
    uint32_t block_count,
    uint32_t *start_block_out
);

#endif
