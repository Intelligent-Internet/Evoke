#include "evoke_block_ranges.h"

#include <stdlib.h>
#include <string.h>

static int
evoke_block_range_compare(const void *left, const void *right)
{
    const evoke_block_range *a = left;
    const evoke_block_range *b = right;

    if (a->start_block != b->start_block)
    {
        return a->start_block < b->start_block ? -1 : 1;
    }
    if (a->block_count != b->block_count)
    {
        return a->block_count < b->block_count ? -1 : 1;
    }
    return 0;
}

void
evoke_block_range_inventory_init(evoke_block_range_inventory *inventory)
{
    if (inventory != NULL)
    {
        memset(inventory, 0, sizeof(*inventory));
    }
}

void
evoke_block_range_inventory_free(evoke_block_range_inventory *inventory)
{
    if (inventory == NULL)
    {
        return;
    }
    free(inventory->ranges);
    memset(inventory, 0, sizeof(*inventory));
}

evoke_status
evoke_block_range_inventory_add(
    evoke_block_range_inventory *inventory,
    uint32_t start_block,
    uint32_t block_count
)
{
    evoke_block_range *resized;
    size_t next_capacity;

    if (inventory == NULL || inventory->finalized || block_count == 0 ||
        (uint64_t) start_block + block_count > UINT32_MAX)
    {
        return EVOKE_ERR_INVALID;
    }
    if (inventory->range_count == inventory->range_capacity)
    {
        next_capacity = inventory->range_capacity == 0
            ? 16
            : inventory->range_capacity * 2;
        if (next_capacity < inventory->range_capacity ||
            next_capacity > SIZE_MAX / sizeof(*inventory->ranges))
        {
            return EVOKE_ERR_RANGE;
        }
        resized = realloc(
            inventory->ranges,
            next_capacity * sizeof(*inventory->ranges)
        );
        if (resized == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        inventory->ranges = resized;
        inventory->range_capacity = next_capacity;
    }
    inventory->ranges[inventory->range_count].start_block = start_block;
    inventory->ranges[inventory->range_count].block_count = block_count;
    inventory->range_count++;
    inventory->observed_range_count++;
    return EVOKE_OK;
}

evoke_status
evoke_block_range_inventory_finalize(
    evoke_block_range_inventory *inventory,
    uint32_t published_block_high_watermark
)
{
    size_t unique_count = 0;
    size_t output_count = 0;
    uint64_t reachable_blocks = 0;

    if (inventory == NULL || inventory->finalized ||
        published_block_high_watermark == 0 ||
        inventory->range_count == 0)
    {
        return EVOKE_ERR_INVALID;
    }
    qsort(
        inventory->ranges,
        inventory->range_count,
        sizeof(*inventory->ranges),
        evoke_block_range_compare
    );

    /*
     * Deduplicate and reject overlap before merging adjacency. Merging first
     * would make a later exact duplicate appear to overlap the enlarged
     * output range.
     */
    for (size_t index = 0; index < inventory->range_count; index++)
    {
        evoke_block_range current = inventory->ranges[index];
        uint64_t current_end =
            (uint64_t) current.start_block + current.block_count;

        if (current_end > published_block_high_watermark)
        {
            return EVOKE_ERR_RANGE;
        }
        if (unique_count > 0)
        {
            evoke_block_range *previous =
                &inventory->ranges[unique_count - 1];
            uint64_t previous_end =
                (uint64_t) previous->start_block +
                previous->block_count;

            if (current.start_block == previous->start_block &&
                current.block_count == previous->block_count)
            {
                continue;
            }
            if (current.start_block < previous_end)
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        inventory->ranges[unique_count++] = current;
    }

    for (size_t index = 0; index < unique_count; index++)
    {
        evoke_block_range current = inventory->ranges[index];

        if (output_count > 0)
        {
            evoke_block_range *previous =
                &inventory->ranges[output_count - 1];
            uint64_t previous_end =
                (uint64_t) previous->start_block +
                previous->block_count;

            if (current.start_block == previous_end)
            {
                previous->block_count += current.block_count;
                continue;
            }
        }
        inventory->ranges[output_count++] = current;
    }

    for (size_t index = 0; index < output_count; index++)
    {
        reachable_blocks += inventory->ranges[index].block_count;
    }
    if (reachable_blocks > published_block_high_watermark)
    {
        return EVOKE_ERR_FORMAT;
    }
    inventory->range_count = output_count;
    inventory->reachable_block_count = reachable_blocks;
    inventory->interior_unreachable_block_count =
        (uint64_t) published_block_high_watermark - reachable_blocks;
    inventory->highest_reachable_block_exclusive =
        inventory->ranges[output_count - 1].start_block +
        inventory->ranges[output_count - 1].block_count;
    inventory->published_block_high_watermark =
        published_block_high_watermark;
    inventory->finalized = true;
    return EVOKE_OK;
}

void
evoke_block_range_allocator_init(evoke_block_range_allocator *allocator)
{
    if (allocator != NULL)
    {
        memset(allocator, 0, sizeof(*allocator));
    }
}

void
evoke_block_range_allocator_free(evoke_block_range_allocator *allocator)
{
    if (allocator == NULL)
    {
        return;
    }
    free(allocator->ranges);
    memset(allocator, 0, sizeof(*allocator));
}

evoke_status
evoke_block_range_allocator_build(
    const evoke_block_range_inventory *inventory,
    evoke_block_range_allocator *allocator
)
{
    uint32_t cursor = 0;
    size_t gap_count = 0;
    uint64_t available_blocks = 0;

    if (inventory == NULL || allocator == NULL ||
        !inventory->finalized || allocator->initialized ||
        inventory->range_count == 0 ||
        inventory->ranges[0].start_block != 0 ||
        inventory->ranges[0].block_count == 0)
    {
        return EVOKE_ERR_INVALID;
    }
    if (inventory->range_count >=
        SIZE_MAX / sizeof(*allocator->ranges))
    {
        return EVOKE_ERR_RANGE;
    }
    allocator->ranges = calloc(
        inventory->range_count,
        sizeof(*allocator->ranges)
    );
    if (allocator->ranges == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    for (size_t index = 0; index < inventory->range_count; index++)
    {
        const evoke_block_range *live = &inventory->ranges[index];
        uint64_t live_end =
            (uint64_t) live->start_block + live->block_count;

        if (live->start_block < cursor ||
            live_end > inventory->published_block_high_watermark)
        {
            evoke_block_range_allocator_free(allocator);
            return EVOKE_ERR_FORMAT;
        }
        if (live->start_block > cursor)
        {
            uint32_t gap_blocks = live->start_block - cursor;

            allocator->ranges[gap_count].start_block = cursor;
            allocator->ranges[gap_count].block_count = gap_blocks;
            available_blocks += gap_blocks;
            gap_count++;
        }
        cursor = (uint32_t) live_end;
    }
    if (cursor < inventory->published_block_high_watermark)
    {
        uint32_t gap_blocks =
            inventory->published_block_high_watermark - cursor;

        allocator->ranges[gap_count].start_block = cursor;
        allocator->ranges[gap_count].block_count = gap_blocks;
        available_blocks += gap_blocks;
        gap_count++;
    }
    if (available_blocks !=
        inventory->interior_unreachable_block_count)
    {
        evoke_block_range_allocator_free(allocator);
        return EVOKE_ERR_FORMAT;
    }
    allocator->range_count = gap_count;
    allocator->available_block_count = available_blocks;
    allocator->initialized = true;
    return EVOKE_OK;
}

evoke_status
evoke_block_range_allocator_build_free(
    const evoke_block_range *ranges,
    size_t range_count,
    uint32_t published_block_high_watermark,
    evoke_block_range_allocator *allocator
)
{
    uint64_t available_blocks = 0;

    if (ranges == NULL || range_count == 0 ||
        published_block_high_watermark <= 1 || allocator == NULL ||
        allocator->initialized ||
        range_count > SIZE_MAX / sizeof(*allocator->ranges))
    {
        return EVOKE_ERR_INVALID;
    }
    allocator->ranges = calloc(
        range_count,
        sizeof(*allocator->ranges)
    );
    if (allocator->ranges == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    for (size_t index = 0; index < range_count; index++)
    {
        const evoke_block_range *range = &ranges[index];
        uint64_t range_end =
            (uint64_t) range->start_block + range->block_count;

        if (range->start_block == 0 || range->block_count == 0 ||
            range_end > published_block_high_watermark ||
            (index > 0 &&
             (uint64_t) ranges[index - 1].start_block +
                 ranges[index - 1].block_count >= range->start_block) ||
            available_blocks > UINT64_MAX - range->block_count)
        {
            evoke_block_range_allocator_free(allocator);
            return EVOKE_ERR_FORMAT;
        }
        allocator->ranges[index] = *range;
        available_blocks += range->block_count;
    }
    allocator->range_count = range_count;
    allocator->available_block_count = available_blocks;
    allocator->initialized = true;
    return EVOKE_OK;
}

bool
evoke_block_range_allocator_allocate_best_fit(
    evoke_block_range_allocator *allocator,
    uint32_t block_count,
    uint32_t *start_block_out
)
{
    size_t selected = SIZE_MAX;

    if (allocator == NULL || !allocator->initialized ||
        block_count == 0 || start_block_out == NULL)
    {
        return false;
    }
    for (size_t index = 0; index < allocator->range_count; index++)
    {
        const evoke_block_range *range = &allocator->ranges[index];

        if (range->block_count < block_count)
        {
            continue;
        }
        if (selected == SIZE_MAX ||
            range->block_count < allocator->ranges[selected].block_count)
        {
            selected = index;
        }
    }
    if (selected == SIZE_MAX)
    {
        return false;
    }

    *start_block_out = allocator->ranges[selected].start_block;
    allocator->ranges[selected].start_block += block_count;
    allocator->ranges[selected].block_count -= block_count;
    allocator->available_block_count -= block_count;
    allocator->allocated_block_count += block_count;
    if (allocator->ranges[selected].block_count == 0)
    {
        if (selected + 1 < allocator->range_count)
        {
            memmove(
                &allocator->ranges[selected],
                &allocator->ranges[selected + 1],
                (allocator->range_count - selected - 1) *
                    sizeof(*allocator->ranges)
            );
        }
        allocator->range_count--;
    }
    return true;
}
