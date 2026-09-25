#ifndef II42_SEMANTIC_H
#define II42_SEMANTIC_H

#include "postgres.h"

#define II42_CHECKOUT_SIGNATURE_HEX_LEN 64
#define II42_AM_SEMANTIC_SIGNATURE_LEN 32
/*
 * The persisted checkout manifest still uses the evoke_model_v1 format label.
 * That label is data-format identity, not an internal or public SQL API.
 */

void ii42_checkout_signature(
    const char *model_path,
    char signature_out[II42_CHECKOUT_SIGNATURE_HEX_LEN + 1]
);

void ii42_checkout_unified_contract(
    const char *model_path,
    uint32 *lexical_dims_out,
    uint32 *total_dims_out,
    float4 *semantic_budget_ratio_out
);

void ii42_unified_require_shared_runtime(void);

#endif
