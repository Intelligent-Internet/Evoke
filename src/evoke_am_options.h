#ifndef EVOKE_AM_OPTIONS_H
#define EVOKE_AM_OPTIONS_H

#include "postgres.h"
#include "utils/rel.h"

#include "evoke_core.h"
#include "evoke_semantic_bmp.h"

typedef enum evoke_am_consistency
{
    EVOKE_AM_CONSISTENCY_REALTIME = 0,
    EVOKE_AM_CONSISTENCY_EVENTUAL,
    EVOKE_AM_CONSISTENCY_MANUAL
} evoke_am_consistency;

typedef enum evoke_am_runtime_precision
{
    EVOKE_AM_RUNTIME_PRECISION_FP16 = 0,
    EVOKE_AM_RUNTIME_PRECISION_FP32
} evoke_am_runtime_precision;

typedef enum evoke_am_build_mode
{
    EVOKE_AM_BUILD_MODE_IDS = 0,
    EVOKE_AM_BUILD_MODE_TEXT_ARRAY_SINGLE,
    EVOKE_AM_BUILD_MODE_TEXT_ARRAY_MULTI,
    EVOKE_AM_BUILD_MODE_SCALAR_SINGLE,
    EVOKE_AM_BUILD_MODE_SCALAR_MULTI
} evoke_am_build_mode;

typedef struct evoke_am_text_policy
{
    bool lowercase;
    bool stem_english;
    bool fold_diacritics;
    const char *raw_stopwords;
} evoke_am_text_policy;

typedef struct evoke_am_policy_recommendation
{
    const char *profile;
    const char *confidence;
    const char *recommended_options;
    const char *recommended_consistency;
    bool matches_current;
    bool refresh_now;
    uint32 docs;
    uint64 pending_total;
    const char *reason;
} evoke_am_policy_recommendation;

void evoke_init_reloptions(void);
bytea *evoke_amoptions(Datum reloptions, bool validate);
int evoke_am_index_natts(Relation index_relation);

int evoke_am_index_total_natts(Relation index_relation);

int evoke_am_index_scope_natts(Relation index_relation);
bool evoke_am_is_multicol_index(Relation index_relation);
Oid evoke_am_source_type(Relation index_relation);
void evoke_am_validate_source_type(Oid source_type);
bool evoke_am_source_type_is_text_array(Oid source_type);
bool evoke_am_source_type_is_scalar_text(Oid source_type);
bool evoke_am_source_type_is_textlike(Oid source_type);
Oid evoke_am_require_textlike_source_type(
    Relation index_relation,
    const char *context
);
evoke_am_build_mode evoke_am_build_mode_from_source_type(
    Oid source_type,
    int natts
);
const char *evoke_am_consistency_name(int consistency);
const char *evoke_am_runtime_precision_name(int precision);
const char *evoke_am_semantic_impact_precision_name(int precision);
void evoke_am_get_policy_recommendation(
    Relation index_relation,
    const char *profile,
    evoke_am_policy_recommendation *recommendation
);
int evoke_am_get_consistency(Relation index_relation);
void evoke_am_validate_relation_policy(Relation index_relation);
int evoke_am_auto_preload_priority(Relation index_relation);
bool evoke_am_field_aware_enabled(Relation index_relation);
bool evoke_am_sae_enabled(Relation index_relation);
int evoke_am_get_runtime_precision(Relation index_relation);
evoke_semantic_impact_precision evoke_am_get_semantic_impact_precision(
    Relation index_relation
);
double evoke_am_semantic_alpha_mass(Relation index_relation);
const char *evoke_am_relation_model_path(Relation index_relation);
void evoke_am_read_text_policy(
    Relation index_relation,
    evoke_am_text_policy *policy_out
);
void evoke_am_read_params(
    Relation index_relation,
    evoke_params *params_out,
    bool *create_empty_token_out
);

#endif
