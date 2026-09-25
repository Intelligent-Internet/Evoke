#include "postgres.h"

#include <float.h>
#include <math.h>

#include "access/reloptions.h"
#include "catalog/pg_class.h"
#include "catalog/pg_type_d.h"
#include "commands/defrem.h"
#include "storage/lockdefs.h"
#include "utils/array.h"
#include "utils/syscache.h"

#include "evoke_am_meta.h"
#include "evoke_am_options.h"
#include "evoke_core.h"

typedef struct evoke_am_options
{
    int32 varlena_header_;
    int method;
    int idf_method;
    int consistency;
    int auto_preload;
    double k1;
    double b;
    double delta;
    bool create_empty_token;
    bool sae;
    bool text_lowercase;
    bool text_stem_english;
    bool text_fold_diacritics;
    bool field_aware;
    int text_stopwords;
    int runtime_precision;
    int semantic_impact_precision;
    double semantic_alpha_mass;
    int model;
    int model_path;
    int atom_space;
    int scoring_profile;
} evoke_am_options;

static relopt_kind evoke_relopt_kind = RELOPT_KIND_LOCAL;
static bool evoke_relopts_initialized = false;

int
evoke_am_index_natts(Relation index_relation)
{
    if (index_relation == NULL || index_relation->rd_index == NULL)
    {
        ereport(ERROR, (errmsg("evoke index metadata is unavailable")));
    }
    return index_relation->rd_index->indnkeyatts;
}

int
evoke_am_index_total_natts(Relation index_relation)
{
    if (index_relation == NULL || index_relation->rd_index == NULL)
    {
        ereport(ERROR, (errmsg("evoke index metadata is unavailable")));
    }
    return index_relation->rd_index->indnatts;
}

int
evoke_am_index_scope_natts(Relation index_relation)
{
    return evoke_am_index_total_natts(index_relation) -
        evoke_am_index_natts(index_relation);
}

bool
evoke_am_is_multicol_index(Relation index_relation)
{
    return evoke_am_index_natts(index_relation) > 1;
}

Oid
evoke_am_source_type(Relation index_relation)
{
    int natts;
    Oid source_type;
    int i;

    natts = evoke_am_index_natts(index_relation);
    if (natts <= 0)
    {
        ereport(ERROR, (errmsg("evoke requires at least one indexed column")));
    }

    source_type = TupleDescAttr(index_relation->rd_att, 0)->atttypid;
    for (i = 1; i < natts; i++)
    {
        Oid column_type = TupleDescAttr(index_relation->rd_att, i)->atttypid;

        if (column_type != source_type)
        {
            ereport(
                ERROR,
                (
                    errmsg(
                        "evoke multicolumn indexes require matching column types"
                    )
                )
            );
        }
    }

    return source_type;
}

void
evoke_am_validate_source_type(Oid source_type)
{
    if (source_type != INT4ARRAYOID &&
        source_type != TEXTARRAYOID &&
        source_type != VARCHARARRAYOID &&
        source_type != TEXTOID &&
        source_type != VARCHAROID)
    {
        ereport(
            ERROR,
            (
                errmsg(
                    "evoke indexes require int4[], text[], varchar[], text, or varchar input"
                )
            )
        );
    }
}

bool
evoke_am_source_type_is_text_array(Oid source_type)
{
    return source_type == TEXTARRAYOID || source_type == VARCHARARRAYOID;
}

bool
evoke_am_source_type_is_scalar_text(Oid source_type)
{
    return source_type == TEXTOID || source_type == VARCHAROID;
}

bool
evoke_am_source_type_is_textlike(Oid source_type)
{
    return evoke_am_source_type_is_text_array(source_type) ||
        evoke_am_source_type_is_scalar_text(source_type);
}

Oid
evoke_am_require_textlike_source_type(
    Relation index_relation,
    const char *context
)
{
    Oid source_type = evoke_am_source_type(index_relation);

    if (!evoke_am_source_type_is_textlike(source_type))
    {
        ereport(ERROR, (errmsg("%s", context)));
    }

    return source_type;
}

evoke_am_build_mode
evoke_am_build_mode_from_source_type(Oid source_type, int natts)
{
    if (source_type == INT4ARRAYOID)
    {
        if (natts != 1)
        {
            ereport(
                ERROR,
                (
                    errmsg(
                        "evoke multicolumn fusion does not support int4[] columns"
                    )
                )
            );
        }
        return EVOKE_AM_BUILD_MODE_IDS;
    }

    if (evoke_am_source_type_is_text_array(source_type))
    {
        if (natts == 1)
        {
            return EVOKE_AM_BUILD_MODE_TEXT_ARRAY_SINGLE;
        }
        return EVOKE_AM_BUILD_MODE_TEXT_ARRAY_MULTI;
    }

    if (natts == 1)
    {
        return EVOKE_AM_BUILD_MODE_SCALAR_SINGLE;
    }

    return EVOKE_AM_BUILD_MODE_SCALAR_MULTI;
}

const char *
evoke_am_consistency_name(int consistency)
{
    switch (consistency)
    {
        case EVOKE_AM_CONSISTENCY_REALTIME:
            return "realtime";
        case EVOKE_AM_CONSISTENCY_EVENTUAL:
            return "eventual";
        case EVOKE_AM_CONSISTENCY_MANUAL:
            return "manual";
        default:
            return "unknown";
    }
}

const char *
evoke_am_runtime_precision_name(int precision)
{
    switch (precision)
    {
        case EVOKE_AM_RUNTIME_PRECISION_FP16:
            return "fp16";
        case EVOKE_AM_RUNTIME_PRECISION_FP32:
            return "fp32";
        default:
            return "unknown";
    }
}

const char *
evoke_am_semantic_impact_precision_name(int precision)
{
    switch (precision)
    {
        case EVOKE_SEMANTIC_IMPACT_PRECISION_F32:
            return "f32";
        case EVOKE_SEMANTIC_IMPACT_PRECISION_FP16:
            return "fp16";
        case EVOKE_SEMANTIC_IMPACT_PRECISION_U8:
            return "u8";
        default:
            return "unknown";
    }
}

void
evoke_am_get_policy_recommendation(
    Relation index_relation,
    const char *profile,
    evoke_am_policy_recommendation *recommendation
)
{
    evoke_am_meta_page meta;
    evoke_am_convergent_mutation_debt debt = {0};
    bool sae_enabled;

    memset(recommendation, 0, sizeof(*recommendation));
    evoke_am_read_meta(index_relation, &meta);
    evoke_am_require_convergent_segment_storage(&meta);
    recommendation->docs = meta.num_docs;
    evoke_am_convergent_mutation_debt_read(
        index_relation,
        &meta,
        &debt
    );
    recommendation->pending_total =
        (uint64) debt.upserts + (uint64) debt.retirements;
    recommendation->refresh_now =
        (meta.flags & EVOKE_AM_FLAG_STALE) != 0;
    sae_enabled = evoke_am_sae_enabled(index_relation);

    if (sae_enabled)
    {
        if (profile == NULL || *profile == '\0' ||
            pg_strcasecmp(profile, "balanced") == 0 ||
            pg_strcasecmp(profile, "query_first") == 0 ||
            pg_strcasecmp(profile, "write_tolerant_query_first") == 0 ||
            pg_strcasecmp(profile, "write_first") == 0)
        {
            recommendation->profile = "balanced";
            recommendation->recommended_options =
                "WITH (consistency = 'eventual')";
            recommendation->recommended_consistency = "eventual";
            recommendation->reason =
                "SAE always publishes lexical evidence first and completes "
                "semantic postings through bounded page-native maintenance";
            recommendation->confidence = "high";
            recommendation->matches_current =
                evoke_am_get_consistency(index_relation) ==
                EVOKE_AM_CONSISTENCY_EVENTUAL;
            return;
        }

        ereport(
            ERROR,
            (
                errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                errmsg(
                    "maintenance recommendation profile \"%s\" applies "
                    "only to sae=false indexes",
                    profile
                ),
                errhint(
                    "Use the \"balanced\" profile for SAE indexes."
                )
            )
        );
        pg_unreachable();
    }

    if (profile == NULL || *profile == '\0' ||
        pg_strcasecmp(profile, "balanced") == 0 ||
        pg_strcasecmp(profile, "query_first") == 0)
    {
        recommendation->profile = pg_strcasecmp(profile, "query_first") == 0
            ? "query_first"
            : "balanced";
        recommendation->recommended_options =
            "WITH (consistency = 'realtime')";
        recommendation->recommended_consistency = "realtime";
        recommendation->reason =
            "BM25 realtime keeps committed lexical evidence query-visible "
            "while bounded page-native maintenance converges structure";
        recommendation->confidence = "high";
        recommendation->matches_current =
            evoke_am_get_consistency(index_relation) ==
            EVOKE_AM_CONSISTENCY_REALTIME;
    }
    else if (pg_strcasecmp(profile, "write_tolerant_query_first") == 0 ||
             pg_strcasecmp(profile, "write_first") == 0)
    {
        recommendation->profile = "write_tolerant_query_first";
        recommendation->recommended_options =
            "WITH (consistency = 'eventual')";
        recommendation->recommended_consistency = "eventual";
        recommendation->reason =
            "eventual consistency minimizes foreground write and query "
            "maintenance while converging automatically";
        recommendation->confidence = "medium";
        recommendation->matches_current =
            evoke_am_get_consistency(index_relation) ==
            EVOKE_AM_CONSISTENCY_EVENTUAL;
    }
    else
    {
        ereport(
            ERROR,
            (errmsg("unsupported maintenance recommendation profile: %s",
                    profile))
        );
        pg_unreachable();
    }
}

static const relopt_enum_elt_def evoke_method_members[] = {
    {"robertson", EVOKE_METHOD_ROBERTSON},
    {"lucene", EVOKE_METHOD_LUCENE},
    {"atire", EVOKE_METHOD_ATIRE},
    {"bm25l", EVOKE_METHOD_BM25L},
    {"bm25+", EVOKE_METHOD_BM25PLUS},
    {NULL, 0}
};

static const relopt_enum_elt_def evoke_consistency_members[] = {
    {"realtime", EVOKE_AM_CONSISTENCY_REALTIME},
    {"eventual", EVOKE_AM_CONSISTENCY_EVENTUAL},
    {"manual", EVOKE_AM_CONSISTENCY_MANUAL},
    {NULL, 0}
};

static const relopt_enum_elt_def evoke_runtime_precision_members[] = {
    {"fp16", EVOKE_AM_RUNTIME_PRECISION_FP16},
    {"fp32", EVOKE_AM_RUNTIME_PRECISION_FP32},
    {NULL, 0}
};

static const relopt_enum_elt_def evoke_semantic_impact_precision_members[] = {
    {"f32", EVOKE_SEMANTIC_IMPACT_PRECISION_F32},
    {"fp16", EVOKE_SEMANTIC_IMPACT_PRECISION_FP16},
    {"u8", EVOKE_SEMANTIC_IMPACT_PRECISION_U8},
    {NULL, 0}
};

static const relopt_parse_elt evoke_relopt_elems[] = {
    {"method", RELOPT_TYPE_ENUM, offsetof(evoke_am_options, method)},
    {"idf_method", RELOPT_TYPE_ENUM, offsetof(evoke_am_options, idf_method)},
    {
        "consistency",
        RELOPT_TYPE_ENUM,
        offsetof(evoke_am_options, consistency)
    },
    {
        "auto_preload",
        RELOPT_TYPE_INT,
        offsetof(evoke_am_options, auto_preload)
    },
    {"k1", RELOPT_TYPE_REAL, offsetof(evoke_am_options, k1)},
    {"b", RELOPT_TYPE_REAL, offsetof(evoke_am_options, b)},
    {"delta", RELOPT_TYPE_REAL, offsetof(evoke_am_options, delta)},
    {
        "create_empty_token",
        RELOPT_TYPE_BOOL,
        offsetof(evoke_am_options, create_empty_token)
    },
    {
        "sae",
        RELOPT_TYPE_BOOL,
        offsetof(evoke_am_options, sae)
    },
    {
        "text_lowercase",
        RELOPT_TYPE_BOOL,
        offsetof(evoke_am_options, text_lowercase)
    },
    {
        "text_stem_english",
        RELOPT_TYPE_BOOL,
        offsetof(evoke_am_options, text_stem_english)
    },
    {
        "text_fold_diacritics",
        RELOPT_TYPE_BOOL,
        offsetof(evoke_am_options, text_fold_diacritics)
    },
    {
        "field_aware",
        RELOPT_TYPE_BOOL,
        offsetof(evoke_am_options, field_aware)
    },
    {
        "text_stopwords",
        RELOPT_TYPE_STRING,
        offsetof(evoke_am_options, text_stopwords)
    },
    {
        "runtime_precision",
        RELOPT_TYPE_ENUM,
        offsetof(evoke_am_options, runtime_precision)
    },
    {
        "semantic_impact_precision",
        RELOPT_TYPE_ENUM,
        offsetof(evoke_am_options, semantic_impact_precision)
    },
    {
        "semantic_alpha_mass",
        RELOPT_TYPE_REAL,
        offsetof(evoke_am_options, semantic_alpha_mass)
    },
    {
        "model",
        RELOPT_TYPE_STRING,
        offsetof(evoke_am_options, model)
    },
    {
        "model_path",
        RELOPT_TYPE_STRING,
        offsetof(evoke_am_options, model_path)
    },
    {
        "atom_space",
        RELOPT_TYPE_STRING,
        offsetof(evoke_am_options, atom_space)
    },
    {
        "scoring_profile",
        RELOPT_TYPE_STRING,
        offsetof(evoke_am_options, scoring_profile)
    }
};

static bool
evoke_am_reloptions_have_name(List *reloption_defs, const char *name)
{
    ListCell *cell;

    foreach (cell, reloption_defs)
    {
        DefElem *defel = (DefElem *) lfirst(cell);

        if (defel != NULL &&
            defel->defname != NULL &&
            strcmp(defel->defname, name) == 0)
        {
            return true;
        }
    }
    return false;
}

static void
evoke_am_validate_policy_reloptions(
    Datum reloptions,
    const evoke_am_options *options
)
{
    List *reloption_defs;
    static const char *sae_option_names[] = {
        "runtime_precision",
        "semantic_impact_precision",
        "semantic_alpha_mass",
        "model",
        "model_path",
        "atom_space",
        "scoring_profile"
    };
    static const char *bm25_option_names[] = {
        "method",
        "idf_method",
        "k1",
        "b",
        "delta",
        "create_empty_token",
        "text_lowercase",
        "text_stopwords",
        "text_stem_english",
        "text_fold_diacritics"
    };
    bool sae_is_explicit;
    bool has_sae_options = false;
    int i;

    if (DatumGetPointer(reloptions) == NULL || options == NULL)
    {
        return;
    }

    reloption_defs = untransformRelOptions(reloptions);
    sae_is_explicit = evoke_am_reloptions_have_name(reloption_defs, "sae");
    for (i = 0; i < lengthof(sae_option_names); i++)
    {
        if (evoke_am_reloptions_have_name(
                reloption_defs,
                sae_option_names[i]
            ))
        {
            has_sae_options = true;
            break;
        }
    }

    if (has_sae_options && (!sae_is_explicit || !options->sae))
    {
        ereport(
            ERROR,
            (
                errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                errmsg(
                    "evoke SAE options require reloption \"sae=true\""
                ),
                errhint(
                    "Remove the SAE options or explicitly set \"sae=true\"."
                )
            )
        );
    }
    if (options->sae)
    {
        if (options->consistency != EVOKE_AM_CONSISTENCY_EVENTUAL)
        {
            ereport(
                ERROR,
                (
                    errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                    errmsg(
                        "evoke SAE indexes require consistency = 'eventual'"
                    ),
                    errdetail(
                        "Foreground SAE writes publish lexical evidence "
                        "without document inference; semantic completion is "
                        "owned by background maintenance."
                    ),
                    errhint(
                        "Omit consistency or set consistency = 'eventual'."
                    )
                )
            );
        }
        for (i = 0; i < lengthof(bm25_option_names); i++)
        {
            if (evoke_am_reloptions_have_name(
                    reloption_defs,
                    bm25_option_names[i]
                ))
            {
                ereport(
                    ERROR,
                    (
                        errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                        errmsg(
                            "evoke reloption \"%s\" is not valid with "
                            "sae=true",
                            bm25_option_names[i]
                        ),
                        errhint(
                            "Configure semantic normalization and scoring "
                            "in the model checkout manifest."
                        )
                    )
                );
            }
        }
    }
    list_free_deep(reloption_defs);
}

static void
evoke_am_apply_reloption_defaults_and_flags(
    Datum reloptions,
    evoke_am_options *options
)
{
    List *reloption_defs;
    bool consistency_is_set;

    if (options == NULL)
    {
        return;
    }

    if (DatumGetPointer(reloptions) == NULL)
    {
        return;
    }

    reloption_defs = untransformRelOptions(reloptions);
    consistency_is_set = evoke_am_reloptions_have_name(
        reloption_defs,
        "consistency"
    );
    if (options->sae && !consistency_is_set)
    {
        options->consistency = EVOKE_AM_CONSISTENCY_EVENTUAL;
    }
    list_free_deep(reloption_defs);
}

void
evoke_init_reloptions(void)
{
    if (evoke_relopts_initialized)
    {
        return;
    }

    evoke_relopt_kind = add_reloption_kind();
    add_enum_reloption(
        evoke_relopt_kind,
        "method",
        "BM25 variant used when building the index",
        (relopt_enum_elt_def *) evoke_method_members,
        EVOKE_METHOD_LUCENE,
        NULL,
        AccessExclusiveLock
    );
    add_enum_reloption(
        evoke_relopt_kind,
        "idf_method",
        "BM25 IDF variant used when building the index",
        (relopt_enum_elt_def *) evoke_method_members,
        EVOKE_METHOD_LUCENE,
        NULL,
        AccessExclusiveLock
    );
    add_enum_reloption(
        evoke_relopt_kind,
        "consistency",
        "Query consistency model for pending index maintenance",
        (relopt_enum_elt_def *) evoke_consistency_members,
        EVOKE_AM_CONSISTENCY_REALTIME,
        NULL,
        AccessExclusiveLock
    );
    add_int_reloption(
        evoke_relopt_kind,
        "auto_preload",
        "Shared-preload auto warmup priority; zero disables auto preload",
        0,
        0,
        INT_MAX,
        AccessExclusiveLock
    );
    add_real_reloption(
        evoke_relopt_kind,
        "k1",
        "BM25 k1 parameter",
        1.5,
        0.0,
        FLT_MAX,
        AccessExclusiveLock
    );
    add_real_reloption(
        evoke_relopt_kind,
        "b",
        "BM25 b parameter",
        0.75,
        0.0,
        1.0,
        AccessExclusiveLock
    );
    add_real_reloption(
        evoke_relopt_kind,
        "delta",
        "BM25 delta parameter",
        0.5,
        0.0,
        FLT_MAX,
        AccessExclusiveLock
    );
    add_bool_reloption(
        evoke_relopt_kind,
        "create_empty_token",
        "Create an empty token for integer-token indexes",
        true,
        AccessExclusiveLock
    );
    add_bool_reloption(
        evoke_relopt_kind,
        "sae",
        "Enable the model-backed unified BM25+SAE index contract",
        false,
        AccessExclusiveLock
    );
    add_bool_reloption(
        evoke_relopt_kind,
        "text_lowercase",
        "Lowercase raw text/varchar inputs before token indexing",
        true,
        AccessExclusiveLock
    );
    add_bool_reloption(
        evoke_relopt_kind,
        "text_stem_english",
        "Apply English Porter stemming to raw text/varchar inputs",
        false,
        AccessExclusiveLock
    );
    add_bool_reloption(
        evoke_relopt_kind,
        "text_fold_diacritics",
        "Fold diacritics in raw text/varchar inputs before indexing",
        false,
        AccessExclusiveLock
    );
    add_bool_reloption(
        evoke_relopt_kind,
        "field_aware",
        "Preserve indexed column identity inside multicolumn text-like indexes",
        false,
        AccessExclusiveLock
    );
    add_enum_reloption(
        evoke_relopt_kind,
        "runtime_precision",
        "Model runtime precision for SAE index build and query encoding",
        (relopt_enum_elt_def *) evoke_runtime_precision_members,
        EVOKE_AM_RUNTIME_PRECISION_FP16,
        NULL,
        AccessExclusiveLock
    );
    add_enum_reloption(
        evoke_relopt_kind,
        "semantic_impact_precision",
        "Stored semantic posting impact precision",
        (relopt_enum_elt_def *) evoke_semantic_impact_precision_members,
        EVOKE_SEMANTIC_IMPACT_PRECISION_F32,
        NULL,
        AccessExclusiveLock
    );
    add_real_reloption(
        evoke_relopt_kind,
        "semantic_alpha_mass",
        "Fraction of positive semantic impact mass retained per field",
        1.0,
        0.01,
        1.0,
        AccessExclusiveLock
    );
    add_string_reloption(
        evoke_relopt_kind,
        "text_stopwords",
        "Comma-separated stopwords for raw text/varchar inputs",
        NULL,
        NULL,
        AccessExclusiveLock
    );
    add_string_reloption(
        evoke_relopt_kind,
        "model",
        "Optional manifest model-id override for this SAE index",
        NULL,
        NULL,
        AccessExclusiveLock
    );
    add_string_reloption(
        evoke_relopt_kind,
        "model_path",
        "Optional server-local SAE checkout override for this index",
        NULL,
        NULL,
        AccessExclusiveLock
    );
    add_string_reloption(
        evoke_relopt_kind,
        "atom_space",
        "Optional manifest atom-space override for this SAE index",
        NULL,
        NULL,
        AccessExclusiveLock
    );
    add_string_reloption(
        evoke_relopt_kind,
        "scoring_profile",
        "Optional manifest scoring-profile override for this SAE index",
        NULL,
        NULL,
        AccessExclusiveLock
    );

    evoke_relopts_initialized = true;
}

bytea *
evoke_amoptions(Datum reloptions, bool validate)
{
    evoke_am_options *options;

    evoke_init_reloptions();
    options = (evoke_am_options *) build_reloptions(
        reloptions,
        validate,
        evoke_relopt_kind,
        sizeof(evoke_am_options),
        evoke_relopt_elems,
        lengthof(evoke_relopt_elems)
    );
    evoke_am_apply_reloption_defaults_and_flags(reloptions, options);
    if (validate && options != NULL)
    {
        evoke_am_validate_policy_reloptions(reloptions, options);
    }
    return (bytea *) options;
}

static bool
evoke_am_relation_has_explicit_reloptions(Relation relation)
{
    HeapTuple tuple;
    Datum datum;
    bool isnull = true;
    bool has_options = false;

    tuple = SearchSysCache1(RELOID, ObjectIdGetDatum(RelationGetRelid(relation)));
    if (!HeapTupleIsValid(tuple))
    {
        return false;
    }

    datum = SysCacheGetAttr(RELOID, tuple, Anum_pg_class_reloptions, &isnull);
    if (!isnull)
    {
        ArrayType *options = DatumGetArrayTypeP(datum);

        has_options = ARR_NDIM(options) > 0 && ArrayGetNItems(
            ARR_NDIM(options),
            ARR_DIMS(options)
        ) > 0;
    }

    ReleaseSysCache(tuple);
    return has_options;
}

void
evoke_am_read_params(
    Relation index_relation,
    evoke_params *params_out,
    bool *create_empty_token_out
)
{
    evoke_am_options *options;

    Assert(params_out != NULL);
    Assert(create_empty_token_out != NULL);

    params_out->method = EVOKE_METHOD_LUCENE;
    params_out->idf_method = EVOKE_METHOD_LUCENE;
    params_out->k1 = 1.5f;
    params_out->b = 0.75f;
    params_out->delta = 0.5f;
    *create_empty_token_out = true;

    options = (evoke_am_options *) index_relation->rd_options;
    if (options == NULL ||
        !evoke_am_relation_has_explicit_reloptions(index_relation))
    {
        return;
    }

    params_out->method = (evoke_method) options->method;
    params_out->idf_method = (evoke_method) options->idf_method;
    if (!isfinite(options->k1) ||
        !isfinite(options->b) ||
        !isfinite(options->delta) ||
        options->k1 < 0.0 ||
        options->k1 > FLT_MAX ||
        options->b < 0.0 ||
        options->b > 1.0 ||
        options->delta < 0.0 ||
        options->delta > FLT_MAX)
    {
        ereport(
            ERROR,
            (
                errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                errmsg(
                    "evoke BM25 reloptions are outside the supported "
                    "finite range"
                ),
                errdetail(
                    "k1=%g, b=%g, delta=%g.",
                    options->k1,
                    options->b,
                    options->delta
                ),
                errhint(
                    "Use finite k1 and delta values representable as real, "
                    "and set b between 0 and 1."
                )
            )
        );
    }
    params_out->k1 = (float) options->k1;
    params_out->b = (float) options->b;
    params_out->delta = (float) options->delta;
    *create_empty_token_out = options->create_empty_token;
}

bool
evoke_am_field_aware_enabled(Relation index_relation)
{
    evoke_am_options *options;

    options = (evoke_am_options *) index_relation->rd_options;
    if (options == NULL)
    {
        return false;
    }

    return options->field_aware;
}

bool
evoke_am_sae_enabled(Relation index_relation)
{
    evoke_am_options *options;

    options = (evoke_am_options *) index_relation->rd_options;
    return options != NULL && options->sae;
}

int
evoke_am_get_runtime_precision(Relation index_relation)
{
    evoke_am_options *options;

    options = (evoke_am_options *) index_relation->rd_options;
    if (options == NULL)
    {
        return EVOKE_AM_RUNTIME_PRECISION_FP16;
    }

    return options->runtime_precision;
}

evoke_semantic_impact_precision
evoke_am_get_semantic_impact_precision(Relation index_relation)
{
    evoke_am_options *options;

    options = (evoke_am_options *) index_relation->rd_options;
    if (options == NULL)
    {
        return EVOKE_SEMANTIC_IMPACT_PRECISION_F32;
    }
    if (evoke_semantic_bmp_impact_width(
            (evoke_semantic_impact_precision)
                options->semantic_impact_precision
        ) == 0)
    {
        ereport(
            ERROR,
            (errmsg("semantic_impact_precision is unsupported"))
        );
    }
    return (evoke_semantic_impact_precision)
        options->semantic_impact_precision;
}

double
evoke_am_semantic_alpha_mass(Relation index_relation)
{
    evoke_am_options *options;

    options = (evoke_am_options *) index_relation->rd_options;
    if (options == NULL)
    {
        return 1.0;
    }
    if (!isfinite(options->semantic_alpha_mass) ||
        options->semantic_alpha_mass < 0.01 ||
        options->semantic_alpha_mass > 1.0)
    {
        ereport(
            ERROR,
            (
                errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                errmsg("semantic_alpha_mass is outside the supported range"),
                errhint("Set semantic_alpha_mass between 0.01 and 1.0.")
            )
        );
    }
    return options->semantic_alpha_mass;
}

int
evoke_am_get_consistency(Relation index_relation)
{
    evoke_am_options *options;

    options = (evoke_am_options *) index_relation->rd_options;
    if (options == NULL ||
        !evoke_am_relation_has_explicit_reloptions(index_relation))
    {
        return EVOKE_AM_CONSISTENCY_REALTIME;
    }

    return options->consistency;
}

void
evoke_am_validate_relation_policy(Relation index_relation)
{
    if (index_relation == NULL || index_relation->rd_rel == NULL)
    {
        return;
    }
    if (index_relation->rd_rel->relpersistence != RELPERSISTENCE_TEMP)
    {
        return;
    }
    if (evoke_am_sae_enabled(index_relation))
    {
        ereport(
            ERROR,
            (
                errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                errmsg(
                    "temporary evoke indexes do not support SAE"
                ),
                errdetail(
                    "The shared model and generation runtime cannot own "
                    "another backend's temporary relation."
                ),
                errhint(
                    "Use a permanent or unlogged relation for SAE indexes."
                )
            )
        );
    }
    if (evoke_am_get_consistency(index_relation) !=
        EVOKE_AM_CONSISTENCY_MANUAL)
    {
        ereport(
            ERROR,
            (
                errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                errmsg(
                    "temporary evoke indexes require manual consistency"
                ),
                errdetail(
                    "Automatic maintenance workers cannot own another "
                    "backend's temporary relation."
                ),
                errhint(
                    "Set consistency=manual and call "
                    "evoke_index_maintain() from the owning session."
                )
            )
        );
    }
}

int
evoke_am_auto_preload_priority(Relation index_relation)
{
    evoke_am_options *options;

    options = (evoke_am_options *) index_relation->rd_options;
    if (options == NULL)
    {
        return 0;
    }

    return options->auto_preload;
}

const char *
evoke_am_relation_model_path(Relation index_relation)
{
    evoke_am_options *options;

    options = (evoke_am_options *) index_relation->rd_options;
    if (options == NULL || options->model_path == 0)
    {
        return NULL;
    }

    return GET_STRING_RELOPTION(options, model_path);
}

void
evoke_am_read_text_policy(
    Relation index_relation,
    evoke_am_text_policy *policy_out
)
{
    evoke_am_options *options;

    Assert(policy_out != NULL);
    policy_out->lowercase = true;
    policy_out->stem_english = false;
    policy_out->fold_diacritics = false;
    policy_out->raw_stopwords = NULL;

    options = (evoke_am_options *) index_relation->rd_options;
    if (options == NULL ||
        !evoke_am_relation_has_explicit_reloptions(index_relation))
    {
        return;
    }

    policy_out->lowercase = options->text_lowercase;
    policy_out->stem_english = options->text_stem_english;
    policy_out->fold_diacritics = options->text_fold_diacritics;
    policy_out->raw_stopwords = options->text_stopwords == 0
        ? NULL
        : GET_STRING_RELOPTION(options, text_stopwords);
}
