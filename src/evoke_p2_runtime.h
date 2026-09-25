#ifndef EVOKE_P2_RUNTIME_H
#define EVOKE_P2_RUNTIME_H

#include "postgres.h"

typedef struct EvokeP2TokenizedInput
{
    int64 *input_ids;
    int64 *attention_mask;
    int32 token_count;
    int64 shape[2];
} EvokeP2TokenizedInput;

typedef struct EvokeP2TokenizedWindows
{
    EvokeP2TokenizedInput *windows;
    int32 window_count;
    int32 full_token_count;
    int32 window_stride;
} EvokeP2TokenizedWindows;

typedef struct EvokeP2CompiledAtoms
{
    int32 *atom_ids;
    float4 *atom_weights;
    int32 atom_count;
    int32 lexical_atom_count;
    int32 semantic_atom_count;
    double lexical_proxy;
    double semantic_proxy;
    double query_scale;
} EvokeP2CompiledAtoms;

void evoke_p2_tokenize_roberta(
    const char *vocabulary_path,
    const char *merges_path,
    const char *checkout_signature,
    const char *query_text,
    int32 max_length,
    EvokeP2TokenizedInput *output
);

void evoke_p2_tokenize_roberta_windows(
    const char *vocabulary_path,
    const char *merges_path,
    const char *checkout_signature,
    const char *query_text,
    int32 max_length,
    int32 window_stride,
    int32 max_windows,
    EvokeP2TokenizedWindows *output
);

void evoke_p2_tokenized_windows_free(EvokeP2TokenizedWindows *tokenized);

void evoke_p2_tokenizer_cache_clear(void);

void evoke_p2_compile_unified_atoms(
    const char *lexical_vocabulary_path,
    const char *calibration_path,
    const char *checkout_signature,
    const char *query_text,
    const int64 *semantic_ids,
    const float *semantic_weights,
    int32 semantic_count,
    int32 lexical_dims,
    int32 total_dims,
    double global_scale,
    double multiplier,
    double clip_min,
    double clip_max,
    EvokeP2CompiledAtoms *output
);

void evoke_p2_compile_document_semantic_atoms(
    const int64 *semantic_ids,
    const float *semantic_weights,
    int32 semantic_count,
    int32 lexical_dims,
    int32 total_dims,
    EvokeP2CompiledAtoms *output
);

bool evoke_p2_lookup_lexical_atom_id(
    const char *lexical_vocabulary_path,
    const char *calibration_path,
    const char *checkout_signature,
    int32 lexical_dims,
    int32 total_dims,
    const char *token,
    uint32 *atom_id_out
);

void evoke_p2_compiler_cache_clear(void);

#endif
