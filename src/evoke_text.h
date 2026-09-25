#ifndef EVOKE_TEXT_H
#define EVOKE_TEXT_H

#include <stdbool.h>
#include <stddef.h>

#include "evoke_core.h"
#include "evoke_query.h"

typedef struct evoke_text_options
{
    bool lowercase;
    bool stem_english;
    bool fold_diacritics;
    const char * const *stopwords;
    size_t num_stopwords;
} evoke_text_options;

void evoke_text_options_init(evoke_text_options *options);

evoke_status evoke_normalize_token(
    const char *input,
    const evoke_text_options *options,
    char **token_out,
    bool *keep_out
);

evoke_status evoke_normalize_query(
    evoke_query *query,
    const evoke_text_options *options
);

evoke_status evoke_tokenize_text(
    const char *input,
    const evoke_text_options *options,
    char ***tokens_out,
    size_t *len_out
);

void evoke_text_tokens_free(char **tokens, size_t len);

#endif
