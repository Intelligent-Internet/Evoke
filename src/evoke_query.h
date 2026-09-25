#ifndef EVOKE_QUERY_H
#define EVOKE_QUERY_H

#include <stdbool.h>
#include <stddef.h>

#include "evoke_core.h"

#define EVOKE_QUERY_MAX_TOKENS 1024U
#define EVOKE_QUERY_MAX_NESTING 128U

typedef enum evoke_query_occur
{
    EVOKE_QUERY_SHOULD = 0,
    EVOKE_QUERY_MUST = 1,
    EVOKE_QUERY_MUST_NOT = 2
} evoke_query_occur;

typedef enum evoke_query_term_kind
{
    EVOKE_QUERY_TERM = 0,
    EVOKE_QUERY_PREFIX = 1,
    EVOKE_QUERY_PHRASE = 2
} evoke_query_term_kind;

typedef enum evoke_query_mode
{
    EVOKE_QUERY_MODE_CLASSIC = 0,
    EVOKE_QUERY_MODE_BOOLEAN_AST = 1
} evoke_query_mode;

typedef struct evoke_query_term
{
    evoke_query_term_kind kind;
    evoke_query_occur occur;
    char **tokens;
    size_t *token_lens;
    size_t len;
} evoke_query_term;

typedef enum evoke_query_node_kind
{
    EVOKE_QUERY_NODE_TERM = 0,
    EVOKE_QUERY_NODE_AND = 1,
    EVOKE_QUERY_NODE_OR = 2,
    EVOKE_QUERY_NODE_NOT = 3
} evoke_query_node_kind;

typedef struct evoke_query_node
{
    evoke_query_node_kind kind;
    size_t term_index;
    struct evoke_query_node *left;
    struct evoke_query_node *right;
} evoke_query_node;

typedef struct evoke_query
{
    evoke_query_mode mode;
    evoke_query_term *terms;
    size_t len;
    evoke_query_node *root;
} evoke_query;

void evoke_query_init(evoke_query *query);
void evoke_query_free(evoke_query *query);

bool evoke_query_is_simple_term_query(const evoke_query *query);
bool evoke_query_uses_boolean_ast(const evoke_query *query);
bool evoke_query_has_phrase(const evoke_query *query);

void evoke_query_rewrite_terms(
    evoke_query *query,
    const size_t *old_to_new,
    size_t old_len
);

evoke_status evoke_parse_query_string(
    const char *input,
    evoke_query *query_out
);

#endif
