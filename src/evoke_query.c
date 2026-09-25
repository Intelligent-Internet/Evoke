#include "evoke_query.h"

#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef enum evoke_query_token_kind
{
    EVOKE_QUERY_TOKEN_TERM = 0,
    EVOKE_QUERY_TOKEN_PREFIX = 1,
    EVOKE_QUERY_TOKEN_PHRASE = 2,
    EVOKE_QUERY_TOKEN_PLUS = 3,
    EVOKE_QUERY_TOKEN_MINUS = 4,
    EVOKE_QUERY_TOKEN_AND = 5,
    EVOKE_QUERY_TOKEN_OR = 6,
    EVOKE_QUERY_TOKEN_NOT = 7,
    EVOKE_QUERY_TOKEN_LPAREN = 8,
    EVOKE_QUERY_TOKEN_RPAREN = 9,
    EVOKE_QUERY_TOKEN_EOF = 10
} evoke_query_token_kind;

typedef struct evoke_query_token
{
    evoke_query_token_kind kind;
    evoke_query_term term;
} evoke_query_token;

typedef struct evoke_query_parser
{
    const evoke_query_token *tokens;
    size_t len;
    size_t pos;
    size_t depth;
    evoke_query *query;
} evoke_query_parser;

static void
evoke_query_term_free(evoke_query_term *term)
{
    size_t i;

    if (term == NULL)
    {
        return;
    }

    if (term->tokens != NULL)
    {
        for (i = 0; i < term->len; i++)
        {
            free(term->tokens[i]);
        }
        free(term->tokens);
    }
    free(term->token_lens);

    memset(term, 0, sizeof(*term));
}

static void
evoke_query_node_free(evoke_query_node *node)
{
    if (node == NULL)
    {
        return;
    }

    evoke_query_node_free(node->left);
    evoke_query_node_free(node->right);
    free(node);
}

static evoke_status
evoke_query_term_clone(
    const evoke_query_term *term,
    evoke_query_term *clone_out
)
{
    size_t i;

    memset(clone_out, 0, sizeof(*clone_out));
    if (term == NULL)
    {
        return EVOKE_ERR_INVALID;
    }

    clone_out->kind = term->kind;
    clone_out->occur = term->occur;
    clone_out->len = term->len;
    if (term->len == 0)
    {
        return EVOKE_OK;
    }

    clone_out->tokens = malloc(sizeof(*clone_out->tokens) * term->len);
    clone_out->token_lens = malloc(sizeof(*clone_out->token_lens) * term->len);
    if (clone_out->tokens == NULL || clone_out->token_lens == NULL)
    {
        evoke_query_term_free(clone_out);
        return EVOKE_ERR_NOMEM;
    }

    for (i = 0; i < term->len; i++)
    {
        size_t token_len;

        token_len = term->token_lens != NULL ?
            term->token_lens[i] :
            strlen(term->tokens[i]);
        clone_out->tokens[i] = malloc(token_len + 1);
        if (clone_out->tokens[i] == NULL)
        {
            evoke_query_term_free(clone_out);
            return EVOKE_ERR_NOMEM;
        }
        memcpy(clone_out->tokens[i], term->tokens[i], token_len + 1);
        clone_out->token_lens[i] = token_len;
    }

    return EVOKE_OK;
}

void
evoke_query_init(evoke_query *query)
{
    if (query == NULL)
    {
        return;
    }

    memset(query, 0, sizeof(*query));
    query->mode = EVOKE_QUERY_MODE_CLASSIC;
}

void
evoke_query_free(evoke_query *query)
{
    size_t i;

    if (query == NULL)
    {
        return;
    }

    for (i = 0; i < query->len; i++)
    {
        evoke_query_term_free(&query->terms[i]);
    }
    free(query->terms);
    evoke_query_node_free(query->root);
    memset(query, 0, sizeof(*query));
}

static bool
evoke_query_node_is_simple_term_disjunction(
    const evoke_query *query,
    const evoke_query_node *node
)
{
    const evoke_query_term *term;

    if (query == NULL || node == NULL)
    {
        return false;
    }

    if (node->kind == EVOKE_QUERY_NODE_OR)
    {
        return
            evoke_query_node_is_simple_term_disjunction(
                query,
                node->left
            ) &&
            evoke_query_node_is_simple_term_disjunction(
                query,
                node->right
            );
    }

    if (node->kind != EVOKE_QUERY_NODE_TERM ||
        node->term_index >= query->len)
    {
        return false;
    }

    term = &query->terms[node->term_index];
    return term->kind == EVOKE_QUERY_TERM &&
           term->len == 1 &&
           term->tokens != NULL &&
           term->tokens[0] != NULL;
}

bool
evoke_query_is_simple_term_query(const evoke_query *query)
{
    size_t i;

    if (query == NULL || query->len == 0)
    {
        return false;
    }

    if (query->mode == EVOKE_QUERY_MODE_BOOLEAN_AST)
    {
        return evoke_query_node_is_simple_term_disjunction(
            query,
            query->root
        );
    }

    for (i = 0; i < query->len; i++)
    {
        const evoke_query_term *term = &query->terms[i];

        if (term->kind != EVOKE_QUERY_TERM ||
            term->occur != EVOKE_QUERY_SHOULD ||
            term->len != 1 ||
            term->tokens == NULL ||
            term->tokens[0] == NULL)
        {
            return false;
        }
    }

    return true;
}

bool
evoke_query_uses_boolean_ast(const evoke_query *query)
{
    return query != NULL &&
           query->mode == EVOKE_QUERY_MODE_BOOLEAN_AST &&
           query->root != NULL;
}

bool
evoke_query_has_phrase(const evoke_query *query)
{
    size_t i;

    if (query == NULL)
    {
        return false;
    }

    for (i = 0; i < query->len; i++)
    {
        if (query->terms[i].kind == EVOKE_QUERY_PHRASE)
        {
            return true;
        }
    }

    return false;
}

static evoke_query_node *
evoke_query_node_new(evoke_query_node_kind kind)
{
    evoke_query_node *node;

    node = calloc(1, sizeof(*node));
    if (node == NULL)
    {
        return NULL;
    }

    node->kind = kind;
    node->term_index = SIZE_MAX;
    return node;
}

static evoke_query_node *
evoke_query_node_term(size_t term_index)
{
    evoke_query_node *node;

    node = evoke_query_node_new(EVOKE_QUERY_NODE_TERM);
    if (node == NULL)
    {
        return NULL;
    }

    node->term_index = term_index;
    return node;
}

static evoke_query_node *
evoke_query_node_unary(
    evoke_query_node_kind kind,
    evoke_query_node *child
)
{
    evoke_query_node *node;

    if (child == NULL)
    {
        return NULL;
    }

    node = evoke_query_node_new(kind);
    if (node == NULL)
    {
        evoke_query_node_free(child);
        return NULL;
    }

    node->left = child;
    return node;
}

static evoke_query_node *
evoke_query_node_binary(
    evoke_query_node_kind kind,
    evoke_query_node *left,
    evoke_query_node *right
)
{
    evoke_query_node *node;

    if (left == NULL)
    {
        return right;
    }
    if (right == NULL)
    {
        return left;
    }

    node = evoke_query_node_new(kind);
    if (node == NULL)
    {
        evoke_query_node_free(left);
        evoke_query_node_free(right);
        return NULL;
    }

    node->left = left;
    node->right = right;
    return node;
}

static const char *
evoke_skip_space(const char *cursor)
{
    while (*cursor != '\0' && isspace((unsigned char) *cursor))
    {
        cursor++;
    }

    return cursor;
}

static char *
evoke_copy_span(const char *start, size_t len)
{
    char *copy;

    copy = malloc(len + 1);
    if (copy == NULL)
    {
        return NULL;
    }

    memcpy(copy, start, len);
    copy[len] = '\0';
    return copy;
}

static void
evoke_query_token_free(evoke_query_token *token)
{
    if (token == NULL)
    {
        return;
    }

    evoke_query_term_free(&token->term);
}

static void
evoke_query_tokens_free(
    evoke_query_token *tokens,
    size_t len
)
{
    size_t i;

    if (tokens == NULL)
    {
        return;
    }

    for (i = 0; i < len; i++)
    {
        evoke_query_token_free(&tokens[i]);
    }
    free(tokens);
}

static evoke_status
evoke_append_token(
    evoke_query_token **tokens,
    size_t *len,
    const evoke_query_token *token
)
{
    evoke_query_token *resized;

    resized = realloc(*tokens, sizeof(*resized) * (*len + 1));
    if (resized == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    *tokens = resized;
    (*tokens)[*len] = *token;
    (*len)++;
    return EVOKE_OK;
}

static evoke_status
evoke_split_phrase(
    const char *start,
    size_t len,
    char ***tokens_out,
    size_t **token_lens_out,
    size_t *len_out
)
{
    const char *cursor = start;
    const char *end = start + len;
    char **tokens = NULL;
    size_t *token_lens = NULL;
    size_t token_len = 0;

    *tokens_out = NULL;
    *token_lens_out = NULL;
    *len_out = 0;

    while (cursor < end)
    {
        const char *term_start;
        size_t span_len;
        char **new_tokens;
        size_t *new_token_lens;
        char *token;

        while (cursor < end && isspace((unsigned char) *cursor))
        {
            cursor++;
        }
        if (cursor >= end)
        {
            break;
        }

        term_start = cursor;
        while (cursor < end && !isspace((unsigned char) *cursor))
        {
            cursor++;
        }
        span_len = (size_t) (cursor - term_start);
        token = evoke_copy_span(term_start, span_len);
        if (token == NULL)
        {
            size_t i;

            for (i = 0; i < token_len; i++)
            {
                free(tokens[i]);
            }
            free(tokens);
            free(token_lens);
            return EVOKE_ERR_NOMEM;
        }
        if (token_len >= EVOKE_QUERY_MAX_TOKENS)
        {
            size_t i;

            free(token);
            for (i = 0; i < token_len; i++)
            {
                free(tokens[i]);
            }
            free(tokens);
            free(token_lens);
            return EVOKE_ERR_RANGE;
        }

        new_tokens = realloc(tokens, sizeof(*tokens) * (token_len + 1));
        new_token_lens = realloc(
            token_lens,
            sizeof(*token_lens) * (token_len + 1)
        );
        if (new_tokens == NULL || new_token_lens == NULL)
        {
            size_t i;

            free(token);
            if (new_tokens != NULL)
            {
                tokens = new_tokens;
            }
            if (new_token_lens != NULL)
            {
                token_lens = new_token_lens;
            }
            for (i = 0; i < token_len; i++)
            {
                free(tokens[i]);
            }
            free(tokens);
            free(token_lens);
            return EVOKE_ERR_NOMEM;
        }

        tokens = new_tokens;
        token_lens = new_token_lens;
        tokens[token_len++] = token;
        token_lens[token_len - 1] = span_len;
    }

    if (token_len == 0)
    {
        free(tokens);
        free(token_lens);
        return EVOKE_ERR_INVALID;
    }

    *tokens_out = tokens;
    *token_lens_out = token_lens;
    *len_out = token_len;
    return EVOKE_OK;
}

static evoke_status
evoke_tokenize_query_string(
    const char *input,
    evoke_query_token **tokens_out,
    size_t *len_out,
    bool *has_boolean_syntax_out
)
{
    const char *cursor;
    evoke_query_token *tokens = NULL;
    size_t len = 0;
    bool has_boolean_syntax = false;

    *tokens_out = NULL;
    *len_out = 0;
    *has_boolean_syntax_out = false;
    if (input == NULL)
    {
        return EVOKE_ERR_INVALID;
    }

    cursor = input;
    while (true)
    {
        evoke_query_token token = {0};
        const char *start;
        size_t span_len;
        evoke_status status;

        cursor = evoke_skip_space(cursor);
        if (*cursor == '\0')
        {
            break;
        }

        if (*cursor == '+')
        {
            token.kind = EVOKE_QUERY_TOKEN_PLUS;
            cursor++;
        }
        else if (*cursor == '-')
        {
            token.kind = EVOKE_QUERY_TOKEN_MINUS;
            cursor++;
        }
        else if (*cursor == '(')
        {
            token.kind = EVOKE_QUERY_TOKEN_LPAREN;
            has_boolean_syntax = true;
            cursor++;
        }
        else if (*cursor == ')')
        {
            token.kind = EVOKE_QUERY_TOKEN_RPAREN;
            has_boolean_syntax = true;
            cursor++;
        }
        else if (*cursor == '"')
        {
            cursor++;
            start = cursor;
            while (*cursor != '\0' && *cursor != '"')
            {
                cursor++;
            }
            if (*cursor != '"')
            {
                evoke_query_tokens_free(tokens, len);
                return EVOKE_ERR_INVALID;
            }

            token.kind = EVOKE_QUERY_TOKEN_PHRASE;
            token.term.kind = EVOKE_QUERY_PHRASE;
            status = evoke_split_phrase(
                start,
                (size_t) (cursor - start),
                &token.term.tokens,
                &token.term.token_lens,
                &token.term.len
            );
            if (status != EVOKE_OK)
            {
                evoke_query_tokens_free(tokens, len);
                return status;
            }
            cursor++;
        }
        else
        {
            start = cursor;
            while (*cursor != '\0' &&
                   !isspace((unsigned char) *cursor) &&
                   *cursor != '(' &&
                   *cursor != ')')
            {
                cursor++;
            }
            span_len = (size_t) (cursor - start);
            if (span_len == 0)
            {
                evoke_query_tokens_free(tokens, len);
                return EVOKE_ERR_INVALID;
            }

            if (span_len == 3 && memcmp(start, "AND", 3) == 0)
            {
                token.kind = EVOKE_QUERY_TOKEN_AND;
                has_boolean_syntax = true;
            }
            else if (span_len == 2 && memcmp(start, "OR", 2) == 0)
            {
                token.kind = EVOKE_QUERY_TOKEN_OR;
                has_boolean_syntax = true;
            }
            else if (span_len == 3 && memcmp(start, "NOT", 3) == 0)
            {
                token.kind = EVOKE_QUERY_TOKEN_NOT;
                has_boolean_syntax = true;
            }
            else
            {
                token.kind = EVOKE_QUERY_TOKEN_TERM;
                token.term.kind = EVOKE_QUERY_TERM;
                if (start[span_len - 1] == '*')
                {
                    if (span_len == 1)
                    {
                        evoke_query_tokens_free(tokens, len);
                        return EVOKE_ERR_INVALID;
                    }
                    token.kind = EVOKE_QUERY_TOKEN_PREFIX;
                    token.term.kind = EVOKE_QUERY_PREFIX;
                    span_len--;
                }

                token.term.tokens = malloc(sizeof(*token.term.tokens));
                token.term.token_lens = malloc(sizeof(*token.term.token_lens));
                if (token.term.tokens == NULL || token.term.token_lens == NULL)
                {
                    evoke_query_token_free(&token);
                    evoke_query_tokens_free(tokens, len);
                    return EVOKE_ERR_NOMEM;
                }
                token.term.tokens[0] = evoke_copy_span(start, span_len);
                if (token.term.tokens[0] == NULL)
                {
                    evoke_query_token_free(&token);
                    evoke_query_tokens_free(tokens, len);
                    return EVOKE_ERR_NOMEM;
                }
                token.term.token_lens[0] = span_len;
                token.term.len = 1;
            }
        }

        if (len >= EVOKE_QUERY_MAX_TOKENS)
        {
            evoke_query_token_free(&token);
            evoke_query_tokens_free(tokens, len);
            return EVOKE_ERR_RANGE;
        }
        status = evoke_append_token(&tokens, &len, &token);
        if (status != EVOKE_OK)
        {
            evoke_query_token_free(&token);
            evoke_query_tokens_free(tokens, len);
            return status;
        }
    }

    *tokens_out = tokens;
    *len_out = len;
    *has_boolean_syntax_out = has_boolean_syntax;
    return EVOKE_OK;
}

static evoke_status
evoke_append_term(
    evoke_query *query,
    const evoke_query_term *term,
    size_t *index_out
)
{
    evoke_query_term *terms;
    evoke_query_term cloned;
    size_t new_len;
    evoke_status status;

    if (query == NULL || term == NULL)
    {
        return EVOKE_ERR_INVALID;
    }

    status = evoke_query_term_clone(term, &cloned);
    if (status != EVOKE_OK)
    {
        return status;
    }

    new_len = query->len + 1;
    terms = realloc(query->terms, sizeof(*terms) * new_len);
    if (terms == NULL)
    {
        evoke_query_term_free(&cloned);
        return EVOKE_ERR_NOMEM;
    }

    query->terms = terms;
    query->terms[query->len] = cloned;
    if (index_out != NULL)
    {
        *index_out = query->len;
    }
    query->len = new_len;
    return EVOKE_OK;
}

static evoke_query_node *
evoke_query_build_classic_root(const evoke_query *query)
{
    evoke_query_node *must_root = NULL;
    evoke_query_node *should_root = NULL;
    evoke_query_node *negative_root = NULL;
    size_t i;

    if (query == NULL)
    {
        return NULL;
    }

    for (i = 0; i < query->len; i++)
    {
        const evoke_query_term *term = &query->terms[i];
        evoke_query_node *leaf;

        leaf = evoke_query_node_term(i);
        if (leaf == NULL)
        {
            evoke_query_node_free(must_root);
            evoke_query_node_free(should_root);
            evoke_query_node_free(negative_root);
            return NULL;
        }

        if (term->occur == EVOKE_QUERY_MUST)
        {
            must_root = evoke_query_node_binary(
                EVOKE_QUERY_NODE_AND,
                must_root,
                leaf
            );
        }
        else if (term->occur == EVOKE_QUERY_MUST_NOT)
        {
            negative_root = evoke_query_node_binary(
                EVOKE_QUERY_NODE_AND,
                negative_root,
                evoke_query_node_unary(EVOKE_QUERY_NODE_NOT, leaf)
            );
        }
        else
        {
            should_root = evoke_query_node_binary(
                EVOKE_QUERY_NODE_OR,
                should_root,
                leaf
            );
        }
    }

    if (must_root == NULL)
    {
        must_root = should_root;
        should_root = NULL;
    }

    if (must_root == NULL)
    {
        return negative_root;
    }

    /*
     * Classic SHOULD terms contribute scores but do not constrain a query that
     * already has MUST terms. Their temporary AST is therefore not attached
     * to the match predicate and must be released here.
     */
    evoke_query_node_free(should_root);

    return evoke_query_node_binary(
        EVOKE_QUERY_NODE_AND,
        must_root,
        negative_root
    );
}

static bool
evoke_parser_token_can_start_unary(
    evoke_query_token_kind kind
)
{
    return kind == EVOKE_QUERY_TOKEN_TERM ||
           kind == EVOKE_QUERY_TOKEN_PREFIX ||
           kind == EVOKE_QUERY_TOKEN_PHRASE ||
           kind == EVOKE_QUERY_TOKEN_LPAREN ||
           kind == EVOKE_QUERY_TOKEN_PLUS ||
           kind == EVOKE_QUERY_TOKEN_MINUS ||
           kind == EVOKE_QUERY_TOKEN_NOT;
}

static const evoke_query_token *
evoke_query_parser_peek(const evoke_query_parser *parser)
{
    static const evoke_query_token eof = {
        .kind = EVOKE_QUERY_TOKEN_EOF
    };

    if (parser->pos >= parser->len)
    {
        return &eof;
    }

    return &parser->tokens[parser->pos];
}

static void
evoke_query_parser_consume(evoke_query_parser *parser)
{
    if (parser->pos < parser->len)
    {
        parser->pos++;
    }
}

static evoke_status
evoke_parse_boolean_or(
    evoke_query_parser *parser,
    evoke_query_node **node_out
);

static evoke_status
evoke_parse_boolean_primary(
    evoke_query_parser *parser,
    evoke_query_node **node_out
)
{
    const evoke_query_token *token;
    evoke_status status;

    token = evoke_query_parser_peek(parser);
    if (token->kind == EVOKE_QUERY_TOKEN_LPAREN)
    {
        evoke_query_node *node;

        if (parser->depth >= EVOKE_QUERY_MAX_NESTING)
        {
            return EVOKE_ERR_RANGE;
        }
        evoke_query_parser_consume(parser);
        parser->depth++;
        status = evoke_parse_boolean_or(parser, &node);
        parser->depth--;
        if (status != EVOKE_OK)
        {
            return status;
        }
        if (evoke_query_parser_peek(parser)->kind !=
            EVOKE_QUERY_TOKEN_RPAREN)
        {
            evoke_query_node_free(node);
            return EVOKE_ERR_INVALID;
        }
        evoke_query_parser_consume(parser);
        *node_out = node;
        return EVOKE_OK;
    }

    if (token->kind == EVOKE_QUERY_TOKEN_TERM ||
        token->kind == EVOKE_QUERY_TOKEN_PREFIX ||
        token->kind == EVOKE_QUERY_TOKEN_PHRASE)
    {
        evoke_query_node *node;
        size_t term_index = SIZE_MAX;

        status = evoke_append_term(parser->query, &token->term, &term_index);
        if (status != EVOKE_OK)
        {
            return status;
        }

        node = evoke_query_node_term(term_index);
        if (node == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        evoke_query_parser_consume(parser);
        *node_out = node;
        return EVOKE_OK;
    }

    return EVOKE_ERR_INVALID;
}

static evoke_status
evoke_parse_boolean_unary(
    evoke_query_parser *parser,
    evoke_query_node **node_out
)
{
    const evoke_query_token *token;
    evoke_query_node *child;
    evoke_status status;

    token = evoke_query_parser_peek(parser);
    if (token->kind == EVOKE_QUERY_TOKEN_PLUS)
    {
        if (parser->depth >= EVOKE_QUERY_MAX_NESTING)
        {
            return EVOKE_ERR_RANGE;
        }
        evoke_query_parser_consume(parser);
        parser->depth++;
        status = evoke_parse_boolean_unary(parser, node_out);
        parser->depth--;
        return status;
    }
    if (token->kind == EVOKE_QUERY_TOKEN_MINUS ||
        token->kind == EVOKE_QUERY_TOKEN_NOT)
    {
        if (parser->depth >= EVOKE_QUERY_MAX_NESTING)
        {
            return EVOKE_ERR_RANGE;
        }
        evoke_query_parser_consume(parser);
        parser->depth++;
        status = evoke_parse_boolean_unary(parser, &child);
        parser->depth--;
        if (status != EVOKE_OK)
        {
            return status;
        }

        *node_out = evoke_query_node_unary(
            EVOKE_QUERY_NODE_NOT,
            child
        );
        if (*node_out == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        return EVOKE_OK;
    }

    return evoke_parse_boolean_primary(parser, node_out);
}

static evoke_status
evoke_parse_boolean_and(
    evoke_query_parser *parser,
    evoke_query_node **node_out
)
{
    evoke_query_node *node;
    evoke_status status;

    status = evoke_parse_boolean_unary(parser, &node);
    if (status != EVOKE_OK)
    {
        return status;
    }

    while (evoke_query_parser_peek(parser)->kind ==
           EVOKE_QUERY_TOKEN_AND)
    {
        evoke_query_node *rhs;

        evoke_query_parser_consume(parser);
        status = evoke_parse_boolean_unary(parser, &rhs);
        if (status != EVOKE_OK)
        {
            evoke_query_node_free(node);
            return status;
        }

        node = evoke_query_node_binary(
            EVOKE_QUERY_NODE_AND,
            node,
            rhs
        );
        if (node == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
    }

    *node_out = node;
    return EVOKE_OK;
}

static evoke_status
evoke_parse_boolean_or(
    evoke_query_parser *parser,
    evoke_query_node **node_out
)
{
    evoke_query_node *node;
    evoke_status status;

    status = evoke_parse_boolean_and(parser, &node);
    if (status != EVOKE_OK)
    {
        return status;
    }

    while (true)
    {
        const evoke_query_token *token;
        evoke_query_node *rhs;

        token = evoke_query_parser_peek(parser);
        if (token->kind == EVOKE_QUERY_TOKEN_OR)
        {
            evoke_query_parser_consume(parser);
        }
        else if (!evoke_parser_token_can_start_unary(token->kind))
        {
            break;
        }

        status = evoke_parse_boolean_and(parser, &rhs);
        if (status != EVOKE_OK)
        {
            evoke_query_node_free(node);
            return status;
        }

        node = evoke_query_node_binary(
            EVOKE_QUERY_NODE_OR,
            node,
            rhs
        );
        if (node == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
    }

    *node_out = node;
    return EVOKE_OK;
}

static evoke_status
evoke_parse_boolean_tokens(
    const evoke_query_token *tokens,
    size_t len,
    evoke_query *query_out
)
{
    evoke_query query = {0};
    evoke_query_parser parser;
    evoke_status status;

    evoke_query_init(&query);
    query.mode = EVOKE_QUERY_MODE_BOOLEAN_AST;

    parser.tokens = tokens;
    parser.len = len;
    parser.pos = 0;
    parser.depth = 0;
    parser.query = &query;

    status = evoke_parse_boolean_or(&parser, &query.root);
    if (status != EVOKE_OK)
    {
        evoke_query_free(&query);
        return status;
    }
    if (query.root == NULL || parser.pos != len)
    {
        evoke_query_free(&query);
        return EVOKE_ERR_INVALID;
    }

    *query_out = query;
    return EVOKE_OK;
}

static evoke_status
evoke_parse_classic_tokens(
    const evoke_query_token *tokens,
    size_t len,
    evoke_query *query_out
)
{
    evoke_query query = {0};
    evoke_query_occur next_occur = EVOKE_QUERY_SHOULD;
    bool expect_term = true;
    size_t i;

    evoke_query_init(&query);
    query.mode = EVOKE_QUERY_MODE_CLASSIC;

    for (i = 0; i < len; i++)
    {
        evoke_query_term term;
        evoke_status status;

        if (tokens[i].kind == EVOKE_QUERY_TOKEN_PLUS)
        {
            next_occur = EVOKE_QUERY_MUST;
            expect_term = true;
            continue;
        }
        if (tokens[i].kind == EVOKE_QUERY_TOKEN_MINUS)
        {
            next_occur = EVOKE_QUERY_MUST_NOT;
            expect_term = true;
            continue;
        }
        if (tokens[i].kind != EVOKE_QUERY_TOKEN_TERM &&
            tokens[i].kind != EVOKE_QUERY_TOKEN_PREFIX &&
            tokens[i].kind != EVOKE_QUERY_TOKEN_PHRASE)
        {
            evoke_query_free(&query);
            return EVOKE_ERR_INVALID;
        }

        term = tokens[i].term;
        term.occur = next_occur;
        status = evoke_append_term(&query, &term, NULL);
        if (status != EVOKE_OK)
        {
            evoke_query_free(&query);
            return status;
        }
        next_occur = EVOKE_QUERY_SHOULD;
        expect_term = false;
    }

    if (expect_term && len > 0)
    {
        evoke_query_free(&query);
        return EVOKE_ERR_INVALID;
    }

    query.root = evoke_query_build_classic_root(&query);
    if (query.len > 0 && query.root == NULL)
    {
        evoke_query_free(&query);
        return EVOKE_ERR_NOMEM;
    }

    *query_out = query;
    return EVOKE_OK;
}

static evoke_query_node *
evoke_query_rewrite_node(
    evoke_query_node *node,
    const size_t *old_to_new,
    size_t old_len
)
{
    if (node == NULL)
    {
        return NULL;
    }

    if (node->kind == EVOKE_QUERY_NODE_TERM)
    {
        if (node->term_index >= old_len ||
            old_to_new[node->term_index] == SIZE_MAX)
        {
            free(node);
            return NULL;
        }

        node->term_index = old_to_new[node->term_index];
        return node;
    }

    node->left = evoke_query_rewrite_node(
        node->left,
        old_to_new,
        old_len
    );
    node->right = evoke_query_rewrite_node(
        node->right,
        old_to_new,
        old_len
    );

    if (node->kind == EVOKE_QUERY_NODE_NOT)
    {
        if (node->left == NULL)
        {
            free(node);
            return NULL;
        }
        return node;
    }

    if (node->left == NULL && node->right == NULL)
    {
        free(node);
        return NULL;
    }
    if (node->left == NULL)
    {
        evoke_query_node *right = node->right;

        free(node);
        return right;
    }
    if (node->right == NULL)
    {
        evoke_query_node *left = node->left;

        free(node);
        return left;
    }

    return node;
}

void
evoke_query_rewrite_terms(
    evoke_query *query,
    const size_t *old_to_new,
    size_t old_len
)
{
    if (query == NULL || old_to_new == NULL)
    {
        return;
    }

    query->root = evoke_query_rewrite_node(
        query->root,
        old_to_new,
        old_len
    );
}

evoke_status
evoke_parse_query_string(
    const char *input,
    evoke_query *query_out
)
{
    evoke_query_token *tokens = NULL;
    size_t len = 0;
    bool has_boolean_syntax = false;
    evoke_status status;

    if (input == NULL || query_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }

    status = evoke_tokenize_query_string(
        input,
        &tokens,
        &len,
        &has_boolean_syntax
    );
    if (status != EVOKE_OK)
    {
        return status;
    }

    if (has_boolean_syntax)
    {
        status = evoke_parse_boolean_tokens(tokens, len, query_out);
    }
    else
    {
        status = evoke_parse_classic_tokens(tokens, len, query_out);
    }

    evoke_query_tokens_free(tokens, len);
    return status;
}
