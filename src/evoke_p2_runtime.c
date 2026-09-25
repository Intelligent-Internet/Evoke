#include "postgres.h"

#include <float.h>
#include <math.h>
#include <sys/stat.h>

#include "common/hashfn.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "storage/fd.h"
#include "utils/builtins.h"
#include "utils/hsearch.h"
#include "utils/jsonb.h"
#include "utils/memutils.h"
#include "utils/numeric.h"

#include <unicode/uregex.h>
#include <unicode/ustring.h>

#include "evoke_p2_runtime.h"

#define EVOKE_P2_BOS_TOKEN_ID 0
#define EVOKE_P2_EOS_TOKEN_ID 2
#define EVOKE_P2_MAX_WINDOWS 256
#define EVOKE_P2_RMS_HEADER_BYTES 24
#define EVOKE_P2_RMS_MAGIC "EVOKEP2R1"
#define EVOKE_P2_CACHE_IDENTITY_BYTES 65
#define EVOKE_P2_BPE_PATTERN \
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+|" \
    " ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"

typedef struct EvokeP2TokenKey
{
    uint64 hash_one;
    uint64 hash_two;
    uint32 length;
    uint32 reserved;
} EvokeP2TokenKey;

typedef struct EvokeP2PairKey
{
    EvokeP2TokenKey left;
    EvokeP2TokenKey right;
} EvokeP2PairKey;

typedef struct EvokeP2VocabularyEntry
{
    EvokeP2TokenKey key;
    int32 token_id;
} EvokeP2VocabularyEntry;

typedef struct EvokeP2MergeEntry
{
    EvokeP2PairKey key;
    int32 rank;
} EvokeP2MergeEntry;

typedef struct EvokeP2BpeSymbol
{
    char *text;
    Size length;
    EvokeP2TokenKey key;
} EvokeP2BpeSymbol;

typedef struct EvokeP2TokenizerCache
{
    MemoryContext context;
    char vocabulary_path[MAXPGPATH];
    char merges_path[MAXPGPATH];
    char checkout_signature[EVOKE_P2_CACHE_IDENTITY_BYTES];
    HTAB *vocabulary;
    HTAB *merges;
    URegularExpression *pattern;
} EvokeP2TokenizerCache;

typedef struct EvokeP2LexicalEntry
{
    EvokeP2TokenKey key;
    int32 token_id;
} EvokeP2LexicalEntry;

typedef struct EvokeP2CompilerCache
{
    MemoryContext context;
    char vocabulary_path[MAXPGPATH];
    char calibration_path[MAXPGPATH];
    char checkout_signature[EVOKE_P2_CACHE_IDENTITY_BYTES];
    HTAB *vocabulary;
    URegularExpression *term_pattern;
    float4 *rms;
    int32 lexical_dims;
    int32 total_dims;
    uint64 document_count;
} EvokeP2CompilerCache;

typedef struct EvokeP2ActiveLexical
{
    int32 atom_id;
    float4 frequency;
} EvokeP2ActiveLexical;

typedef struct EvokeP2ActiveSemantic
{
    int32 atom_id;
    float4 weight;
} EvokeP2ActiveSemantic;

typedef struct EvokeP2TokenBuilder
{
    int64 *values;
    int32 count;
    int32 capacity;
    int32 limit;
} EvokeP2TokenBuilder;

static EvokeP2TokenizerCache *evoke_p2_tokenizer_cache = NULL;
static EvokeP2CompilerCache *evoke_p2_compiler_cache = NULL;

static EvokeP2TokenKey
evoke_p2_token_key(const char *text, Size length)
{
    EvokeP2TokenKey key;

    memset(&key, 0, sizeof(key));
    key.hash_one = hash_any_extended(
        (const unsigned char *) text,
        (int) length,
        UINT64CONST(0x9e3779b97f4a7c15)
    );
    key.hash_two = hash_any_extended(
        (const unsigned char *) text,
        (int) length,
        UINT64CONST(0xd1b54a32d192ed03)
    );
    key.length = (uint32) length;
    return key;
}

static char *
evoke_p2_read_file(const char *path, Size *length_out)
{
    struct stat stat_buffer;
    FILE *handle;
    char *contents;
    Size length;

    if (stat(path, &stat_buffer) != 0 || stat_buffer.st_size <= 0)
    {
        ereport(
            ERROR,
            (
                errmsg("could not stat P2 tokenizer artifact"),
                errdetail("path=%s", path)
            )
        );
    }
    if ((uint64) stat_buffer.st_size > (uint64) MaxAllocSize - 1)
    {
        ereport(ERROR, (errmsg("P2 tokenizer artifact is too large")));
    }
    length = (Size) stat_buffer.st_size;
    contents = palloc(length + 1);
    handle = AllocateFile(path, "rb");
    if (handle == NULL)
    {
        ereport(ERROR, (errmsg("could not open P2 tokenizer artifact")));
    }
    if (fread(contents, 1, length, handle) != length)
    {
        FreeFile(handle);
        ereport(ERROR, (errmsg("could not read P2 tokenizer artifact")));
    }
    FreeFile(handle);
    contents[length] = '\0';
    *length_out = length;
    return contents;
}

static HTAB *
evoke_p2_create_vocabulary_hash(MemoryContext context)
{
    HASHCTL control;

    memset(&control, 0, sizeof(control));
    control.keysize = sizeof(EvokeP2TokenKey);
    control.entrysize = sizeof(EvokeP2VocabularyEntry);
    control.hcxt = context;
    return hash_create(
        "evoke P2 tokenizer vocabulary",
        65536,
        &control,
        HASH_ELEM | HASH_BLOBS | HASH_CONTEXT
    );
}

static HTAB *
evoke_p2_create_merge_hash(MemoryContext context)
{
    HASHCTL control;

    memset(&control, 0, sizeof(control));
    control.keysize = sizeof(EvokeP2PairKey);
    control.entrysize = sizeof(EvokeP2MergeEntry);
    control.hcxt = context;
    return hash_create(
        "evoke P2 tokenizer merges",
        65536,
        &control,
        HASH_ELEM | HASH_BLOBS | HASH_CONTEXT
    );
}

static void
evoke_p2_load_vocabulary(
    EvokeP2TokenizerCache *cache,
    const char *path
)
{
    Size length;
    char *contents;
    Jsonb *value;
    JsonbIterator *iterator;
    JsonbIteratorToken token;
    JsonbValue item;
    EvokeP2TokenKey pending_key;
    bool has_pending_key = false;

    contents = evoke_p2_read_file(path, &length);
    value = DatumGetJsonbP(DirectFunctionCall1(
        jsonb_in,
        CStringGetDatum(contents)
    ));
    iterator = JsonbIteratorInit(&value->root);
    while ((token = JsonbIteratorNext(&iterator, &item, true)) != WJB_DONE)
    {
        if (token == WJB_KEY)
        {
            pending_key = evoke_p2_token_key(
                item.val.string.val,
                (Size) item.val.string.len
            );
            has_pending_key = true;
        }
        else if (token == WJB_VALUE && has_pending_key)
        {
            EvokeP2VocabularyEntry *entry;
            bool found;

            if (item.type != jbvNumeric)
            {
                ereport(ERROR, (errmsg("invalid P2 tokenizer vocabulary")));
            }
            entry = hash_search(
                cache->vocabulary,
                &pending_key,
                HASH_ENTER,
                &found
            );
            if (found)
            {
                ereport(
                    ERROR,
                    (errmsg("P2 tokenizer vocabulary hash collision"))
                );
            }
            entry->token_id = DatumGetInt32(DirectFunctionCall1(
                numeric_int4,
                NumericGetDatum(item.val.numeric)
            ));
            has_pending_key = false;
        }
    }
    pfree(contents);
    if (hash_get_num_entries(cache->vocabulary) != 50265)
    {
        ereport(
            ERROR,
            (
                errmsg("unexpected P2 tokenizer vocabulary size"),
                errdetail(
                    "expected=50265 actual=%ld",
                    hash_get_num_entries(cache->vocabulary)
                )
            )
        );
    }
}

static void
evoke_p2_load_merges(EvokeP2TokenizerCache *cache, const char *path)
{
    Size length;
    char *contents;
    char *line;
    char *save_pointer = NULL;
    int32 rank = 0;

    contents = evoke_p2_read_file(path, &length);
    line = strtok_r(contents, "\n", &save_pointer);
    while (line != NULL)
    {
        Size line_length = strlen(line);

        if (line_length > 0 && line[line_length - 1] == '\r')
        {
            line[--line_length] = '\0';
        }
        if (line_length > 0 && line[0] != '#')
        {
            char *separator = strchr(line, ' ');
            EvokeP2PairKey key;
            EvokeP2MergeEntry *entry;
            bool found;

            if (separator == NULL || separator == line || separator[1] == '\0')
            {
                ereport(ERROR, (errmsg("invalid P2 tokenizer merge row")));
            }
            *separator = '\0';
            key.left = evoke_p2_token_key(line, strlen(line));
            key.right = evoke_p2_token_key(
                separator + 1,
                strlen(separator + 1)
            );
            entry = hash_search(cache->merges, &key, HASH_ENTER, &found);
            if (found)
            {
                ereport(ERROR, (errmsg("duplicate P2 tokenizer merge row")));
            }
            entry->rank = rank++;
        }
        line = strtok_r(NULL, "\n", &save_pointer);
    }
    pfree(contents);
    if (rank != 49992)
    {
        ereport(
            ERROR,
            (
                errmsg("unexpected P2 tokenizer merge count"),
                errdetail("expected=49992 actual=%d", rank)
            )
        );
    }
}

void
evoke_p2_tokenizer_cache_clear(void)
{
    MemoryContext context;

    if (evoke_p2_tokenizer_cache == NULL)
    {
        return;
    }
    if (evoke_p2_tokenizer_cache->pattern != NULL)
    {
        uregex_close(evoke_p2_tokenizer_cache->pattern);
    }
    context = evoke_p2_tokenizer_cache->context;
    evoke_p2_tokenizer_cache = NULL;
    MemoryContextDelete(context);
}

static EvokeP2TokenizerCache *
evoke_p2_get_tokenizer(
    const char *vocabulary_path,
    const char *merges_path,
    const char *checkout_signature
)
{
    EvokeP2TokenizerCache *volatile candidate = NULL;
    MemoryContext context;
    MemoryContext old_context;
    UErrorCode status = U_ZERO_ERROR;

    if (
        evoke_p2_tokenizer_cache != NULL &&
        strcmp(
            evoke_p2_tokenizer_cache->vocabulary_path,
            vocabulary_path
        ) == 0 &&
        strcmp(evoke_p2_tokenizer_cache->merges_path, merges_path) == 0 &&
        strcmp(
            evoke_p2_tokenizer_cache->checkout_signature,
            checkout_signature
        ) == 0
    )
    {
        return evoke_p2_tokenizer_cache;
    }
    evoke_p2_tokenizer_cache_clear();
    context = AllocSetContextCreate(
        TopMemoryContext,
        "evoke P2 tokenizer cache",
        ALLOCSET_DEFAULT_SIZES
    );
    old_context = MemoryContextSwitchTo(context);
    PG_TRY();
    {
        candidate = palloc0(sizeof(*candidate));
        candidate->context = context;
        strlcpy(
            candidate->vocabulary_path,
            vocabulary_path,
            sizeof(candidate->vocabulary_path)
        );
        strlcpy(
            candidate->merges_path,
            merges_path,
            sizeof(candidate->merges_path)
        );
        strlcpy(
            candidate->checkout_signature,
            checkout_signature,
            sizeof(candidate->checkout_signature)
        );
        candidate->vocabulary = evoke_p2_create_vocabulary_hash(context);
        candidate->merges = evoke_p2_create_merge_hash(context);
        evoke_p2_load_vocabulary(candidate, vocabulary_path);
        evoke_p2_load_merges(candidate, merges_path);
        candidate->pattern = uregex_openC(
            EVOKE_P2_BPE_PATTERN,
            0,
            NULL,
            &status
        );
        if (U_FAILURE(status) || candidate->pattern == NULL)
        {
            ereport(
                ERROR,
                (
                    errmsg("could not compile P2 byte-level BPE pattern"),
                    errdetail("ICU error=%s", u_errorName(status))
                )
            );
        }
        MemoryContextSwitchTo(old_context);
    }
    PG_CATCH();
    {
        MemoryContextSwitchTo(old_context);
        if (candidate != NULL && candidate->pattern != NULL)
        {
            uregex_close(candidate->pattern);
        }
        MemoryContextDelete(context);
        PG_RE_THROW();
    }
    PG_END_TRY();

    evoke_p2_tokenizer_cache = (EvokeP2TokenizerCache *) candidate;
    return evoke_p2_tokenizer_cache;
}

static UChar *
evoke_p2_utf8_to_utf16(const char *text, int32 *length_out)
{
    UErrorCode status = U_ZERO_ERROR;
    UChar *output;
    int32 length = 0;

    u_strFromUTF8(NULL, 0, &length, text, -1, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status))
    {
        ereport(ERROR, (errmsg("P2 query text is not valid UTF-8")));
    }
    status = U_ZERO_ERROR;
    output = palloc(sizeof(UChar) * (Size) (length + 1));
    u_strFromUTF8(output, length + 1, NULL, text, -1, &status);
    if (U_FAILURE(status))
    {
        ereport(ERROR, (errmsg("could not convert P2 query text to UTF-16")));
    }
    *length_out = length;
    return output;
}

static char *
evoke_p2_utf16_slice_to_utf8(
    const UChar *text,
    int32 start,
    int32 end,
    Size *length_out
)
{
    UErrorCode status = U_ZERO_ERROR;
    int32 length = 0;
    char *output;

    u_strToUTF8(NULL, 0, &length, text + start, end - start, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status))
    {
        ereport(ERROR, (errmsg("could not size P2 UTF-8 token")));
    }
    status = U_ZERO_ERROR;
    output = palloc((Size) length + 1);
    u_strToUTF8(output, length + 1, NULL, text + start, end - start, &status);
    if (U_FAILURE(status))
    {
        ereport(ERROR, (errmsg("could not encode P2 UTF-8 token")));
    }
    output[length] = '\0';
    *length_out = (Size) length;
    return output;
}

static bool
evoke_p2_byte_uses_identity_codepoint(unsigned int value)
{
    return (
        (value >= 33 && value <= 126) ||
        (value >= 161 && value <= 172) ||
        (value >= 174 && value <= 255)
    );
}

static int32
evoke_p2_byte_codepoint(unsigned int value)
{
    int32 extra = 0;

    if (evoke_p2_byte_uses_identity_codepoint(value))
    {
        return (int32) value;
    }
    for (unsigned int candidate = 0; candidate < value; candidate++)
    {
        if (!evoke_p2_byte_uses_identity_codepoint(candidate))
        {
            extra++;
        }
    }
    return 256 + extra;
}

static EvokeP2BpeSymbol
evoke_p2_initial_symbol(unsigned char value)
{
    EvokeP2BpeSymbol symbol;
    UChar unicode[2];
    int32 unicode_length = 0;
    UErrorCode status = U_ZERO_ERROR;
    int32 utf8_length = 0;
    int32 codepoint = evoke_p2_byte_codepoint((unsigned int) value);

    U16_APPEND_UNSAFE(unicode, unicode_length, codepoint);
    u_strToUTF8(NULL, 0, &utf8_length, unicode, unicode_length, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status))
    {
        ereport(ERROR, (errmsg("could not size P2 byte symbol")));
    }
    status = U_ZERO_ERROR;
    symbol.text = palloc((Size) utf8_length + 1);
    u_strToUTF8(
        symbol.text,
        utf8_length + 1,
        NULL,
        unicode,
        unicode_length,
        &status
    );
    if (U_FAILURE(status))
    {
        ereport(ERROR, (errmsg("could not encode P2 byte symbol")));
    }
    symbol.text[utf8_length] = '\0';
    symbol.length = (Size) utf8_length;
    symbol.key = evoke_p2_token_key(symbol.text, symbol.length);
    return symbol;
}

static EvokeP2PairKey
evoke_p2_pair_key(
    const EvokeP2BpeSymbol *left,
    const EvokeP2BpeSymbol *right
)
{
    EvokeP2PairKey key;

    key.left = left->key;
    key.right = right->key;
    return key;
}

static EvokeP2BpeSymbol
evoke_p2_merge_symbols(
    const EvokeP2BpeSymbol *left,
    const EvokeP2BpeSymbol *right
)
{
    EvokeP2BpeSymbol merged;

    merged.length = left->length + right->length;
    merged.text = palloc(merged.length + 1);
    memcpy(merged.text, left->text, left->length);
    memcpy(merged.text + left->length, right->text, right->length);
    merged.text[merged.length] = '\0';
    merged.key = evoke_p2_token_key(merged.text, merged.length);
    return merged;
}

static bool
evoke_p2_token_builder_append(
    EvokeP2TokenBuilder *builder,
    int32 token_id
)
{
    int32 new_capacity;

    if (builder->count >= builder->limit)
    {
        return false;
    }
    if (builder->count == builder->capacity)
    {
        new_capacity = builder->capacity == 0
            ? Min(builder->limit, 256)
            : Min(builder->limit, builder->capacity * 2);
        if (new_capacity <= builder->capacity)
        {
            ereport(ERROR, (errmsg("P2 token buffer cannot grow")));
        }
        builder->values = builder->values == NULL
            ? palloc(sizeof(*builder->values) * (Size) new_capacity)
            : repalloc(
                builder->values,
                sizeof(*builder->values) * (Size) new_capacity
            );
        builder->capacity = new_capacity;
    }
    builder->values[builder->count++] = (int64) token_id;
    return true;
}

static bool
evoke_p2_append_piece(
    EvokeP2TokenizerCache *cache,
    const char *piece,
    Size piece_length,
    EvokeP2TokenBuilder *builder
)
{
    EvokeP2BpeSymbol *symbols;
    int32 symbol_count = (int32) piece_length;

    symbols = palloc(sizeof(*symbols) * Max(symbol_count, 1));
    for (int32 i = 0; i < symbol_count; i++)
    {
        symbols[i] = evoke_p2_initial_symbol((unsigned char) piece[i]);
    }
    while (symbol_count > 1)
    {
        int32 best_rank = PG_INT32_MAX;
        EvokeP2PairKey best_pair;
        bool has_pair = false;
        EvokeP2BpeSymbol *next_symbols;
        int32 next_count = 0;

        for (int32 i = 0; i + 1 < symbol_count; i++)
        {
            EvokeP2PairKey key = evoke_p2_pair_key(
                &symbols[i],
                &symbols[i + 1]
            );
            EvokeP2MergeEntry *entry = hash_search(
                cache->merges,
                &key,
                HASH_FIND,
                NULL
            );

            if (entry != NULL && entry->rank < best_rank)
            {
                best_rank = entry->rank;
                best_pair = key;
                has_pair = true;
            }
        }
        if (!has_pair)
        {
            break;
        }
        next_symbols = palloc(sizeof(*next_symbols) * symbol_count);
        for (int32 i = 0; i < symbol_count;)
        {
            bool merge = false;

            if (i + 1 < symbol_count)
            {
                EvokeP2PairKey key = evoke_p2_pair_key(
                    &symbols[i],
                    &symbols[i + 1]
                );

                merge = memcmp(&key, &best_pair, sizeof(key)) == 0;
            }
            if (merge)
            {
                next_symbols[next_count++] = evoke_p2_merge_symbols(
                    &symbols[i],
                    &symbols[i + 1]
                );
                i += 2;
            }
            else
            {
                next_symbols[next_count++] = symbols[i++];
            }
        }
        symbols = next_symbols;
        symbol_count = next_count;
    }
    for (int32 i = 0; i < symbol_count; i++)
    {
        EvokeP2VocabularyEntry *entry = hash_search(
            cache->vocabulary,
            &symbols[i].key,
            HASH_FIND,
            NULL
        );

        if (entry == NULL)
        {
            ereport(
                ERROR,
                (
                    errmsg("P2 BPE token is absent from vocabulary"),
                    errdetail("token=%s", symbols[i].text)
                )
            );
        }
        if (!evoke_p2_token_builder_append(builder, entry->token_id))
        {
            return false;
        }
    }
    return true;
}

void
evoke_p2_tokenize_roberta_windows(
    const char *vocabulary_path,
    const char *merges_path,
    const char *checkout_signature,
    const char *query_text,
    int32 max_length,
    int32 window_stride,
    int32 max_windows,
    EvokeP2TokenizedWindows *output
)
{
    EvokeP2TokenizerCache *cache;
    EvokeP2TokenBuilder builder;
    UChar *query_utf16;
    int32 query_utf16_length;
    UErrorCode status = U_ZERO_ERROR;
    int32 window_content_capacity;
    int32 content_limit;
    int32 window_count = 0;
    int32 window_start = 0;

    if (max_length < 3 || max_length > 4096)
    {
        ereport(ERROR, (errmsg("invalid P2 tokenizer max_length")));
    }
    window_content_capacity = max_length - 2;
    if (window_stride < 1 || window_stride > window_content_capacity)
    {
        ereport(ERROR, (errmsg("invalid P2 tokenizer window_stride")));
    }
    if (max_windows < 1 || max_windows > EVOKE_P2_MAX_WINDOWS)
    {
        ereport(ERROR, (errmsg("invalid P2 tokenizer max_windows")));
    }
    if (checkout_signature == NULL ||
        strlen(checkout_signature) != EVOKE_P2_CACHE_IDENTITY_BYTES - 1)
    {
        ereport(ERROR, (errmsg("invalid P2 tokenizer cache identity")));
    }
    if (window_stride >
        (PG_INT32_MAX - window_content_capacity) / Max(max_windows - 1, 1))
    {
        ereport(ERROR, (errmsg("P2 tokenizer window limit is too large")));
    }
    content_limit = window_content_capacity +
        window_stride * (max_windows - 1);
    memset(output, 0, sizeof(*output));
    memset(&builder, 0, sizeof(builder));
    builder.limit = content_limit;
    cache = evoke_p2_get_tokenizer(
        vocabulary_path,
        merges_path,
        checkout_signature
    );
    query_utf16 = evoke_p2_utf8_to_utf16(query_text, &query_utf16_length);

    uregex_setText(
        cache->pattern,
        query_utf16,
        query_utf16_length,
        &status
    );
    while (U_SUCCESS(status) && uregex_findNext(cache->pattern, &status))
    {
        int32 start = uregex_start(cache->pattern, 0, &status);
        int32 end = uregex_end(cache->pattern, 0, &status);
        Size piece_length;
        char *piece;

        if (U_FAILURE(status))
        {
            break;
        }
        piece = evoke_p2_utf16_slice_to_utf8(
            query_utf16,
            start,
            end,
            &piece_length
        );
        if (!evoke_p2_append_piece(
            cache,
            piece,
            piece_length,
            &builder
        ))
        {
            break;
        }
    }
    if (U_FAILURE(status))
    {
        ereport(
            ERROR,
            (
                errmsg("P2 byte-level BPE tokenization failed"),
                errdetail("ICU error=%s", u_errorName(status))
            )
        );
    }
    output->full_token_count = builder.count + 2;
    output->window_stride = window_stride;
    output->windows = palloc0(
        sizeof(*output->windows) * (Size) max_windows
    );
    do
    {
        EvokeP2TokenizedInput *window = &output->windows[window_count];
        int32 remaining = builder.count - window_start;
        int32 content_count = Min(remaining, window_content_capacity);
        int32 token_count = content_count + 2;

        window->input_ids = palloc0(
            sizeof(*window->input_ids) * (Size) token_count
        );
        window->attention_mask = palloc0(
            sizeof(*window->attention_mask) * (Size) token_count
        );
        window->input_ids[0] = EVOKE_P2_BOS_TOKEN_ID;
        if (content_count > 0)
        {
            memcpy(
                window->input_ids + 1,
                builder.values + window_start,
                sizeof(*window->input_ids) * (Size) content_count
            );
        }
        window->input_ids[token_count - 1] = EVOKE_P2_EOS_TOKEN_ID;
        for (int32 i = 0; i < token_count; i++)
        {
            window->attention_mask[i] = 1;
        }
        window->token_count = token_count;
        window->shape[0] = 1;
        window->shape[1] = token_count;
        window_count++;
        if (window_start + content_count >= builder.count)
        {
            break;
        }
        window_start += window_stride;
    }
    while (window_count < max_windows);
    output->window_count = window_count;
    if (builder.values != NULL)
    {
        pfree(builder.values);
    }
}

void
evoke_p2_tokenized_windows_free(EvokeP2TokenizedWindows *tokenized)
{
    if (tokenized == NULL)
    {
        return;
    }
    for (int32 i = 0; i < tokenized->window_count; i++)
    {
        if (tokenized->windows[i].input_ids != NULL)
        {
            pfree(tokenized->windows[i].input_ids);
        }
        if (tokenized->windows[i].attention_mask != NULL)
        {
            pfree(tokenized->windows[i].attention_mask);
        }
    }
    if (tokenized->windows != NULL)
    {
        pfree(tokenized->windows);
    }
    memset(tokenized, 0, sizeof(*tokenized));
}

void
evoke_p2_tokenize_roberta(
    const char *vocabulary_path,
    const char *merges_path,
    const char *checkout_signature,
    const char *query_text,
    int32 max_length,
    EvokeP2TokenizedInput *output
)
{
    EvokeP2TokenizedWindows tokenized;

    evoke_p2_tokenize_roberta_windows(
        vocabulary_path,
        merges_path,
        checkout_signature,
        query_text,
        max_length,
        max_length - 2,
        1,
        &tokenized
    );
    *output = tokenized.windows[0];
    pfree(tokenized.windows);
}

static uint32
evoke_p2_read_le32(const unsigned char *value)
{
    return (
        (uint32) value[0] |
        ((uint32) value[1] << 8) |
        ((uint32) value[2] << 16) |
        ((uint32) value[3] << 24)
    );
}

static uint64
evoke_p2_read_le64(const unsigned char *value)
{
    uint64 result = 0;

    for (int shift = 0; shift < 64; shift += 8)
    {
        result |= ((uint64) *value++) << shift;
    }
    return result;
}

static float4
evoke_p2_read_le_float4(const unsigned char *value)
{
    uint32 bits = evoke_p2_read_le32(value);
    float4 result;

    memcpy(&result, &bits, sizeof(result));
    return result;
}

static HTAB *
evoke_p2_create_lexical_hash(MemoryContext context, int32 lexical_dims)
{
    HASHCTL control;

    memset(&control, 0, sizeof(control));
    control.keysize = sizeof(EvokeP2TokenKey);
    control.entrysize = sizeof(EvokeP2LexicalEntry);
    control.hcxt = context;
    return hash_create(
        "evoke P2 lexical vocabulary",
        lexical_dims,
        &control,
        HASH_ELEM | HASH_BLOBS | HASH_CONTEXT
    );
}

static void
evoke_p2_load_lexical_vocabulary(
    EvokeP2CompilerCache *cache,
    const char *path
)
{
    Size length;
    char *contents;
    Jsonb *value;
    JsonbIterator *iterator;
    JsonbIteratorToken token;
    JsonbValue item;
    int32 token_id = 0;

    contents = evoke_p2_read_file(path, &length);
    value = DatumGetJsonbP(DirectFunctionCall1(
        jsonb_in,
        CStringGetDatum(contents)
    ));
    if (!JB_ROOT_IS_ARRAY(value))
    {
        ereport(ERROR, (errmsg("invalid P2 lexical vocabulary")));
    }
    iterator = JsonbIteratorInit(&value->root);
    while ((token = JsonbIteratorNext(&iterator, &item, true)) != WJB_DONE)
    {
        EvokeP2TokenKey key;
        EvokeP2LexicalEntry *entry;
        bool found;

        if (token != WJB_ELEM)
        {
            continue;
        }
        if (item.type != jbvString || item.val.string.len <= 0)
        {
            ereport(ERROR, (errmsg("invalid P2 lexical vocabulary token")));
        }
        if (token_id >= cache->lexical_dims)
        {
            ereport(ERROR, (errmsg("P2 lexical vocabulary is too large")));
        }
        key = evoke_p2_token_key(
            item.val.string.val,
            (Size) item.val.string.len
        );
        entry = hash_search(cache->vocabulary, &key, HASH_ENTER, &found);
        if (found)
        {
            ereport(
                ERROR,
                (errmsg("duplicate or colliding P2 lexical token"))
            );
        }
        entry->token_id = token_id++;
    }
    pfree(contents);
    if (token_id != cache->lexical_dims)
    {
        ereport(
            ERROR,
            (
                errmsg("unexpected P2 lexical vocabulary size"),
                errdetail(
                    "expected=%d actual=%d",
                    cache->lexical_dims,
                    token_id
                )
            )
        );
    }
}

static void
evoke_p2_load_calibration(
    EvokeP2CompilerCache *cache,
    const char *path
)
{
    Size length;
    unsigned char *contents;
    uint32 version;
    uint32 total_dims;
    uint64 expected_bytes;

    contents = (unsigned char *) evoke_p2_read_file(path, &length);
    if (length < EVOKE_P2_RMS_HEADER_BYTES ||
        memcmp(contents, EVOKE_P2_RMS_MAGIC, 8) != 0)
    {
        ereport(ERROR, (errmsg("invalid P2 RMS calibration header")));
    }
    version = evoke_p2_read_le32(contents + 8);
    total_dims = evoke_p2_read_le32(contents + 12);
    cache->document_count = evoke_p2_read_le64(contents + 16);
    expected_bytes = EVOKE_P2_RMS_HEADER_BYTES + (uint64) total_dims * 4;
    if (version != 1 || total_dims != (uint32) cache->total_dims ||
        cache->document_count == 0 || expected_bytes != (uint64) length)
    {
        ereport(
            ERROR,
            (
                errmsg("invalid P2 RMS calibration payload"),
                errdetail(
                    "version=%u dims=%u documents=%llu bytes=%zu",
                    version,
                    total_dims,
                    (unsigned long long) cache->document_count,
                    length
                )
            )
        );
    }
    cache->rms = palloc(sizeof(float4) * (Size) cache->total_dims);
    for (int32 i = 0; i < cache->total_dims; i++)
    {
        float4 rms = evoke_p2_read_le_float4(
            contents + EVOKE_P2_RMS_HEADER_BYTES + (Size) i * 4
        );

        if (!isfinite((double) rms) || rms < 0.0f)
        {
            ereport(ERROR, (errmsg("invalid P2 RMS calibration value")));
        }
        cache->rms[i] = rms;
    }
    pfree(contents);
}

void
evoke_p2_compiler_cache_clear(void)
{
    MemoryContext context;

    if (evoke_p2_compiler_cache == NULL)
    {
        return;
    }
    if (evoke_p2_compiler_cache->term_pattern != NULL)
    {
        uregex_close(evoke_p2_compiler_cache->term_pattern);
    }
    context = evoke_p2_compiler_cache->context;
    evoke_p2_compiler_cache = NULL;
    MemoryContextDelete(context);
}

static EvokeP2CompilerCache *
evoke_p2_get_compiler(
    const char *vocabulary_path,
    const char *calibration_path,
    const char *checkout_signature,
    int32 lexical_dims,
    int32 total_dims
)
{
    EvokeP2CompilerCache *volatile candidate = NULL;
    MemoryContext context;
    MemoryContext old_context;
    UErrorCode status = U_ZERO_ERROR;

    if (
        evoke_p2_compiler_cache != NULL &&
        strcmp(
            evoke_p2_compiler_cache->vocabulary_path,
            vocabulary_path
        ) == 0 &&
        strcmp(
            evoke_p2_compiler_cache->calibration_path,
            calibration_path
        ) == 0 &&
        strcmp(
            evoke_p2_compiler_cache->checkout_signature,
            checkout_signature
        ) == 0 &&
        evoke_p2_compiler_cache->lexical_dims == lexical_dims &&
        evoke_p2_compiler_cache->total_dims == total_dims
    )
    {
        return evoke_p2_compiler_cache;
    }
    evoke_p2_compiler_cache_clear();
    context = AllocSetContextCreate(
        TopMemoryContext,
        "evoke P2 compiler cache",
        ALLOCSET_DEFAULT_SIZES
    );
    old_context = MemoryContextSwitchTo(context);
    PG_TRY();
    {
        candidate = palloc0(sizeof(*candidate));
        candidate->context = context;
        candidate->lexical_dims = lexical_dims;
        candidate->total_dims = total_dims;
        strlcpy(
            candidate->vocabulary_path,
            vocabulary_path,
            sizeof(candidate->vocabulary_path)
        );
        strlcpy(
            candidate->calibration_path,
            calibration_path,
            sizeof(candidate->calibration_path)
        );
        strlcpy(
            candidate->checkout_signature,
            checkout_signature,
            sizeof(candidate->checkout_signature)
        );
        candidate->vocabulary = evoke_p2_create_lexical_hash(
            context,
            lexical_dims
        );
        evoke_p2_load_lexical_vocabulary(candidate, vocabulary_path);
        evoke_p2_load_calibration(candidate, calibration_path);
        candidate->term_pattern = uregex_openC(
            "[\\p{L}\\p{N}]+",
            0,
            NULL,
            &status
        );
        if (U_FAILURE(status) || candidate->term_pattern == NULL)
        {
            ereport(
                ERROR,
                (
                    errmsg("could not compile P2 lexical term pattern"),
                    errdetail("ICU error=%s", u_errorName(status))
                )
            );
        }
        MemoryContextSwitchTo(old_context);
    }
    PG_CATCH();
    {
        MemoryContextSwitchTo(old_context);
        if (candidate != NULL && candidate->term_pattern != NULL)
        {
            uregex_close(candidate->term_pattern);
        }
        MemoryContextDelete(context);
        PG_RE_THROW();
    }
    PG_END_TRY();

    evoke_p2_compiler_cache = (EvokeP2CompilerCache *) candidate;
    return evoke_p2_compiler_cache;
}

bool
evoke_p2_lookup_lexical_atom_id(
    const char *lexical_vocabulary_path,
    const char *calibration_path,
    const char *checkout_signature,
    int32 lexical_dims,
    int32 total_dims,
    const char *token,
    uint32 *atom_id_out
)
{
    EvokeP2CompilerCache *cache;
    EvokeP2TokenKey key;
    EvokeP2LexicalEntry *entry;

    if (token == NULL || token[0] == '\0' || atom_id_out == NULL)
    {
        return false;
    }
    if (checkout_signature == NULL ||
        strlen(checkout_signature) != EVOKE_P2_CACHE_IDENTITY_BYTES - 1)
    {
        ereport(ERROR, (errmsg("invalid P2 compiler cache identity")));
    }
    cache = evoke_p2_get_compiler(
        lexical_vocabulary_path,
        calibration_path,
        checkout_signature,
        lexical_dims,
        total_dims
    );
    key = evoke_p2_token_key(token, strlen(token));
    entry = hash_search(cache->vocabulary, &key, HASH_FIND, NULL);
    if (entry == NULL)
    {
        return false;
    }
    *atom_id_out = (uint32) entry->token_id;
    return true;
}

static UChar *
evoke_p2_lower_utf16(const UChar *input, int32 input_length, int32 *length_out)
{
    UErrorCode status = U_ZERO_ERROR;
    UChar *output;
    int32 output_length;

    output_length = u_strToLower(
        NULL,
        0,
        input,
        input_length,
        "",
        &status
    );
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status))
    {
        ereport(ERROR, (errmsg("could not size lower-cased P2 query")));
    }
    status = U_ZERO_ERROR;
    output = palloc(sizeof(UChar) * (Size) (output_length + 1));
    output_length = u_strToLower(
        output,
        output_length + 1,
        input,
        input_length,
        "",
        &status
    );
    if (U_FAILURE(status))
    {
        ereport(ERROR, (errmsg("could not lower-case P2 query")));
    }
    *length_out = output_length;
    return output;
}

static int
evoke_p2_compare_lexical(const void *left, const void *right)
{
    const EvokeP2ActiveLexical *left_value = left;
    const EvokeP2ActiveLexical *right_value = right;

    return (left_value->atom_id > right_value->atom_id) -
        (left_value->atom_id < right_value->atom_id);
}

static int
evoke_p2_compare_semantic(const void *left, const void *right)
{
    const EvokeP2ActiveSemantic *left_value = left;
    const EvokeP2ActiveSemantic *right_value = right;

    return (left_value->atom_id > right_value->atom_id) -
        (left_value->atom_id < right_value->atom_id);
}

static EvokeP2ActiveLexical *
evoke_p2_lexical_terms(
    EvokeP2CompilerCache *cache,
    const char *query_text,
    int32 *active_count_out
)
{
    UChar *query_utf16;
    UChar *lowered;
    int32 query_length;
    int32 lowered_length;
    UErrorCode status = U_ZERO_ERROR;
    EvokeP2ActiveLexical *active;
    int32 active_count = 0;
    int32 term_count = 0;

    query_utf16 = evoke_p2_utf8_to_utf16(query_text, &query_length);
    lowered = evoke_p2_lower_utf16(query_utf16, query_length, &lowered_length);
    active = palloc0(
        sizeof(*active) * (Size) Max(lowered_length, 1)
    );
    uregex_setText(
        cache->term_pattern,
        lowered,
        lowered_length,
        &status
    );
    while (U_SUCCESS(status) && uregex_findNext(cache->term_pattern, &status))
    {
        int32 start = uregex_start(cache->term_pattern, 0, &status);
        int32 end = uregex_end(cache->term_pattern, 0, &status);
        Size token_length;
        char *token;
        EvokeP2TokenKey key;
        EvokeP2LexicalEntry *entry;
        int32 active_index = -1;

        if (U_FAILURE(status))
        {
            break;
        }
        term_count++;
        token = evoke_p2_utf16_slice_to_utf8(
            lowered,
            start,
            end,
            &token_length
        );
        key = evoke_p2_token_key(token, token_length);
        entry = hash_search(cache->vocabulary, &key, HASH_FIND, NULL);
        if (entry == NULL)
        {
            continue;
        }
        for (int32 i = 0; i < active_count; i++)
        {
            if (active[i].atom_id == entry->token_id)
            {
                active_index = i;
                break;
            }
        }
        if (active_index < 0)
        {
            active_index = active_count++;
            active[active_index].atom_id = entry->token_id;
        }
        active[active_index].frequency += 1.0f;
    }
    if (U_FAILURE(status))
    {
        ereport(
            ERROR,
            (
                errmsg("P2 lexical query tokenization failed"),
                errdetail("ICU error=%s", u_errorName(status))
            )
        );
    }
    if (term_count == 0)
    {
        ereport(ERROR, (errmsg("P2 query has no searchable lexical terms")));
    }
    qsort(active, active_count, sizeof(*active), evoke_p2_compare_lexical);
    *active_count_out = active_count;
    return active;
}

void
evoke_p2_compile_unified_atoms(
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
)
{
    EvokeP2CompilerCache *cache;
    EvokeP2ActiveLexical *lexical;
    EvokeP2ActiveSemantic *semantic;
    int32 lexical_count;
    int32 active_semantic_count = 0;
    int32 semantic_dims;
    double raw_scale;

    if (semantic_count <= 0 || lexical_dims <= 0 || total_dims <= lexical_dims)
    {
        ereport(ERROR, (errmsg("invalid P2 compiler dimensions")));
    }
    if (
        !isfinite(global_scale) || global_scale <= 0.0 ||
        !isfinite(multiplier) || multiplier <= 0.0 ||
        !isfinite(clip_min) || clip_min <= 0.0 ||
        !isfinite(clip_max) || clip_max < clip_min
    )
    {
        ereport(ERROR, (errmsg("invalid P2 compiler calibration")));
    }
    if (checkout_signature == NULL ||
        strlen(checkout_signature) != EVOKE_P2_CACHE_IDENTITY_BYTES - 1)
    {
        ereport(ERROR, (errmsg("invalid P2 compiler cache identity")));
    }
    cache = evoke_p2_get_compiler(
        lexical_vocabulary_path,
        calibration_path,
        checkout_signature,
        lexical_dims,
        total_dims
    );
    lexical = evoke_p2_lexical_terms(cache, query_text, &lexical_count);
    semantic_dims = total_dims - lexical_dims;
    semantic = palloc(sizeof(*semantic) * (Size) semantic_count);
    memset(output, 0, sizeof(*output));

    for (int32 i = 0; i < lexical_count; i++)
    {
        output->lexical_proxy +=
            (double) lexical[i].frequency * cache->rms[lexical[i].atom_id];
    }
    for (int32 i = 0; i < semantic_count; i++)
    {
        int64 atom_id = semantic_ids[i];
        float4 weight = semantic_weights[i];

        if (atom_id < 0 || atom_id >= semantic_dims)
        {
            ereport(ERROR, (errmsg("P2 semantic atom id is out of range")));
        }
        if (!isfinite((double) weight) || weight < 0.0f)
        {
            ereport(ERROR, (errmsg("invalid P2 semantic atom weight")));
        }
        if (weight == 0.0f)
        {
            continue;
        }
        semantic[active_semantic_count].atom_id = (int32) atom_id;
        semantic[active_semantic_count].weight = weight;
        output->semantic_proxy += (double) weight *
            cache->rms[lexical_dims + (int32) atom_id];
        active_semantic_count++;
    }
    if (active_semantic_count == 0)
    {
        ereport(ERROR, (errmsg("P2 semantic encoder produced no atoms")));
    }
    qsort(
        semantic,
        active_semantic_count,
        sizeof(*semantic),
        evoke_p2_compare_semantic
    );
    for (int32 i = 1; i < active_semantic_count; i++)
    {
        if (semantic[i - 1].atom_id == semantic[i].atom_id)
        {
            ereport(ERROR, (errmsg("duplicate P2 semantic atom id")));
        }
    }

    if (output->lexical_proxy > 0.0 && output->semantic_proxy > 0.0)
    {
        raw_scale = output->lexical_proxy / output->semantic_proxy;
    }
    else
    {
        raw_scale = global_scale;
    }
    output->query_scale = raw_scale * multiplier;
    output->query_scale = Max(
        global_scale * clip_min,
        Min(global_scale * clip_max, output->query_scale)
    );
    output->atom_count = lexical_count + active_semantic_count;
    output->lexical_atom_count = lexical_count;
    output->semantic_atom_count = active_semantic_count;
    output->atom_ids = palloc(sizeof(int32) * (Size) output->atom_count);
    output->atom_weights = palloc(
        sizeof(float4) * (Size) output->atom_count
    );
    for (int32 i = 0; i < lexical_count; i++)
    {
        output->atom_ids[i] = lexical[i].atom_id;
        output->atom_weights[i] = lexical[i].frequency;
    }
    for (int32 i = 0; i < active_semantic_count; i++)
    {
        int32 output_index = lexical_count + i;
        double calibrated =
            (double) semantic[i].weight * output->query_scale;

        if (!isfinite(calibrated) || calibrated <= 0.0 || calibrated > FLT_MAX)
        {
            ereport(ERROR, (errmsg("invalid calibrated P2 semantic weight")));
        }
        output->atom_ids[output_index] =
            lexical_dims + semantic[i].atom_id;
        output->atom_weights[output_index] = (float4) calibrated;
    }
}

void
evoke_p2_compile_document_semantic_atoms(
    const int64 *semantic_ids,
    const float *semantic_weights,
    int32 semantic_count,
    int32 lexical_dims,
    int32 total_dims,
    EvokeP2CompiledAtoms *output
)
{
    EvokeP2ActiveSemantic *semantic;
    int32 active_semantic_count = 0;
    int32 semantic_dims;

    if (semantic_count <= 0 || lexical_dims <= 0 || total_dims <= lexical_dims)
    {
        ereport(ERROR, (errmsg("invalid P2 document compiler dimensions")));
    }
    semantic_dims = total_dims - lexical_dims;
    semantic = palloc(sizeof(*semantic) * (Size) semantic_count);
    memset(output, 0, sizeof(*output));

    for (int32 i = 0; i < semantic_count; i++)
    {
        int64 atom_id = semantic_ids[i];
        float4 weight = semantic_weights[i];

        if (atom_id < 0 || atom_id >= semantic_dims)
        {
            ereport(
                ERROR,
                (errmsg("P2 document semantic atom id is out of range"))
            );
        }
        if (!isfinite((double) weight) || weight < 0.0f)
        {
            ereport(ERROR, (errmsg("invalid P2 document semantic weight")));
        }
        if (weight == 0.0f)
        {
            continue;
        }
        semantic[active_semantic_count].atom_id = (int32) atom_id;
        semantic[active_semantic_count].weight = weight;
        output->semantic_proxy += (double) weight;
        active_semantic_count++;
    }
    if (active_semantic_count == 0)
    {
        ereport(
            ERROR,
            (errmsg("P2 document semantic encoder produced no atoms"))
        );
    }
    qsort(
        semantic,
        active_semantic_count,
        sizeof(*semantic),
        evoke_p2_compare_semantic
    );
    for (int32 i = 1; i < active_semantic_count; i++)
    {
        if (semantic[i - 1].atom_id == semantic[i].atom_id)
        {
            ereport(
                ERROR,
                (errmsg("duplicate P2 document semantic atom id"))
            );
        }
    }

    output->atom_count = active_semantic_count;
    output->semantic_atom_count = active_semantic_count;
    output->query_scale = 1.0;
    output->atom_ids = palloc(sizeof(int32) * (Size) active_semantic_count);
    output->atom_weights = palloc(
        sizeof(float4) * (Size) active_semantic_count
    );
    for (int32 i = 0; i < active_semantic_count; i++)
    {
        output->atom_ids[i] = lexical_dims + semantic[i].atom_id;
        output->atom_weights[i] = semantic[i].weight;
    }
    pfree(semantic);
}
