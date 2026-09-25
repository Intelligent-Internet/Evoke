#include "postgres.h"

#include "miscadmin.h"

#include "access/htup_details.h"
#include "catalog/namespace.h"
#include "catalog/pg_type_d.h"
#include "executor/spi.h"
#include "storage/itemptr.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/pg_locale.h"
#include "utils/rel.h"
#include "utils/typcache.h"

#include "evoke_filter.h"
#include "evoke_scope.h"
#include "evoke_segment_pages.h"

#define EVOKE_FILTER_MAX_PREDICATES 32
#define EVOKE_FILTER_MAX_VALUES 4096
#define EVOKE_FILTER_FETCH_ROWS 4096L
#define EVOKE_FILTER_MATERIALIZE_MAX_ROWS UINT64_C(4194304)
#define EVOKE_FILTER_MATERIALIZE_ROW_BUDGET_BYTES UINT64_C(24)
#define EVOKE_FILTER_SCOPE_POSTING_READ_DOCUMENTS 16384U
#define EVOKE_FILTER_SCOPE_MAX_VALUE_BYTES (1024U * 1024U)
#define EVOKE_FILTER_SCOPE_RANGE_MAX_VALUES 262144U
#define EVOKE_FILTER_SCOPE_RANGE_MAX_DICTIONARY_BYTES (16U * 1024U * 1024U)
#define EVOKE_FILTER_SCOPE_ILIKE_MAX_COMPARISONS UINT64_C(4194304)
#define EVOKE_FILTER_SCOPE_ILIKE_VALUE_WINDOW 4096U
#define EVOKE_FILTER_SCOPE_ILIKE_DICTIONARY_WINDOW_BYTES \
    (4U * 1024U * 1024U)
#define EVOKE_FILTER_SCOPE_GRAM_CACHE_BLOCKS 256U
#define EVOKE_FILTER_SCOPE_PATTERN_GRAMS 8U

static char *
evoke_filter_json_key(const JsonbValue *value)
{
    if (value == NULL || value->type != jbvString ||
        value->val.string.len <= 0)
    {
        ereport(ERROR, (errmsg("evoke filter keys must be non-empty strings")));
    }
    return pnstrdup(value->val.string.val, value->val.string.len);
}

static bool
evoke_filter_string_equals(const char *value, const char *expected)
{
    return value != NULL && expected != NULL &&
        strcmp(value, expected) == 0;
}

bool
evoke_filter_is_range_only(Jsonb *filters)
{
    JsonbIterator *iterator;
    JsonbValue item;
    JsonbIteratorToken token;
    uint32 predicate_count = 0;

    if (filters == NULL || !JB_ROOT_IS_OBJECT(filters) ||
        JB_ROOT_COUNT(filters) == 0)
    {
        return false;
    }
    iterator = JsonbIteratorInit(&filters->root);
    if (JsonbIteratorNext(&iterator, &item, true) != WJB_BEGIN_OBJECT)
    {
        return false;
    }
    while ((token = JsonbIteratorNext(&iterator, &item, true)) !=
           WJB_END_OBJECT)
    {
        JsonbValue predicate;
        JsonbIterator *predicate_iterator;
        JsonbValue operation;

        if (token != WJB_KEY ||
            JsonbIteratorNext(&iterator, &predicate, true) != WJB_VALUE ||
            predicate.type != jbvBinary ||
            !JsonContainerIsObject(predicate.val.binary.data) ||
            JsonContainerSize(predicate.val.binary.data) != 1)
        {
            return false;
        }
        predicate_iterator = JsonbIteratorInit(predicate.val.binary.data);
        if (JsonbIteratorNext(
                &predicate_iterator,
                &operation,
                true) != WJB_BEGIN_OBJECT ||
            JsonbIteratorNext(
                &predicate_iterator,
                &operation,
                true) != WJB_KEY ||
            operation.type != jbvString ||
            operation.val.string.len != strlen("range") ||
            memcmp(
                operation.val.string.val,
                "range",
                operation.val.string.len
            ) != 0)
        {
            return false;
        }
        predicate_count++;
    }
    return predicate_count == JB_ROOT_COUNT(filters);
}

uint64
evoke_filter_tid_key(const ItemPointerData *tid)
{
    return ((uint64) ItemPointerGetBlockNumber(tid) << 16) |
        (uint64) ItemPointerGetOffsetNumber(tid);
}

static int
evoke_filter_compare_tid_keys(const void *left, const void *right)
{
    uint64 left_key = *(const uint64 *) left;
    uint64 right_key = *(const uint64 *) right;

    if (left_key < right_key)
    {
        return -1;
    }
    if (left_key > right_key)
    {
        return 1;
    }
    return 0;
}

size_t
evoke_filter_sort_unique_tid_keys(uint64 *keys, size_t key_count)
{
    size_t retained = 0;

    if (key_count > 0 && keys == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke TID set")));
    }
    if (key_count == 0)
    {
        return 0;
    }
    qsort(
        keys,
        key_count,
        sizeof(*keys),
        evoke_filter_compare_tid_keys
    );
    for (size_t index = 0; index < key_count; index++)
    {
        if (retained == 0 || keys[index] != keys[retained - 1])
        {
            keys[retained++] = keys[index];
        }
    }
    return retained;
}

typedef struct evoke_filter_tid_candidate_ref
{
    uint64 key;
    size_t candidate_index;
} evoke_filter_tid_candidate_ref;

static int
evoke_filter_compare_tid_candidate_refs(const void *left, const void *right)
{
    const evoke_filter_tid_candidate_ref *a = left;
    const evoke_filter_tid_candidate_ref *b = right;

    if (a->key != b->key)
    {
        return a->key < b->key ? -1 : 1;
    }
    return a->candidate_index < b->candidate_index
        ? -1
        : a->candidate_index > b->candidate_index;
}

static void
evoke_filter_append_spi_tid_rows(
    MemoryContext caller_context,
    uint64 **keys,
    size_t *capacity,
    size_t *count
)
{
    size_t processed = (size_t) SPI_processed;

    if (SPI_processed > (uint64) SIZE_MAX ||
        *count > SIZE_MAX - processed ||
        *count + processed > MaxAllocSize / sizeof(**keys))
    {
        ereport(ERROR, (errmsg("evoke filter result is too large")));
    }
    if (*count + processed > *capacity)
    {
        size_t required = *count + processed;
        size_t next = *capacity == 0 ? 4096 : *capacity;

        while (next < required)
        {
            if (next > MaxAllocSize / sizeof(**keys) / 2)
            {
                next = required;
                break;
            }
            next *= 2;
        }
        *keys = *keys == NULL
            ? MemoryContextAlloc(caller_context, sizeof(**keys) * next)
            : repalloc(*keys, sizeof(**keys) * next);
        *capacity = next;
    }
    for (uint64 row = 0; row < SPI_processed; row++)
    {
        bool is_null;
        Datum datum = SPI_getbinval(
            SPI_tuptable->vals[row],
            SPI_tuptable->tupdesc,
            1,
            &is_null
        );
        ItemPointer tid;

        if (is_null)
        {
            ereport(ERROR, (errmsg("evoke filter returned a null TID")));
        }
        tid = DatumGetItemPointer(datum);
        if (!ItemPointerIsValid(tid))
        {
            ereport(ERROR, (errmsg("evoke filter returned an invalid TID")));
        }
        (*keys)[(*count)++] = evoke_filter_tid_key(tid);
    }
}

static void
evoke_filter_require_array(
    const JsonbValue *value,
    const char *column,
    const char *operation
)
{
    if (value == NULL || value->type != jbvBinary ||
        !JsonContainerIsArray(value->val.binary.data))
    {
        ereport(
            ERROR,
            (
                errmsg("invalid evoke filter for column %s", column),
                errdetail("Operation %s requires a JSON array.", operation)
            )
        );
    }
    if (JsonContainerSize(value->val.binary.data) > EVOKE_FILTER_MAX_VALUES)
    {
        ereport(
            ERROR,
            (
                errmsg("invalid evoke filter for column %s", column),
                errdetail(
                    "Operation %s accepts at most %d values.",
                    operation,
                    EVOKE_FILTER_MAX_VALUES
                )
            )
        );
    }
}

static void
evoke_filter_require_string(
    const JsonbValue *value,
    const char *column,
    const char *operation
)
{
    if (value == NULL || value->type != jbvString)
    {
        ereport(
            ERROR,
            (
                errmsg("invalid evoke filter for column %s", column),
                errdetail("Operation %s requires a JSON string.", operation)
            )
        );
    }
}

static void
evoke_filter_require_string_array(
    const JsonbValue *value,
    const char *column,
    const char *operation
)
{
    JsonbIterator *iterator;
    JsonbValue item;
    JsonbIteratorToken token;

    evoke_filter_require_array(value, column, operation);
    iterator = JsonbIteratorInit(value->val.binary.data);
    token = JsonbIteratorNext(&iterator, &item, true);
    if (token != WJB_BEGIN_ARRAY)
    {
        ereport(ERROR, (errmsg("invalid evoke string filter array")));
    }
    while ((token = JsonbIteratorNext(&iterator, &item, true)) !=
           WJB_END_ARRAY)
    {
        if (token != WJB_ELEM || item.type != jbvString)
        {
            ereport(
                ERROR,
                (
                    errmsg("invalid evoke filter for column %s", column),
                    errdetail(
                        "Operation %s requires an array of JSON strings.",
                        operation
                    )
                )
            );
        }
    }
}

static void
evoke_filter_require_range(
    const JsonbValue *value,
    const char *column
)
{
    JsonbIterator *iterator;
    JsonbValue item;
    JsonbIteratorToken token;
    int bound_count = 0;

    if (value == NULL || value->type != jbvBinary ||
        !JsonContainerIsObject(value->val.binary.data))
    {
        ereport(
            ERROR,
            (
                errmsg("invalid evoke filter for column %s", column),
                errdetail("Operation range requires a JSON object.")
            )
        );
    }
    iterator = JsonbIteratorInit(value->val.binary.data);
    token = JsonbIteratorNext(&iterator, &item, true);
    if (token != WJB_BEGIN_OBJECT)
    {
        ereport(ERROR, (errmsg("invalid evoke range filter")));
    }
    while ((token = JsonbIteratorNext(&iterator, &item, true)) !=
           WJB_END_OBJECT)
    {
        char *bound;

        if (token != WJB_KEY)
        {
            ereport(ERROR, (errmsg("invalid evoke range filter")));
        }
        bound = evoke_filter_json_key(&item);
        if (!evoke_filter_string_equals(bound, "gt") &&
            !evoke_filter_string_equals(bound, "gte") &&
            !evoke_filter_string_equals(bound, "lt") &&
            !evoke_filter_string_equals(bound, "lte"))
        {
            ereport(
                ERROR,
                (
                    errmsg("invalid evoke range bound %s", bound),
                    errhint("Use gt, gte, lt, or lte.")
                )
            );
        }
        pfree(bound);
        token = JsonbIteratorNext(&iterator, &item, true);
        if (token != WJB_VALUE || item.type == jbvNull ||
            item.type == jbvBinary)
        {
            ereport(
                ERROR,
                (
                    errmsg("invalid evoke range filter for column %s", column),
                    errdetail("Range bounds must be non-null scalar values.")
                )
            );
        }
        bound_count++;
    }
    if (bound_count == 0)
    {
        ereport(
            ERROR,
            (
                errmsg("invalid evoke range filter for column %s", column),
                errdetail("At least one range bound is required.")
            )
        );
    }
}

static void
evoke_filter_append_typed_value(
    StringInfo query,
    const char *qualified_relation,
    const char *quoted_column,
    const char *column_literal,
    const char *json_expression
)
{
    appendStringInfo(
        query,
        "(jsonb_populate_record(NULL::%s, "
        "jsonb_build_object(%s, %s))).%s",
        qualified_relation,
        column_literal,
        json_expression,
        quoted_column
    );
}

static void
evoke_filter_append_range(
    StringInfo query,
    const char *qualified_relation,
    const char *quoted_column,
    const char *column_literal,
    JsonbContainer *range
)
{
    JsonbIterator *iterator = JsonbIteratorInit(range);
    JsonbValue item;
    JsonbIteratorToken token;
    bool first = true;

    token = JsonbIteratorNext(&iterator, &item, true);
    if (token != WJB_BEGIN_OBJECT)
    {
        ereport(ERROR, (errmsg("invalid evoke range filter")));
    }
    while ((token = JsonbIteratorNext(&iterator, &item, true)) !=
           WJB_END_OBJECT)
    {
        char *bound = evoke_filter_json_key(&item);
        const char *sql_operator;
        char *bound_literal;
        char *json_expression;

        token = JsonbIteratorNext(&iterator, &item, true);
        if (token != WJB_VALUE)
        {
            ereport(ERROR, (errmsg("invalid evoke range filter")));
        }
        if (evoke_filter_string_equals(bound, "gt"))
        {
            sql_operator = ">";
        }
        else if (evoke_filter_string_equals(bound, "gte"))
        {
            sql_operator = ">=";
        }
        else if (evoke_filter_string_equals(bound, "lt"))
        {
            sql_operator = "<";
        }
        else
        {
            sql_operator = "<=";
        }
        if (!first)
        {
            appendStringInfoString(query, " AND ");
        }
        first = false;
        appendStringInfo(query, "source.%s %s ", quoted_column, sql_operator);
        bound_literal = quote_literal_cstr(bound);
        json_expression = psprintf(
            "$1 -> %s -> 'range' -> %s",
            column_literal,
            bound_literal
        );
        evoke_filter_append_typed_value(
            query,
            qualified_relation,
            quoted_column,
            column_literal,
            json_expression
        );
        pfree(json_expression);
        pfree(bound_literal);
        pfree(bound);
    }
}

static char *
evoke_filter_find_array_trigram_expression(
    Relation heap_relation,
    AttrNumber attribute_number,
    char **predicate_out
)
{
    static const char query[] =
        "SELECT pg_catalog.pg_get_expr(i.indexprs, i.indrelid, true), "
        "pg_catalog.pg_get_expr(i.indpred, i.indrelid, true) "
        "FROM pg_catalog.pg_index AS i "
        "JOIN pg_catalog.pg_class AS index_relation "
        "ON index_relation.oid = i.indexrelid "
        "JOIN pg_catalog.pg_am AS access_method "
        "ON access_method.oid = index_relation.relam "
        "JOIN pg_catalog.pg_opclass AS operator_class "
        "ON operator_class.oid = i.indclass[0] "
        "WHERE i.indrelid = $1 "
        "AND i.indisvalid AND i.indisready AND i.indislive "
        "AND i.indnkeyatts = 1 AND i.indexprs IS NOT NULL "
        "AND access_method.amname = 'gin' "
        "AND operator_class.opcname = 'gin_trgm_ops' "
        "AND EXISTS ("
        "SELECT 1 FROM pg_catalog.pg_depend AS dependency "
        "WHERE dependency.classid = 'pg_catalog.pg_class'::regclass "
        "AND dependency.objid = i.indexrelid "
        "AND dependency.refclassid = 'pg_catalog.pg_class'::regclass "
        "AND dependency.refobjid = i.indrelid "
        "AND dependency.refobjsubid = $2) "
        "AND NOT EXISTS ("
        "SELECT 1 FROM pg_catalog.pg_depend AS dependency "
        "WHERE dependency.classid = 'pg_catalog.pg_class'::regclass "
        "AND dependency.objid = i.indexrelid "
        "AND dependency.refclassid = 'pg_catalog.pg_class'::regclass "
        "AND dependency.refobjid = i.indrelid "
        "AND dependency.refobjsubid <> 0 "
        "AND dependency.refobjsubid <> $2) "
        "ORDER BY i.indexrelid LIMIT 1";
    Oid parameter_types[2] = {OIDOID, INT2OID};
    Datum parameter_values[2] = {
        ObjectIdGetDatum(RelationGetRelid(heap_relation)),
        Int16GetDatum(attribute_number),
    };
    char parameter_nulls[2] = {' ', ' '};
    char *expression = NULL;
    int status;

    *predicate_out = NULL;
    status = SPI_execute_with_args(
        query,
        lengthof(parameter_types),
        parameter_types,
        parameter_values,
        parameter_nulls,
        true,
        1
    );
    if (status != SPI_OK_SELECT)
    {
        ereport(
            ERROR,
            (errmsg("could not inspect evoke filter support indexes"))
        );
    }
    if (SPI_processed == 1)
    {
        expression = SPI_getvalue(
            SPI_tuptable->vals[0],
            SPI_tuptable->tupdesc,
            1
        );
        *predicate_out = SPI_getvalue(
            SPI_tuptable->vals[0],
            SPI_tuptable->tupdesc,
            2
        );
    }
    if (SPI_tuptable != NULL)
    {
        SPI_freetuptable(SPI_tuptable);
    }
    return expression;
}

static void
evoke_filter_append_predicate(
    StringInfo query,
    Relation heap_relation,
    const char *qualified_relation,
    const char *column,
    const char *operation,
    const JsonbValue *operand
)
{
    AttrNumber attribute_number;
    Form_pg_attribute attribute;
    Oid element_type = InvalidOid;
    char type_category = TYPCATEGORY_INVALID;
    bool type_preferred = false;
    bool string_array = false;
    const char *quoted_column;
    char *column_literal;
    char *operation_literal;
    char *json_expression;
    char *array_candidate_expression = NULL;
    char *array_candidate_predicate = NULL;

    attribute_number = get_attnum(RelationGetRelid(heap_relation), column);
    if (attribute_number <= 0)
    {
        ereport(
            ERROR,
            (
                errmsg("evoke filter column %s does not exist", column),
                errdetail(
                    "Filters may reference ordinary columns of relation %s.",
                    RelationGetRelationName(heap_relation)
                )
            )
        );
    }
    attribute = TupleDescAttr(
        RelationGetDescr(heap_relation),
        attribute_number - 1
    );
    if (attribute->attisdropped)
    {
        ereport(ERROR, (errmsg("evoke filter column %s was dropped", column)));
    }
    element_type = get_element_type(attribute->atttypid);
    if (evoke_filter_string_equals(operation, "overlap") &&
        element_type == InvalidOid)
    {
        ereport(
            ERROR,
            (
                errmsg("invalid evoke overlap filter for column %s", column),
                errdetail("The source column must have an array type.")
            )
        );
    }
    if (evoke_filter_string_equals(operation, "ilike") ||
        evoke_filter_string_equals(operation, "ilike_any"))
    {
        Oid comparison_type = element_type == InvalidOid
            ? attribute->atttypid
            : element_type;

        get_type_category_preferred(
            getBaseType(comparison_type),
            &type_category,
            &type_preferred
        );
        if (type_category != TYPCATEGORY_STRING)
        {
            ereport(
                ERROR,
                (
                    errmsg("invalid evoke text filter for column %s", column),
                    errdetail("The source column must have a string type.")
                )
            );
        }
        string_array = element_type != InvalidOid;
        if (string_array)
        {
            array_candidate_expression =
                evoke_filter_find_array_trigram_expression(
                    heap_relation,
                    attribute_number,
                    &array_candidate_predicate
                );
        }
    }

    quoted_column = quote_identifier(column);
    column_literal = quote_literal_cstr(column);
    operation_literal = quote_literal_cstr(operation);
    json_expression = psprintf(
        "$1 -> %s -> %s",
        column_literal,
        operation_literal
    );

    if (evoke_filter_string_equals(operation, "eq"))
    {
        if (operand->type == jbvNull)
        {
            appendStringInfo(query, "source.%s IS NULL", quoted_column);
        }
        else
        {
            appendStringInfo(query, "source.%s = ", quoted_column);
            evoke_filter_append_typed_value(
                query,
                qualified_relation,
                quoted_column,
                column_literal,
                json_expression
            );
        }
    }
    else if (evoke_filter_string_equals(operation, "in"))
    {
        evoke_filter_require_array(operand, column, operation);
        if (JsonContainerSize(operand->val.binary.data) == 0)
        {
            appendStringInfoString(query, "FALSE");
        }
        else
        {
            appendStringInfo(query, "source.%s = ANY (ARRAY(", quoted_column);
            appendStringInfo(
                query,
                "SELECT (jsonb_populate_record(NULL::%s, "
                "jsonb_build_object(%s, item.value))).%s "
                "FROM jsonb_array_elements(%s) AS item(value)",
                qualified_relation,
                column_literal,
                quoted_column,
                json_expression
            );
            appendStringInfoString(query, "))");
        }
    }
    else if (evoke_filter_string_equals(operation, "overlap"))
    {
        evoke_filter_require_array(operand, column, operation);
        if (JsonContainerSize(operand->val.binary.data) == 0)
        {
            appendStringInfoString(query, "FALSE");
        }
        else
        {
            /*
             * Array overlap implies a non-empty source array.  State that
             * implication explicitly so PostgreSQL can use partial GIN
             * indexes declared with cardinality(column) > 0.
             */
            appendStringInfo(
                query,
                "(cardinality(source.%s) > 0 AND source.%s && ",
                quoted_column,
                quoted_column
            );
            evoke_filter_append_typed_value(
                query,
                qualified_relation,
                quoted_column,
                column_literal,
                json_expression
            );
            appendStringInfoChar(query, ')');
        }
    }
    else if (evoke_filter_string_equals(operation, "ilike"))
    {
        evoke_filter_require_string(operand, column, operation);
        if (string_array)
        {
            appendStringInfoChar(query, '(');
            if (array_candidate_expression != NULL)
            {
                if (array_candidate_predicate != NULL)
                {
                    appendStringInfo(
                        query,
                        "(%s) AND ",
                        array_candidate_predicate
                    );
                }
                appendStringInfo(
                    query,
                    "(%s)::text ILIKE (%s #>> '{}') AND ",
                    array_candidate_expression,
                    json_expression
                );
            }
            appendStringInfo(
                query,
                "EXISTS (SELECT 1 FROM unnest(source.%s) AS "
                "source_item(value) WHERE source_item.value::text ILIKE "
                "(%s #>> '{}'))",
                quoted_column,
                json_expression
            );
            appendStringInfoChar(query, ')');
        }
        else
        {
            appendStringInfo(
                query,
                "source.%s::text ILIKE (%s #>> '{}')",
                quoted_column,
                json_expression
            );
        }
    }
    else if (evoke_filter_string_equals(operation, "ilike_any"))
    {
        evoke_filter_require_string_array(operand, column, operation);
        if (JsonContainerSize(operand->val.binary.data) == 0)
        {
            appendStringInfoString(query, "FALSE");
        }
        else
        {
            if (string_array)
            {
                appendStringInfoChar(query, '(');
                if (array_candidate_expression != NULL)
                {
                    if (array_candidate_predicate != NULL)
                    {
                        appendStringInfo(
                            query,
                            "(%s) AND ",
                            array_candidate_predicate
                        );
                    }
                    appendStringInfo(
                        query,
                        "(%s)::text ILIKE ANY (ARRAY(SELECT "
                        "item.value #>> '{}' FROM jsonb_array_elements(%s) "
                        "AS item(value))) AND ",
                        array_candidate_expression,
                        json_expression
                    );
                }
                appendStringInfo(
                    query,
                    "EXISTS (SELECT 1 FROM unnest(source.%s) AS "
                    "source_item(value) WHERE source_item.value::text "
                    "ILIKE ANY (ARRAY(SELECT item.value #>> '{}' FROM "
                    "jsonb_array_elements(%s) AS item(value))))",
                    quoted_column,
                    json_expression
                );
                appendStringInfoChar(query, ')');
            }
            else
            {
                appendStringInfo(
                    query,
                    "source.%s::text ILIKE ANY (ARRAY(SELECT "
                    "item.value #>> '{}' FROM jsonb_array_elements(%s) "
                    "AS item(value)))",
                    quoted_column,
                    json_expression
                );
            }
        }
    }
    else
    {
        evoke_filter_require_range(operand, column);
        appendStringInfoChar(query, '(');
        evoke_filter_append_range(
            query,
            qualified_relation,
            quoted_column,
            column_literal,
            operand->val.binary.data
        );
        appendStringInfoChar(query, ')');
    }

    pfree(json_expression);
    pfree(operation_literal);
    pfree(column_literal);
    if (array_candidate_predicate != NULL)
    {
        pfree(array_candidate_predicate);
    }
    if (array_candidate_expression != NULL)
    {
        pfree(array_candidate_expression);
    }
}

static char *
evoke_filter_build_query(
    Relation heap_relation,
    Jsonb *filters,
    bool candidate_limited
)
{
    JsonbIterator *iterator;
    JsonbValue item;
    JsonbIteratorToken token;
    StringInfoData query;
    char *namespace_name;
    char *qualified_relation;
    int predicate_count = 0;

    if (filters == NULL || !JB_ROOT_IS_OBJECT(filters) ||
        JB_ROOT_COUNT(filters) == 0)
    {
        ereport(
            ERROR,
            (
                errmsg("evoke filters must be a non-empty JSON object"),
                errhint(
                    "Use {\"column\": {\"eq\": value}} or call "
                    "evoke_query without filters."
                )
            )
        );
    }
    if (JB_ROOT_COUNT(filters) > EVOKE_FILTER_MAX_PREDICATES)
    {
        ereport(
            ERROR,
            (
                errmsg("too many evoke filter predicates"),
                errdetail(
                    "At most %d predicates are supported.",
                    EVOKE_FILTER_MAX_PREDICATES
                )
            )
        );
    }

    namespace_name = get_namespace_name(RelationGetNamespace(heap_relation));
    qualified_relation = quote_qualified_identifier(
        namespace_name,
        RelationGetRelationName(heap_relation)
    );
    initStringInfo(&query);
    if (candidate_limited)
    {
        appendStringInfo(
            &query,
            "SELECT source.ctid FROM unnest($2) AS candidate(ctid) "
            "JOIN ONLY %s AS source ON source.ctid = candidate.ctid "
            "WHERE ",
            qualified_relation
        );
    }
    else
    {
        appendStringInfo(
            &query,
            "SELECT source.ctid FROM ONLY %s AS source WHERE ",
            qualified_relation
        );
    }

    iterator = JsonbIteratorInit(&filters->root);
    token = JsonbIteratorNext(&iterator, &item, true);
    if (token != WJB_BEGIN_OBJECT)
    {
        ereport(ERROR, (errmsg("invalid evoke filter object")));
    }
    while ((token = JsonbIteratorNext(&iterator, &item, true)) !=
           WJB_END_OBJECT)
    {
        JsonbValue predicate;
        JsonbIterator *predicate_iterator;
        JsonbIteratorToken predicate_token;
        JsonbValue operation_value;
        JsonbValue operand;
        char *column;
        char *operation;

        if (token != WJB_KEY)
        {
            ereport(ERROR, (errmsg("invalid evoke filter object")));
        }
        column = evoke_filter_json_key(&item);
        token = JsonbIteratorNext(&iterator, &predicate, true);
        if (token != WJB_VALUE || predicate.type != jbvBinary ||
            !JsonContainerIsObject(predicate.val.binary.data) ||
            JsonContainerSize(predicate.val.binary.data) != 1)
        {
            ereport(
                ERROR,
                (
                    errmsg("invalid evoke filter for column %s", column),
                    errdetail("Each column must specify exactly one operation.")
                )
            );
        }
        predicate_iterator = JsonbIteratorInit(predicate.val.binary.data);
        predicate_token = JsonbIteratorNext(
            &predicate_iterator,
            &operation_value,
            true
        );
        if (predicate_token != WJB_BEGIN_OBJECT ||
            JsonbIteratorNext(
                &predicate_iterator,
                &operation_value,
                true
            ) != WJB_KEY)
        {
            ereport(ERROR, (errmsg("invalid evoke filter predicate")));
        }
        operation = evoke_filter_json_key(&operation_value);
        predicate_token = JsonbIteratorNext(
            &predicate_iterator,
            &operand,
            true
        );
        if (predicate_token != WJB_VALUE ||
            (!evoke_filter_string_equals(operation, "eq") &&
             !evoke_filter_string_equals(operation, "in") &&
             !evoke_filter_string_equals(operation, "overlap") &&
             !evoke_filter_string_equals(operation, "ilike") &&
             !evoke_filter_string_equals(operation, "ilike_any") &&
             !evoke_filter_string_equals(operation, "range")))
        {
            ereport(
                ERROR,
                (
                    errmsg("unsupported evoke filter operation %s", operation),
                    errhint(
                        "Use eq, in, overlap, ilike, ilike_any, or range."
                    )
                )
            );
        }
        if (predicate_count > 0)
        {
            appendStringInfoString(&query, " AND ");
        }
        evoke_filter_append_predicate(
            &query,
            heap_relation,
            qualified_relation,
            column,
            operation,
            &operand
        );
        predicate_count++;
        pfree(operation);
        pfree(column);
    }
    pfree(qualified_relation);
    pfree(namespace_name);
    return query.data;
}

bool
evoke_filter_collect_tid_keys(
    Relation heap_relation,
    Jsonb *filters,
    uint64 result_upper_bound,
    uint64 row_limit,
    uint64 **keys_out,
    size_t *key_count_out
)
{
    Oid parameter_types[1] = {JSONBOID};
    Datum parameter_values[1];
    char parameter_nulls[1] = {' '};
    SPIPlanPtr plan = NULL;
    Portal portal = NULL;
    char *query = NULL;
    uint64 *keys = NULL;
    size_t capacity = 0;
    size_t count = 0;
    uint64 materialize_row_limit = Min(
        EVOKE_FILTER_MATERIALIZE_MAX_ROWS,
        (uint64) Max(work_mem, 1) * UINT64_C(1024) /
            EVOKE_FILTER_MATERIALIZE_ROW_BUDGET_BYTES
    );
    bool materialize_result = row_limit == 0 && result_upper_bound > 0 &&
        result_upper_bound <= materialize_row_limit;
    bool complete = true;
    bool spi_connected = false;
    MemoryContext caller_context = CurrentMemoryContext;

    if (heap_relation == NULL || filters == NULL || keys_out == NULL ||
        key_count_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke structured filter request")));
    }
    *keys_out = NULL;
    *key_count_out = 0;
    parameter_values[0] = JsonbPGetDatum(filters);

    PG_TRY();
    {
        int status = SPI_connect();

        if (status != SPI_OK_CONNECT)
        {
            ereport(ERROR, (errmsg("SPI_connect failed: %d", status)));
        }
        spi_connected = true;
        {
            MemoryContext previous_context = MemoryContextSwitchTo(
                caller_context
            );

            query = evoke_filter_build_query(heap_relation, filters, false);
            if (row_limit > 0)
            {
                char *bounded_query;

                if (row_limit == UINT64_MAX)
                {
                    ereport(ERROR, (errmsg("invalid evoke filter row limit")));
                }
                bounded_query = psprintf(
                    "%s LIMIT " UINT64_FORMAT,
                    query,
                    row_limit + UINT64_C(1)
                );
                pfree(query);
                query = bounded_query;
            }
            MemoryContextSwitchTo(previous_context);
        }
        plan = SPI_prepare(query, 1, parameter_types);
        if (plan == NULL)
        {
            ereport(
                ERROR,
                (errmsg("could not prepare the evoke structured filter"))
            );
        }
        if (materialize_result)
        {
            status = SPI_execute_plan(
                plan,
                parameter_values,
                parameter_nulls,
                true,
                0
            );
            if (status != SPI_OK_SELECT)
            {
                ereport(
                    ERROR,
                    (errmsg("could not execute the evoke structured filter"))
                );
            }
            evoke_filter_append_spi_tid_rows(
                caller_context,
                &keys,
                &capacity,
                &count
            );
            if (SPI_tuptable != NULL)
            {
                SPI_freetuptable(SPI_tuptable);
            }
        }
        else
        {
            portal = SPI_cursor_open(
                NULL,
                plan,
                parameter_values,
                parameter_nulls,
                true
            );
            if (portal == NULL)
            {
                ereport(
                    ERROR,
                    (errmsg("could not execute the evoke structured filter"))
                );
            }
            for (;;)
            {
                uint64 fetch_rows = EVOKE_FILTER_FETCH_ROWS;

                if (row_limit > 0)
                {
                    uint64 remaining = count < row_limit
                        ? row_limit - count
                        : 0;

                    fetch_rows = Min(fetch_rows, remaining + UINT64_C(1));
                }
                SPI_cursor_fetch(portal, true, (long) fetch_rows);
                if (SPI_processed == 0)
                {
                    if (SPI_tuptable != NULL)
                    {
                        SPI_freetuptable(SPI_tuptable);
                    }
                    break;
                }
                evoke_filter_append_spi_tid_rows(
                    caller_context,
                    &keys,
                    &capacity,
                    &count
                );
                SPI_freetuptable(SPI_tuptable);
                if (row_limit > 0 && count > row_limit)
                {
                    complete = false;
                    break;
                }
                CHECK_FOR_INTERRUPTS();
            }
            SPI_cursor_close(portal);
            portal = NULL;
        }
        SPI_freeplan(plan);
        plan = NULL;
        SPI_finish();
        spi_connected = false;
    }
    PG_CATCH();
    {
        if (portal != NULL)
        {
            SPI_cursor_close(portal);
        }
        if (plan != NULL)
        {
            SPI_freeplan(plan);
        }
        if (spi_connected)
        {
            SPI_finish();
        }
        if (keys != NULL)
        {
            pfree(keys);
        }
        if (query != NULL)
        {
            pfree(query);
        }
        PG_RE_THROW();
    }
    PG_END_TRY();

    if (query != NULL)
    {
        pfree(query);
    }
    if (count > 1)
    {
        size_t unique_count = 1;

        qsort(keys, count, sizeof(*keys), evoke_filter_compare_tid_keys);
        for (size_t index = 1; index < count; index++)
        {
            if (keys[index] != keys[unique_count - 1])
            {
                keys[unique_count++] = keys[index];
            }
        }
        count = unique_count;
    }
    if (count > 0 && capacity > count)
    {
        keys = repalloc(keys, sizeof(*keys) * count);
    }
    if (keys == NULL)
    {
        keys = palloc(1);
    }
    *keys_out = keys;
    *key_count_out = count;
    return complete;
}

void
evoke_filter_match_tid_candidates(
    Relation heap_relation,
    Jsonb *filters,
    const ItemPointerData *candidate_tids,
    size_t candidate_count,
    bool *matches_out,
    size_t *matched_count_out
)
{
    Oid parameter_types[2] = {JSONBOID, TIDARRAYOID};
    Datum parameter_values[2];
    char parameter_nulls[2] = {' ', ' '};
    Datum *candidate_datums;
    evoke_filter_tid_candidate_ref *candidate_refs;
    ArrayType *candidate_array;
    SPIPlanPtr plan = NULL;
    char *query = NULL;
    bool spi_connected = false;
    MemoryContext caller_context = CurrentMemoryContext;

    if (heap_relation == NULL || filters == NULL ||
        candidate_tids == NULL || candidate_count == 0 ||
        matches_out == NULL || matched_count_out == NULL ||
        candidate_count > (size_t) INT_MAX ||
        candidate_count > MaxAllocSize / sizeof(*candidate_datums) ||
        candidate_count > MaxAllocSize / sizeof(*candidate_refs))
    {
        ereport(ERROR, (errmsg("invalid evoke filter candidate probe")));
    }
    memset(matches_out, 0, candidate_count * sizeof(*matches_out));
    *matched_count_out = 0;
    candidate_datums = palloc(
        candidate_count * sizeof(*candidate_datums)
    );
    candidate_refs = palloc(
        candidate_count * sizeof(*candidate_refs)
    );
    for (size_t candidate = 0; candidate < candidate_count; candidate++)
    {
        if (!ItemPointerIsValid(&candidate_tids[candidate]))
        {
            ereport(ERROR, (errmsg("invalid evoke filter candidate TID")));
        }
        candidate_datums[candidate] =
            ItemPointerGetDatum(&candidate_tids[candidate]);
        candidate_refs[candidate].key =
            evoke_filter_tid_key(&candidate_tids[candidate]);
        candidate_refs[candidate].candidate_index = candidate;
    }
    qsort(
        candidate_refs,
        candidate_count,
        sizeof(*candidate_refs),
        evoke_filter_compare_tid_candidate_refs
    );
    candidate_array = construct_array(
        candidate_datums,
        (int) candidate_count,
        TIDOID,
        6,
        false,
        TYPALIGN_SHORT
    );
    parameter_values[0] = JsonbPGetDatum(filters);
    parameter_values[1] = PointerGetDatum(candidate_array);

    PG_TRY();
    {
        int status = SPI_connect();

        if (status != SPI_OK_CONNECT)
        {
            ereport(ERROR, (errmsg("SPI_connect failed: %d", status)));
        }
        spi_connected = true;
        {
            MemoryContext previous_context = MemoryContextSwitchTo(
                caller_context
            );

            query = evoke_filter_build_query(heap_relation, filters, true);
            MemoryContextSwitchTo(previous_context);
        }
        plan = SPI_prepare(query, 2, parameter_types);
        if (plan == NULL)
        {
            ereport(
                ERROR,
                (errmsg("could not prepare the evoke filter candidate probe"))
            );
        }
        status = SPI_execute_plan(
            plan,
            parameter_values,
            parameter_nulls,
            true,
            0
        );
        if (status != SPI_OK_SELECT)
        {
            ereport(
                ERROR,
                (errmsg("could not execute the evoke filter candidate probe"))
            );
        }
        for (uint64 row = 0; row < SPI_processed; row++)
        {
            bool is_null;
            Datum datum = SPI_getbinval(
                SPI_tuptable->vals[row],
                SPI_tuptable->tupdesc,
                1,
                &is_null
            );
            ItemPointer tid;
            uint64 key;
            size_t low = 0;
            size_t high = candidate_count;

            if (is_null)
            {
                ereport(ERROR, (errmsg("evoke filter probe returned null")));
            }
            tid = DatumGetItemPointer(datum);
            key = evoke_filter_tid_key(tid);
            while (low < high)
            {
                size_t middle = low + (high - low) / 2;

                if (candidate_refs[middle].key < key)
                {
                    low = middle + 1;
                }
                else
                {
                    high = middle;
                }
            }
            while (low < candidate_count &&
                   candidate_refs[low].key == key)
            {
                size_t candidate = candidate_refs[low].candidate_index;

                if (!matches_out[candidate])
                {
                    matches_out[candidate] = true;
                    (*matched_count_out)++;
                    break;
                }
                low++;
            }
        }
        if (SPI_tuptable != NULL)
        {
            SPI_freetuptable(SPI_tuptable);
        }
        SPI_freeplan(plan);
        plan = NULL;
        SPI_finish();
        spi_connected = false;
    }
    PG_CATCH();
    {
        if (plan != NULL)
        {
            SPI_freeplan(plan);
        }
        if (spi_connected)
        {
            SPI_finish();
        }
        if (query != NULL)
        {
            pfree(query);
        }
        pfree(candidate_array);
        pfree(candidate_datums);
        pfree(candidate_refs);
        PG_RE_THROW();
    }
    PG_END_TRY();

    if (query != NULL)
    {
        pfree(query);
    }
    pfree(candidate_array);
    pfree(candidate_datums);
    pfree(candidate_refs);
}

typedef struct evoke_filter_scope_reader
{
    Relation index_relation;
    evoke_segment_query_context context;
    evoke_segment_object_ref scope_ref;
    evoke_scope_header header;
    evoke_scope_column_entry *columns;
    char **column_names;
} evoke_filter_scope_reader;

static uint32
evoke_filter_scope_read_u32(const uint8 *bytes)
{
    return (uint32) bytes[0] |
        ((uint32) bytes[1] << 8) |
        ((uint32) bytes[2] << 16) |
        ((uint32) bytes[3] << 24);
}

static void
evoke_filter_scope_reader_init(evoke_filter_scope_reader *reader)
{
    memset(reader, 0, sizeof(*reader));
    evoke_segment_query_context_init(&reader->context);
}

static void
evoke_filter_scope_reader_free(evoke_filter_scope_reader *reader)
{
    if (reader == NULL)
    {
        return;
    }
    if (reader->column_names != NULL)
    {
        for (uint32 column = 0; column < reader->header.column_count; column++)
        {
            if (reader->column_names[column] != NULL)
            {
                pfree(reader->column_names[column]);
            }
        }
        pfree(reader->column_names);
    }
    if (reader->columns != NULL)
    {
        pfree(reader->columns);
    }
    evoke_segment_query_context_free(&reader->context);
    evoke_filter_scope_reader_init(reader);
}

static void
evoke_filter_scope_read(
    const evoke_filter_scope_reader *reader,
    uint64 offset,
    uint64 length,
    uint8 *bytes_out
)
{
    if (reader == NULL || offset > SIZE_MAX || length > SIZE_MAX ||
        offset > reader->header.total_size ||
        length > reader->header.total_size - offset)
    {
        ereport(ERROR, (errmsg("invalid evoke scope-posting range")));
    }
    evoke_segment_pages_read_semantic_accelerator_scope_range(
        reader->index_relation,
        &reader->context,
        &reader->scope_ref,
        (Size) offset,
        (Size) length,
        bytes_out
    );
}

static bool
evoke_filter_scope_reader_open(
    evoke_filter_scope_reader *reader,
    Relation index_relation,
    const evoke_segment_read_root *root
)
{
    uint8 *column_bytes = NULL;
    size_t columns_size;
    uint32 expected_column_count;

    if (index_relation->rd_index->indnatts ==
        index_relation->rd_index->indnkeyatts)
    {
        return false;
    }
    reader->index_relation = index_relation;
    evoke_segment_pages_load_query_context(
        index_relation,
        root,
        &reader->context
    );
    if (!evoke_segment_pages_open_semantic_accelerator_scope(
            index_relation,
            &reader->context,
            &reader->scope_ref,
            &reader->header))
    {
        return false;
    }
    expected_column_count = (uint32) (
        index_relation->rd_index->indnatts -
        index_relation->rd_index->indnkeyatts
    );
    if (reader->header.column_count != expected_column_count)
    {
        ereport(ERROR, (errmsg("evoke scope column contract changed")));
    }
    if (reader->header.column_count >
        MaxAllocSize / EVOKE_SCOPE_COLUMN_ENTRY_SIZE)
    {
        ereport(ERROR, (errmsg("evoke scope column metadata is too large")));
    }
    columns_size = (size_t) reader->header.column_count *
        EVOKE_SCOPE_COLUMN_ENTRY_SIZE;
    column_bytes = palloc(columns_size);
    reader->columns = palloc0(
        sizeof(*reader->columns) * reader->header.column_count
    );
    reader->column_names = palloc0(
        sizeof(*reader->column_names) * reader->header.column_count
    );
    evoke_filter_scope_read(
        reader,
        EVOKE_SCOPE_HEADER_SIZE,
        columns_size,
        column_bytes
    );
    for (uint32 column = 0; column < reader->header.column_count; column++)
    {
        evoke_status status = evoke_scope_column_entry_deserialize(
            column_bytes,
            columns_size,
            &reader->header,
            column,
            &reader->columns[column]
        );
        const evoke_scope_column_entry *entry = &reader->columns[column];
        uint32 expected_index_attribute =
            index_relation->rd_index->indnkeyatts + column + 1U;
        Form_pg_attribute attribute;

        if (status != EVOKE_OK || entry->name_size > MaxAllocSize - 1 ||
            entry->index_attribute != expected_index_attribute ||
            entry->heap_attribute <= 0)
        {
            ereport(ERROR, (errmsg("invalid evoke scope column metadata")));
        }
        attribute = TupleDescAttr(
            RelationGetDescr(index_relation),
            entry->index_attribute - 1U
        );
        if (attribute->attisdropped ||
            attribute->atttypid != entry->type_oid)
        {
            ereport(ERROR, (errmsg("evoke scope column contract changed")));
        }
        reader->column_names[column] = palloc(entry->name_size + 1U);
        evoke_filter_scope_read(
            reader,
            reader->header.dictionary_offset + entry->name_offset,
            entry->name_size,
            (uint8 *) reader->column_names[column]
        );
        reader->column_names[column][entry->name_size] = '\0';
    }
    pfree(column_bytes);
    return true;
}

static int
evoke_filter_scope_find_column(
    const evoke_filter_scope_reader *reader,
    const char *name
)
{
    for (uint32 column = 0; column < reader->header.column_count; column++)
    {
        if (strcmp(reader->column_names[column], name) == 0)
        {
            return (int) column;
        }
    }
    return -1;
}

static bool
evoke_filter_scope_type_supported(Oid type_oid)
{
    switch (getBaseType(type_oid))
    {
        case BOOLOID:
        case INT2OID:
        case INT4OID:
        case INT8OID:
        case OIDOID:
        case TEXTOID:
        case VARCHAROID:
        case BPCHAROID:
        case NAMEOID:
        case UUIDOID:
        case DATEOID:
        case TIMESTAMPOID:
        case TIMESTAMPTZOID:
        case NUMERICOID:
            return true;
        default:
            return false;
    }
}

static bool
evoke_filter_scope_canonical_value(
    const JsonbValue *value,
    Oid type_oid,
    int32 type_modifier,
    uint8 **bytes_out,
    uint32 *size_out
)
{
    Oid input_function;
    Oid input_parameter;
    Oid output_function;
    bool output_is_varlena;
    Datum typed_value;
    char *input = NULL;
    char *output;
    size_t output_size;

    *bytes_out = NULL;
    *size_out = 0;
    if (value == NULL || value->type == jbvNull ||
        value->type == jbvBinary ||
        !evoke_filter_scope_type_supported(type_oid))
    {
        return false;
    }
    if (value->type == jbvString)
    {
        input = pnstrdup(value->val.string.val, value->val.string.len);
    }
    else if (value->type == jbvNumeric)
    {
        Oid numeric_output;
        bool numeric_varlena;

        getTypeOutputInfo(NUMERICOID, &numeric_output, &numeric_varlena);
        input = OidOutputFunctionCall(
            numeric_output,
            NumericGetDatum(value->val.numeric)
        );
    }
    else if (value->type == jbvBool)
    {
        input = pstrdup(value->val.boolean ? "true" : "false");
    }
    else
    {
        return false;
    }
    getTypeInputInfo(type_oid, &input_function, &input_parameter);
    getTypeOutputInfo(type_oid, &output_function, &output_is_varlena);
    typed_value = OidInputFunctionCall(
        input_function,
        input,
        input_parameter,
        type_modifier
    );
    output = OidOutputFunctionCall(output_function, typed_value);
    pfree(input);
    output_size = strlen(output);
    if (output_size > UINT32_MAX ||
        output_size > EVOKE_FILTER_SCOPE_MAX_VALUE_BYTES)
    {
        pfree(output);
        return false;
    }
    *bytes_out = (uint8 *) output;
    *size_out = (uint32) output_size;
    return true;
}

static void
evoke_filter_scope_read_value_entry(
    const evoke_filter_scope_reader *reader,
    uint32 value_index,
    evoke_scope_value_entry *entry_out
)
{
    uint8 bytes[EVOKE_SCOPE_VALUE_ENTRY_SIZE];
    uint64 value_table = EVOKE_SCOPE_HEADER_SIZE +
        (uint64) reader->header.column_count *
            EVOKE_SCOPE_COLUMN_ENTRY_SIZE;
    evoke_status status;

    evoke_filter_scope_read(
        reader,
        value_table + (uint64) value_index * EVOKE_SCOPE_VALUE_ENTRY_SIZE,
        sizeof(bytes),
        bytes
    );
    status = evoke_scope_value_entry_deserialize_one(
        bytes,
        sizeof(bytes),
        &reader->header,
        value_index,
        entry_out
    );
    if (status != EVOKE_OK)
    {
        ereport(ERROR, (errmsg("invalid evoke scope value metadata")));
    }
}

static int
evoke_filter_scope_compare_value(
    const evoke_filter_scope_reader *reader,
    const evoke_scope_value_entry *entry,
    uint8 kind,
    const uint8 *target,
    uint32 target_size
)
{
    uint32 comparison_size;
    uint8 *value = NULL;
    int comparison = 0;

    if (entry->kind != kind)
    {
        return entry->kind < kind ? -1 : 1;
    }
    comparison_size = Min(entry->value_size, target_size);
    if (comparison_size > 0)
    {
        value = palloc(comparison_size);
        evoke_filter_scope_read(
            reader,
            reader->header.dictionary_offset + entry->value_offset,
            comparison_size,
            value
        );
        comparison = memcmp(value, target, comparison_size);
        pfree(value);
    }
    if (comparison != 0)
    {
        return comparison;
    }
    if (entry->value_size == target_size)
    {
        return 0;
    }
    return entry->value_size < target_size ? -1 : 1;
}

static bool
evoke_filter_scope_find_value(
    const evoke_filter_scope_reader *reader,
    uint32 column_index,
    uint8 kind,
    const uint8 *target,
    uint32 target_size,
    evoke_scope_value_entry *entry_out
)
{
    const evoke_scope_column_entry *column = &reader->columns[column_index];
    uint32 low = column->first_value;
    uint32 high = column->first_value + column->value_count;

    while (low < high)
    {
        uint32 middle = low + (high - low) / 2U;
        evoke_scope_value_entry entry;
        int comparison;

        evoke_filter_scope_read_value_entry(reader, middle, &entry);
        comparison = evoke_filter_scope_compare_value(
            reader,
            &entry,
            kind,
            target,
            target_size
        );
        if (comparison < 0)
        {
            low = middle + 1U;
        }
        else
        {
            high = middle;
        }
    }
    if (low >= column->first_value + column->value_count)
    {
        return false;
    }
    evoke_filter_scope_read_value_entry(reader, low, entry_out);
    return evoke_filter_scope_compare_value(
        reader,
        entry_out,
        kind,
        target,
        target_size
    ) == 0;
}

static void
evoke_filter_scope_add_posting_entries(
    const evoke_filter_scope_reader *reader,
    const evoke_scope_value_entry *entries,
    uint32 entry_count,
    uint8 *bitmap
)
{
    uint8 *buffer = palloc(
        EVOKE_FILTER_SCOPE_POSTING_READ_DOCUMENTS * sizeof(uint32)
    );
    uint64 total_documents = 0;
    uint64 completed = 0;
    uint64 expected_offset;
    uint32 entry_index = 0;
    uint32 entry_completed = 0;
    uint32 previous_document = 0;
    bool have_previous = false;

    if (entry_count == 0)
    {
        pfree(buffer);
        return;
    }
    expected_offset = entries[0].postings_offset;
    for (uint32 entry = 0; entry < entry_count; entry++)
    {
        uint64 posting_bytes =
            (uint64) entries[entry].document_count * sizeof(uint32);

        if (entries[entry].postings_offset != expected_offset ||
            total_documents > UINT64_MAX - entries[entry].document_count)
        {
            ereport(ERROR, (errmsg("invalid evoke scope posting range")));
        }
        expected_offset += posting_bytes;
        total_documents += entries[entry].document_count;
    }
    while (completed < total_documents)
    {
        uint32 count = Min(
            (uint64) EVOKE_FILTER_SCOPE_POSTING_READ_DOCUMENTS,
            total_documents - completed
        );

        evoke_filter_scope_read(
            reader,
            reader->header.postings_offset + entries[0].postings_offset +
                (uint64) completed * sizeof(uint32),
            (uint64) count * sizeof(uint32),
            buffer
        );
        for (uint32 item = 0; item < count; item++)
        {
            uint32 document = evoke_filter_scope_read_u32(
                buffer + (size_t) item * sizeof(uint32)
            );

            if (entry_completed == entries[entry_index].document_count)
            {
                entry_index++;
                entry_completed = 0;
                have_previous = false;
            }
            if (entry_index >= entry_count)
            {
                ereport(ERROR, (errmsg("invalid evoke scope posting range")));
            }
            if (document >= reader->header.document_count ||
                (have_previous && document <= previous_document))
            {
                ereport(ERROR, (errmsg("invalid evoke scope posting list")));
            }
            bitmap[document >> 3] |=
                (uint8) (UINT8_C(1) << (document & 7U));
            previous_document = document;
            have_previous = true;
            entry_completed++;
        }
        completed += count;
        CHECK_FOR_INTERRUPTS();
    }
    if (entry_index + 1U != entry_count ||
        entry_completed != entries[entry_index].document_count)
    {
        ereport(ERROR, (errmsg("invalid evoke scope posting range")));
    }
    pfree(buffer);
}

static void
evoke_filter_scope_add_postings(
    const evoke_filter_scope_reader *reader,
    const evoke_scope_value_entry *entry,
    uint8 *bitmap
)
{
    evoke_filter_scope_add_posting_entries(reader, entry, 1, bitmap);
}

typedef enum evoke_filter_scope_range_operation
{
    EVOKE_FILTER_SCOPE_RANGE_GT = 1,
    EVOKE_FILTER_SCOPE_RANGE_GTE = 2,
    EVOKE_FILTER_SCOPE_RANGE_LT = 3,
    EVOKE_FILTER_SCOPE_RANGE_LTE = 4
} evoke_filter_scope_range_operation;

typedef struct evoke_filter_scope_range_bound
{
    Datum value;
    evoke_filter_scope_range_operation operation;
} evoke_filter_scope_range_bound;

static Datum
evoke_filter_scope_input_canonical(
    const uint8 *bytes,
    uint32 size,
    Oid type_oid,
    int32 type_modifier
)
{
    Oid input_function;
    Oid input_parameter;
    char *input = pnstrdup((const char *) bytes, size);
    Datum result;

    getTypeInputInfo(type_oid, &input_function, &input_parameter);
    result = OidInputFunctionCall(
        input_function,
        input,
        input_parameter,
        type_modifier
    );
    pfree(input);
    return result;
}

static bool
evoke_filter_scope_range_matches(
    FmgrInfo *comparator,
    Oid collation_oid,
    Datum value,
    const evoke_filter_scope_range_bound *bounds,
    uint32 bound_count
)
{
    for (uint32 bound = 0; bound < bound_count; bound++)
    {
        int32 comparison = DatumGetInt32(FunctionCall2Coll(
            comparator,
            collation_oid,
            value,
            bounds[bound].value
        ));

        if ((bounds[bound].operation == EVOKE_FILTER_SCOPE_RANGE_GT &&
             comparison <= 0) ||
            (bounds[bound].operation == EVOKE_FILTER_SCOPE_RANGE_GTE &&
             comparison < 0) ||
            (bounds[bound].operation == EVOKE_FILTER_SCOPE_RANGE_LT &&
             comparison >= 0) ||
            (bounds[bound].operation == EVOKE_FILTER_SCOPE_RANGE_LTE &&
             comparison > 0))
        {
            return false;
        }
    }
    return true;
}

static bool
evoke_filter_scope_add_range(
    const evoke_filter_scope_reader *reader,
    Relation heap_relation,
    uint32 column_index,
    const JsonbValue *operand,
    uint8 *bitmap
)
{
    const evoke_scope_column_entry *scope_column =
        &reader->columns[column_index];
    Form_pg_attribute attribute;
    TypeCacheEntry *type_cache;
    evoke_filter_scope_range_bound bounds[4];
    evoke_scope_value_entry *entries = NULL;
    uint8 *metadata = NULL;
    uint8 *dictionary = NULL;
    uint8 *matched_entries = NULL;
    uint32 bound_count = 0;
    uint64 metadata_size;
    uint64 dictionary_start = 0;
    uint64 dictionary_end = 0;
    MemoryContext caller_context = CurrentMemoryContext;
    MemoryContext bounds_context = NULL;
    MemoryContext value_context = NULL;
    bool supported = false;

    if (scope_column->element_type_oid != InvalidOid ||
        operand == NULL || operand->type != jbvBinary ||
        !JsonContainerIsObject(operand->val.binary.data) ||
        scope_column->heap_attribute >
            RelationGetDescr(heap_relation)->natts)
    {
        return false;
    }
    attribute = TupleDescAttr(
        RelationGetDescr(heap_relation),
        scope_column->heap_attribute - 1
    );
    if (attribute->attisdropped ||
        attribute->atttypid != scope_column->type_oid ||
        attribute->attcollation != scope_column->collation_oid ||
        attribute->atttypmod != scope_column->type_modifier ||
        get_element_type(attribute->atttypid) != InvalidOid ||
        !evoke_filter_scope_type_supported(attribute->atttypid))
    {
        return false;
    }
    type_cache = lookup_type_cache(
        getBaseType(attribute->atttypid),
        TYPECACHE_CMP_PROC_FINFO
    );
    if (type_cache == NULL ||
        !OidIsValid(type_cache->cmp_proc_finfo.fn_oid))
    {
        return false;
    }
    bounds_context = AllocSetContextCreate(
        caller_context,
        "evoke scope range bounds",
        ALLOCSET_SMALL_SIZES
    );
    MemoryContextSwitchTo(bounds_context);
    {
        JsonbIterator *iterator = JsonbIteratorInit(
            operand->val.binary.data
        );
        JsonbValue item;
        JsonbIteratorToken token = JsonbIteratorNext(
            &iterator,
            &item,
            true
        );

        if (token != WJB_BEGIN_OBJECT)
        {
            MemoryContextSwitchTo(caller_context);
            goto cleanup;
        }
        while ((token = JsonbIteratorNext(&iterator, &item, true)) !=
               WJB_END_OBJECT)
        {
            char *bound_name;
            uint8 *canonical = NULL;
            uint32 canonical_size = 0;

            if (token != WJB_KEY || bound_count >= lengthof(bounds))
            {
                MemoryContextSwitchTo(caller_context);
                goto cleanup;
            }
            bound_name = evoke_filter_json_key(&item);
            if (evoke_filter_string_equals(bound_name, "gt"))
            {
                bounds[bound_count].operation =
                    EVOKE_FILTER_SCOPE_RANGE_GT;
            }
            else if (evoke_filter_string_equals(bound_name, "gte"))
            {
                bounds[bound_count].operation =
                    EVOKE_FILTER_SCOPE_RANGE_GTE;
            }
            else if (evoke_filter_string_equals(bound_name, "lt"))
            {
                bounds[bound_count].operation =
                    EVOKE_FILTER_SCOPE_RANGE_LT;
            }
            else if (evoke_filter_string_equals(bound_name, "lte"))
            {
                bounds[bound_count].operation =
                    EVOKE_FILTER_SCOPE_RANGE_LTE;
            }
            else
            {
                pfree(bound_name);
                MemoryContextSwitchTo(caller_context);
                goto cleanup;
            }
            pfree(bound_name);
            token = JsonbIteratorNext(&iterator, &item, true);
            if (token != WJB_VALUE ||
                !evoke_filter_scope_canonical_value(
                    &item,
                    attribute->atttypid,
                    attribute->atttypmod,
                    &canonical,
                    &canonical_size))
            {
                MemoryContextSwitchTo(caller_context);
                goto cleanup;
            }
            bounds[bound_count].value = evoke_filter_scope_input_canonical(
                canonical,
                canonical_size,
                attribute->atttypid,
                attribute->atttypmod
            );
            pfree(canonical);
            bound_count++;
        }
    }
    MemoryContextSwitchTo(caller_context);
    if (bound_count == 0)
    {
        goto cleanup;
    }
    if (scope_column->value_count == 0)
    {
        supported = true;
        goto cleanup;
    }
    if (scope_column->value_count > EVOKE_FILTER_SCOPE_RANGE_MAX_VALUES)
    {
        goto cleanup;
    }
    metadata_size = (uint64) scope_column->value_count *
        EVOKE_SCOPE_VALUE_ENTRY_SIZE;
    if (metadata_size > MaxAllocSize)
    {
        goto cleanup;
    }
    metadata = palloc((Size) metadata_size);
    entries = palloc(
        sizeof(*entries) * (Size) scope_column->value_count
    );
    evoke_filter_scope_read(
        reader,
        EVOKE_SCOPE_HEADER_SIZE +
            (uint64) reader->header.column_count *
                EVOKE_SCOPE_COLUMN_ENTRY_SIZE +
            (uint64) scope_column->first_value *
                EVOKE_SCOPE_VALUE_ENTRY_SIZE,
        metadata_size,
        metadata
    );
    for (uint32 value = 0; value < scope_column->value_count; value++)
    {
        evoke_status status = evoke_scope_value_entry_deserialize_one(
            metadata + (Size) value * EVOKE_SCOPE_VALUE_ENTRY_SIZE,
            EVOKE_SCOPE_VALUE_ENTRY_SIZE,
            &reader->header,
            scope_column->first_value + value,
            &entries[value]
        );

        if (status != EVOKE_OK ||
            entries[value].column_index != column_index ||
            entries[value].kind != EVOKE_SCOPE_VALUE_SCALAR)
        {
            ereport(ERROR, (errmsg("invalid evoke scope range metadata")));
        }
        if (value == 0)
        {
            dictionary_start = entries[value].value_offset;
            dictionary_end = dictionary_start;
        }
        if (entries[value].value_offset != dictionary_end ||
            entries[value].value_size >
                reader->header.dictionary_size - dictionary_end)
        {
            ereport(ERROR, (errmsg("invalid evoke scope range dictionary")));
        }
        dictionary_end += entries[value].value_size;
    }
    if (dictionary_end < dictionary_start ||
        dictionary_end - dictionary_start >
            EVOKE_FILTER_SCOPE_RANGE_MAX_DICTIONARY_BYTES ||
        dictionary_end - dictionary_start > MaxAllocSize)
    {
        goto cleanup;
    }
    dictionary = palloc(Max(
        (Size) (dictionary_end - dictionary_start),
        (Size) 1
    ));
    matched_entries = palloc0((Size) scope_column->value_count);
    if (dictionary_end > dictionary_start)
    {
        evoke_filter_scope_read(
            reader,
            reader->header.dictionary_offset + dictionary_start,
            dictionary_end - dictionary_start,
            dictionary
        );
    }
    value_context = AllocSetContextCreate(
        caller_context,
        "evoke scope range value",
        ALLOCSET_SMALL_SIZES
    );
    for (uint32 value = 0; value < scope_column->value_count; value++)
    {
        MemoryContext previous_context;
        Datum typed_value;
        bool matches;

        MemoryContextReset(value_context);
        previous_context = MemoryContextSwitchTo(value_context);
        typed_value = evoke_filter_scope_input_canonical(
            dictionary + entries[value].value_offset - dictionary_start,
            entries[value].value_size,
            attribute->atttypid,
            attribute->atttypmod
        );
        matches = evoke_filter_scope_range_matches(
            &type_cache->cmp_proc_finfo,
            attribute->attcollation,
            typed_value,
            bounds,
            bound_count
        );
        MemoryContextSwitchTo(previous_context);
        if (matches)
        {
            matched_entries[value] = 1;
        }
        if ((value & UINT32_C(4095)) == 0)
        {
            CHECK_FOR_INTERRUPTS();
        }
    }
    for (uint32 value = 0; value < scope_column->value_count;)
    {
        uint32 first;

        if (matched_entries[value] == 0)
        {
            value++;
            continue;
        }
        first = value++;
        while (value < scope_column->value_count &&
               matched_entries[value] != 0)
        {
            value++;
        }
        evoke_filter_scope_add_posting_entries(
            reader,
            &entries[first],
            value - first,
            bitmap
        );
    }
    supported = true;

cleanup:
    MemoryContextSwitchTo(caller_context);
    if (value_context != NULL)
    {
        MemoryContextDelete(value_context);
    }
    if (bounds_context != NULL)
    {
        MemoryContextDelete(bounds_context);
    }
    if (dictionary != NULL)
    {
        pfree(dictionary);
    }
    if (matched_entries != NULL)
    {
        pfree(matched_entries);
    }
    if (entries != NULL)
    {
        pfree(entries);
    }
    if (metadata != NULL)
    {
        pfree(metadata);
    }
    return supported;
}

static bool
evoke_filter_scope_add_operand(
    const evoke_filter_scope_reader *reader,
    Relation heap_relation,
    uint32 column_index,
    uint8 kind,
    const JsonbValue *operand,
    uint8 *bitmap
)
{
    const evoke_scope_column_entry *scope_column =
        &reader->columns[column_index];
    Form_pg_attribute attribute;
    Oid type_oid = kind == EVOKE_SCOPE_VALUE_ARRAY_ELEMENT
        ? scope_column->element_type_oid
        : scope_column->type_oid;
    int32 type_modifier;
    uint8 *target = NULL;
    uint32 target_size = 0;
    evoke_scope_value_entry entry;
    bool supported;

    if (scope_column->heap_attribute >
        RelationGetDescr(heap_relation)->natts)
    {
        ereport(ERROR, (errmsg("evoke scope column contract changed")));
    }
    attribute = TupleDescAttr(
        RelationGetDescr(heap_relation),
        scope_column->heap_attribute - 1
    );
    type_modifier = kind == EVOKE_SCOPE_VALUE_ARRAY_ELEMENT
        ? -1
        : attribute->atttypmod;
    if (attribute->attisdropped ||
        attribute->atttypid != scope_column->type_oid ||
        attribute->attcollation != scope_column->collation_oid ||
        attribute->atttypmod != scope_column->type_modifier ||
        get_element_type(attribute->atttypid) !=
            scope_column->element_type_oid)
    {
        ereport(
            ERROR,
            (
                errmsg("evoke scope column contract changed"),
                errdetail(
                    "Heap attribute %d has type %u, collation %u, typmod %d, "
                    "and element type %u; scope metadata has type %u, "
                    "collation %u, typmod %d, and element type %u.",
                    scope_column->heap_attribute,
                    attribute->atttypid,
                    attribute->attcollation,
                    attribute->atttypmod,
                    get_element_type(attribute->atttypid),
                    scope_column->type_oid,
                    scope_column->collation_oid,
                    scope_column->type_modifier,
                    scope_column->element_type_oid
                )
            )
        );
    }
    supported = evoke_filter_scope_canonical_value(
        operand,
        type_oid,
        type_modifier,
        &target,
        &target_size
    );
    if (!supported)
    {
        return false;
    }
    if (evoke_filter_scope_find_value(
            reader,
            column_index,
            kind,
            target,
            target_size,
            &entry))
    {
        evoke_filter_scope_add_postings(reader, &entry, bitmap);
    }
    pfree(target);
    return true;
}

typedef struct evoke_filter_scope_pattern_grams
{
    uint16 bits[EVOKE_FILTER_SCOPE_PATTERN_GRAMS];
    uint32 value_mask;
    uint8 count;
} evoke_filter_scope_pattern_grams;

typedef struct evoke_filter_scope_gram_reader
{
    const evoke_filter_scope_reader *scope;
    uint8 *cache;
    uint32 cache_first_block;
    uint32 cache_block_count;
} evoke_filter_scope_gram_reader;

typedef struct evoke_filter_scope_simple_pattern
{
    const uint8 *literal;
    Size literal_size;
    bool valid;
} evoke_filter_scope_simple_pattern;

static bool
evoke_filter_scope_ascii_casefold_safe(Oid collation)
{
    static const char uppercase[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    static const char lowercase[] = "abcdefghijklmnopqrstuvwxyz";
    text *input;
    text *folded;
    bool safe;

    if (!OidIsValid(collation))
    {
        return false;
    }
    input = cstring_to_text_with_len(uppercase, sizeof(uppercase) - 1U);
    folded = DatumGetTextPP(DirectFunctionCall1Coll(
        lower,
        collation,
        PointerGetDatum(input)
    ));
    safe = VARSIZE_ANY_EXHDR(folded) == sizeof(lowercase) - 1U &&
        memcmp(
            VARDATA_ANY(folded),
            lowercase,
            sizeof(lowercase) - 1U
        ) == 0;
    if ((Pointer) folded != (Pointer) input)
    {
        pfree(folded);
    }
    pfree(input);
    return safe;
}

static bool
evoke_filter_scope_pattern_grams_build(
    const text *pattern,
    evoke_filter_scope_pattern_grams *grams_out
)
{
    const uint8 *bytes = (const uint8 *) VARDATA_ANY(pattern);
    Size size = VARSIZE_ANY_EXHDR(pattern);
    uint8 previous[2] = {0, 0};
    uint32 literal_size = 0;

    memset(grams_out, 0, sizeof(*grams_out));
    for (Size offset = 0; offset < size; offset++)
    {
        uint8 value = bytes[offset];

        if (value == (uint8) '\\')
        {
            if (++offset >= size)
            {
                literal_size = 0;
                break;
            }
            value = bytes[offset];
        }
        else if (value == (uint8) '%' || value == (uint8) '_')
        {
            literal_size = 0;
            continue;
        }
        if (value >= UINT8_C(0x80))
        {
            literal_size = 0;
            continue;
        }
        if (literal_size >= 2)
        {
            uint16 bit = (uint16) evoke_scope_ascii_gram_bit(
                previous[0],
                previous[1],
                value
            );
            bool duplicate = false;

            for (uint8 index = 0; index < grams_out->count; index++)
            {
                if (grams_out->bits[index] == bit)
                {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate &&
                grams_out->count < EVOKE_FILTER_SCOPE_PATTERN_GRAMS)
            {
                grams_out->bits[grams_out->count++] = bit;
                grams_out->value_mask |=
                    UINT32_C(1) << (bit % 32U);
            }
        }
        previous[0] = literal_size == 0 ? value : previous[1];
        previous[1] = value;
        literal_size++;
    }
    return grams_out->count > 0;
}

static bool
evoke_filter_scope_value_gram_matches(
    const evoke_scope_value_entry *entry,
    const evoke_filter_scope_pattern_grams *patterns,
    uint32 pattern_count
)
{
    for (uint32 pattern = 0; pattern < pattern_count; pattern++)
    {
        if ((entry->gram_mask & patterns[pattern].value_mask) ==
            patterns[pattern].value_mask)
        {
            return true;
        }
    }
    return false;
}

static void
evoke_filter_scope_gram_reader_free(evoke_filter_scope_gram_reader *reader)
{
    if (reader->cache != NULL)
    {
        pfree(reader->cache);
    }
    memset(reader, 0, sizeof(*reader));
}

static const uint8 *
evoke_filter_scope_gram_reader_block(
    evoke_filter_scope_gram_reader *reader,
    uint32 block
)
{
    uint64 total_blocks = reader->scope->header.value_count == 0
        ? 0
        : ((uint64) reader->scope->header.value_count +
            EVOKE_SCOPE_GRAM_BLOCK_VALUES - 1U) /
            EVOKE_SCOPE_GRAM_BLOCK_VALUES;

    if (block >= total_blocks)
    {
        ereport(ERROR, (errmsg("invalid evoke scope gram-filter block")));
    }
    if (reader->cache == NULL || block < reader->cache_first_block ||
        block >= reader->cache_first_block + reader->cache_block_count)
    {
        uint32 first =
            block - block % EVOKE_FILTER_SCOPE_GRAM_CACHE_BLOCKS;
        uint32 count = (uint32) Min(
            (uint64) EVOKE_FILTER_SCOPE_GRAM_CACHE_BLOCKS,
            total_blocks - first
        );

        if (reader->cache == NULL)
        {
            reader->cache = palloc(
                (Size) EVOKE_FILTER_SCOPE_GRAM_CACHE_BLOCKS *
                    EVOKE_SCOPE_GRAM_FILTER_BYTES
            );
        }
        evoke_filter_scope_read(
            reader->scope,
            reader->scope->header.gram_filter_offset +
                (uint64) first * EVOKE_SCOPE_GRAM_FILTER_BYTES,
            (uint64) count * EVOKE_SCOPE_GRAM_FILTER_BYTES,
            reader->cache
        );
        reader->cache_first_block = first;
        reader->cache_block_count = count;
    }
    return reader->cache +
        (Size) (block - reader->cache_first_block) *
            EVOKE_SCOPE_GRAM_FILTER_BYTES;
}

static bool
evoke_filter_scope_gram_block_matches(
    evoke_filter_scope_gram_reader *reader,
    uint32 block,
    const evoke_filter_scope_pattern_grams *patterns,
    uint32 pattern_count
)
{
    const uint8 *filter = evoke_filter_scope_gram_reader_block(reader, block);

    for (uint32 pattern = 0; pattern < pattern_count; pattern++)
    {
        bool matches = true;

        for (uint8 gram = 0; gram < patterns[pattern].count; gram++)
        {
            uint16 bit = patterns[pattern].bits[gram];

            if ((filter[bit / 8U] &
                 (uint8) (UINT8_C(1) << (bit % 8U))) == 0)
            {
                matches = false;
                break;
            }
        }
        if (matches)
        {
            return true;
        }
    }
    return false;
}

static bool
evoke_filter_scope_add_ilike(
    const evoke_filter_scope_reader *reader,
    Relation heap_relation,
    uint32 column_index,
    const JsonbValue *operand,
    bool multiple_patterns,
    uint8 *bitmap,
    evoke_filter_scope_trace *trace
)
{
    const evoke_scope_column_entry *scope_column =
        &reader->columns[column_index];
    Form_pg_attribute attribute;
    evoke_scope_value_entry *entries = NULL;
    text **patterns = NULL;
    uint8 *metadata = NULL;
    uint8 *dictionary = NULL;
    uint8 *matched_entries = NULL;
    text *candidate = NULL;
    Size candidate_capacity = 0;
    evoke_filter_scope_pattern_grams *pattern_grams = NULL;
    evoke_filter_scope_simple_pattern *simple_patterns = NULL;
    evoke_filter_scope_gram_reader gram_reader;
    uint64 candidate_values;
    uint64 comparison_candidates = 0;
    uint32 pattern_count = 0;
    Oid comparison_type;
    char type_category;
    bool type_preferred;
    uint8 kind;
    MemoryContext caller_context = CurrentMemoryContext;
    MemoryContext value_context = NULL;
    bool ascii_contains_safe = false;
    bool gram_pruning = false;
    bool value_gram_pruning = false;
    bool supported = false;

    memset(&gram_reader, 0, sizeof(gram_reader));
    gram_reader.scope = reader;

    if (scope_column->heap_attribute >
        RelationGetDescr(heap_relation)->natts)
    {
        return false;
    }
    attribute = TupleDescAttr(
        RelationGetDescr(heap_relation),
        scope_column->heap_attribute - 1
    );
    if (attribute->attisdropped ||
        attribute->atttypid != scope_column->type_oid ||
        attribute->attcollation != scope_column->collation_oid ||
        attribute->atttypmod != scope_column->type_modifier ||
        get_element_type(attribute->atttypid) !=
            scope_column->element_type_oid)
    {
        return false;
    }
    comparison_type = scope_column->element_type_oid == InvalidOid
        ? scope_column->type_oid
        : scope_column->element_type_oid;
    get_type_category_preferred(
        getBaseType(comparison_type),
        &type_category,
        &type_preferred
    );
    (void) type_preferred;
    if (type_category != TYPCATEGORY_STRING)
    {
        return false;
    }
    kind = scope_column->element_type_oid == InvalidOid
        ? EVOKE_SCOPE_VALUE_SCALAR
        : EVOKE_SCOPE_VALUE_ARRAY_ELEMENT;
    if (!multiple_patterns)
    {
        if (operand == NULL || operand->type != jbvString)
        {
            return false;
        }
        pattern_count = 1;
        patterns = palloc(sizeof(*patterns));
        patterns[0] = cstring_to_text_with_len(
            operand->val.string.val,
            operand->val.string.len
        );
    }
    else
    {
        JsonbIterator *iterator;
        JsonbValue item;
        JsonbIteratorToken token;
        uint32 pattern_index = 0;

        if (operand == NULL || operand->type != jbvBinary ||
            !JsonContainerIsArray(operand->val.binary.data) ||
            JsonContainerSize(operand->val.binary.data) >
                EVOKE_FILTER_MAX_VALUES)
        {
            return false;
        }
        pattern_count = JsonContainerSize(operand->val.binary.data);
        patterns = palloc0(
            Max((Size) pattern_count, (Size) 1) * sizeof(*patterns)
        );
        iterator = JsonbIteratorInit(operand->val.binary.data);
        token = JsonbIteratorNext(&iterator, &item, true);
        if (token != WJB_BEGIN_ARRAY)
        {
            goto cleanup;
        }
        while ((token = JsonbIteratorNext(&iterator, &item, true)) !=
               WJB_END_ARRAY)
        {
            if (token != WJB_ELEM || item.type != jbvString ||
                pattern_index >= pattern_count)
            {
                goto cleanup;
            }
            patterns[pattern_index++] = cstring_to_text_with_len(
                item.val.string.val,
                item.val.string.len
            );
        }
        if (pattern_index != pattern_count)
        {
            goto cleanup;
        }
    }
    if (pattern_count == 0 || scope_column->value_count == 0)
    {
        supported = true;
        goto cleanup;
    }
    pattern_grams = palloc0(
        (Size) pattern_count * sizeof(*pattern_grams)
    );
    if (OidIsValid(attribute->attcollation))
    {
        pg_locale_t locale = pg_newlocale_from_collation(
            attribute->attcollation
        );

        ascii_contains_safe = locale != NULL && locale->deterministic &&
            evoke_filter_scope_ascii_casefold_safe(
                attribute->attcollation
            );
        gram_pruning = evoke_scope_header_is_current(&reader->header) &&
            ascii_contains_safe;
    }
    simple_patterns = palloc0(
        (Size) pattern_count * sizeof(*simple_patterns)
    );
    for (uint32 pattern = 0;
         ascii_contains_safe && pattern < pattern_count;
         pattern++)
    {
        simple_patterns[pattern].valid =
            evoke_scope_ascii_ilike_contains_pattern(
                (const uint8 *) VARDATA_ANY(patterns[pattern]),
                VARSIZE_ANY_EXHDR(patterns[pattern]),
                &simple_patterns[pattern].literal,
                &simple_patterns[pattern].literal_size
            );
    }
    for (uint32 pattern = 0; gram_pruning && pattern < pattern_count; pattern++)
    {
        gram_pruning = evoke_filter_scope_pattern_grams_build(
            patterns[pattern],
            &pattern_grams[pattern]
        );
    }
    value_gram_pruning = gram_pruning &&
        evoke_scope_header_is_current(&reader->header);
    candidate_values = scope_column->value_count;
    if (gram_pruning)
    {
        uint64 first = scope_column->first_value;
        uint64 end = first + scope_column->value_count;

        candidate_values = 0;
        while (first < end)
        {
            uint32 block = (uint32) (
                first / EVOKE_SCOPE_GRAM_BLOCK_VALUES
            );
            uint64 block_end = Min(
                end,
                ((uint64) block + 1U) * EVOKE_SCOPE_GRAM_BLOCK_VALUES
            );
            bool block_matches = evoke_filter_scope_gram_block_matches(
                &gram_reader,
                block,
                pattern_grams,
                pattern_count
            );

            if (trace != NULL)
            {
                trace->ilike_blocks_considered++;
                if (!block_matches)
                {
                    trace->ilike_blocks_skipped++;
                }
            }
            if (block_matches)
            {
                candidate_values += block_end - first;
            }
            first = block_end;
        }
    }
    if (trace != NULL)
    {
        trace->ilike_values = trace->ilike_values >
                UINT32_MAX - scope_column->value_count
            ? UINT32_MAX
            : trace->ilike_values + scope_column->value_count;
    }
    if (!value_gram_pruning && candidate_values >
        EVOKE_FILTER_SCOPE_ILIKE_MAX_COMPARISONS / pattern_count)
    {
        if (trace != NULL)
        {
            trace->ilike_comparison_limit_exceeded = true;
        }
        goto cleanup;
    }
    metadata = palloc(
        (Size) EVOKE_FILTER_SCOPE_ILIKE_VALUE_WINDOW *
            EVOKE_SCOPE_VALUE_ENTRY_SIZE
    );
    entries = palloc(
        (Size) EVOKE_FILTER_SCOPE_ILIKE_VALUE_WINDOW * sizeof(*entries)
    );
    matched_entries = palloc(
        (Size) EVOKE_FILTER_SCOPE_ILIKE_VALUE_WINDOW
    );
    value_context = AllocSetContextCreate(
        caller_context,
        "evoke scope ILIKE value",
        ALLOCSET_SMALL_SIZES
    );
    for (uint32 first_value = 0;
         first_value < scope_column->value_count;)
    {
        uint32 window_count;
        uint64 dictionary_start;
        uint64 dictionary_end;

        if (gram_pruning)
        {
            uint64 global_value =
                scope_column->first_value + first_value;
            uint32 block = (uint32) (
                global_value / EVOKE_SCOPE_GRAM_BLOCK_VALUES
            );
            uint64 block_end = Min(
                (uint64) scope_column->first_value +
                    scope_column->value_count,
                ((uint64) block + 1U) * EVOKE_SCOPE_GRAM_BLOCK_VALUES
            );

            if (!evoke_filter_scope_gram_block_matches(
                    &gram_reader,
                    block,
                    pattern_grams,
                    pattern_count))
            {
                first_value = (uint32) (
                    block_end - scope_column->first_value
                );
                continue;
            }
            window_count = (uint32) (block_end - global_value);
        }
        else
        {
            window_count = Min(
                EVOKE_FILTER_SCOPE_ILIKE_VALUE_WINDOW,
                scope_column->value_count - first_value
            );
        }

        for (;;)
        {
            uint64 metadata_size =
                (uint64) window_count * EVOKE_SCOPE_VALUE_ENTRY_SIZE;

            memset(matched_entries, 0, (Size) window_count);
            evoke_filter_scope_read(
                reader,
                EVOKE_SCOPE_HEADER_SIZE +
                    (uint64) reader->header.column_count *
                        EVOKE_SCOPE_COLUMN_ENTRY_SIZE +
                    (uint64) (scope_column->first_value + first_value) *
                        EVOKE_SCOPE_VALUE_ENTRY_SIZE,
                metadata_size,
                metadata
            );
            dictionary_start = 0;
            dictionary_end = 0;
            for (uint32 value = 0; value < window_count; value++)
            {
                evoke_status status =
                    evoke_scope_value_entry_deserialize_one(
                        metadata +
                            (Size) value * EVOKE_SCOPE_VALUE_ENTRY_SIZE,
                        EVOKE_SCOPE_VALUE_ENTRY_SIZE,
                        &reader->header,
                        scope_column->first_value + first_value + value,
                        &entries[value]
                    );

                if (status != EVOKE_OK ||
                    entries[value].column_index != column_index ||
                    entries[value].kind != kind)
                {
                    ereport(
                        ERROR,
                        (errmsg("invalid evoke scope ILIKE metadata"))
                    );
                }
                if (value == 0)
                {
                    dictionary_start = entries[value].value_offset;
                    dictionary_end = dictionary_start;
                }
                if (entries[value].value_offset != dictionary_end ||
                    entries[value].value_size >
                        reader->header.dictionary_size - dictionary_end)
                {
                    ereport(
                        ERROR,
                        (errmsg("invalid evoke scope ILIKE dictionary"))
                    );
                }
                dictionary_end += entries[value].value_size;
                if (!value_gram_pruning ||
                    evoke_filter_scope_value_gram_matches(
                        &entries[value],
                        pattern_grams,
                        pattern_count
                    ))
                {
                    matched_entries[value] = 1;
                }
            }
            if (dictionary_end < dictionary_start)
            {
                ereport(
                    ERROR,
                    (errmsg("invalid evoke scope ILIKE dictionary"))
                );
            }
            if (dictionary_end - dictionary_start <=
                    EVOKE_FILTER_SCOPE_ILIKE_DICTIONARY_WINDOW_BYTES ||
                window_count == 1)
            {
                break;
            }
            window_count = Max(window_count / 2U, 1U);
        }
        {
            uint64 window_candidates = 0;

            for (uint32 value = 0; value < window_count; value++)
            {
                window_candidates += matched_entries[value] != 0;
            }
            if (trace != NULL)
            {
                trace->ilike_values_examined =
                    trace->ilike_values_examined >
                        UINT64_MAX - window_candidates
                        ? UINT64_MAX
                        : trace->ilike_values_examined + window_candidates;
            }
            if (window_candidates >
                EVOKE_FILTER_SCOPE_ILIKE_MAX_COMPARISONS / pattern_count -
                    comparison_candidates)
            {
                if (trace != NULL)
                {
                    trace->ilike_comparison_limit_exceeded = true;
                }
                goto cleanup;
            }
            comparison_candidates += window_candidates;
            if (window_candidates == 0)
            {
                first_value += window_count;
                CHECK_FOR_INTERRUPTS();
                continue;
            }
        }
        dictionary = palloc(Max(
            (Size) (dictionary_end - dictionary_start),
            (Size) 1
        ));
        if (dictionary_end > dictionary_start)
        {
            evoke_filter_scope_read(
                reader,
                reader->header.dictionary_offset + dictionary_start,
                dictionary_end - dictionary_start,
                dictionary
            );
        }
        for (uint32 value = 0; value < window_count; value++)
        {
            const uint8 *candidate_bytes = dictionary +
                entries[value].value_offset - dictionary_start;
            MemoryContext previous_context = NULL;
            bool context_switched = false;

            if (matched_entries[value] == 0)
            {
                continue;
            }
            matched_entries[value] = 0;

            for (uint32 pattern = 0; pattern < pattern_count; pattern++)
            {
                bool matches = false;
                bool handled = simple_patterns[pattern].valid &&
                    evoke_scope_ascii_case_insensitive_contains(
                        candidate_bytes,
                        entries[value].value_size,
                        simple_patterns[pattern].literal,
                        simple_patterns[pattern].literal_size,
                        &matches
                    );

                if (trace != NULL)
                {
                    trace->ilike_comparisons =
                        trace->ilike_comparisons == UINT64_MAX
                            ? UINT64_MAX
                            : trace->ilike_comparisons + 1U;
                }
                if (!handled)
                {
                    Size candidate_size;

                    if ((Size) entries[value].value_size >
                        MaxAllocSize - VARHDRSZ)
                    {
                        ereport(
                            ERROR,
                            (errmsg("invalid evoke scope ILIKE value"))
                        );
                    }
                    candidate_size = VARHDRSZ +
                        (Size) entries[value].value_size;
                    if (candidate_size > candidate_capacity)
                    {
                        candidate = candidate == NULL
                            ? palloc(candidate_size)
                            : repalloc(candidate, candidate_size);
                        candidate_capacity = candidate_size;
                    }
                    SET_VARSIZE(candidate, candidate_size);
                    memcpy(
                        VARDATA(candidate),
                        candidate_bytes,
                        entries[value].value_size
                    );
                    if (!context_switched)
                    {
                        MemoryContextReset(value_context);
                        previous_context = MemoryContextSwitchTo(
                            value_context
                        );
                        context_switched = true;
                    }
                    matches = DatumGetBool(DirectFunctionCall2Coll(
                        texticlike,
                        attribute->attcollation,
                        PointerGetDatum(candidate),
                        PointerGetDatum(patterns[pattern])
                    ));
                }
                if (matches)
                {
                    matched_entries[value] = 1;
                    break;
                }
            }
            if (context_switched)
            {
                MemoryContextSwitchTo(previous_context);
            }
            if (((first_value + value) & UINT32_C(4095)) == 0)
            {
                CHECK_FOR_INTERRUPTS();
            }
        }
        for (uint32 value = 0; value < window_count;)
        {
            uint32 first_match;

            if (matched_entries[value] == 0)
            {
                value++;
                continue;
            }
            first_match = value++;
            while (value < window_count && matched_entries[value] != 0)
            {
                value++;
            }
            evoke_filter_scope_add_posting_entries(
                reader,
                &entries[first_match],
                value - first_match,
                bitmap
            );
        }
        pfree(dictionary);
        dictionary = NULL;
        first_value += window_count;
        CHECK_FOR_INTERRUPTS();
    }
    supported = true;

cleanup:
    MemoryContextSwitchTo(caller_context);
    evoke_filter_scope_gram_reader_free(&gram_reader);
    if (value_context != NULL)
    {
        MemoryContextDelete(value_context);
    }
    if (patterns != NULL)
    {
        for (uint32 pattern = 0; pattern < pattern_count; pattern++)
        {
            if (patterns[pattern] != NULL)
            {
                pfree(patterns[pattern]);
            }
        }
        pfree(patterns);
    }
    if (matched_entries != NULL)
    {
        pfree(matched_entries);
    }
    if (candidate != NULL)
    {
        pfree(candidate);
    }
    if (dictionary != NULL)
    {
        pfree(dictionary);
    }
    if (entries != NULL)
    {
        pfree(entries);
    }
    if (metadata != NULL)
    {
        pfree(metadata);
    }
    if (pattern_grams != NULL)
    {
        pfree(pattern_grams);
    }
    if (simple_patterns != NULL)
    {
        pfree(simple_patterns);
    }
    return supported;
}

static bool
evoke_filter_scope_build_predicate(
    const evoke_filter_scope_reader *reader,
    Relation heap_relation,
    uint32 column_index,
    const char *operation,
    const JsonbValue *operand,
    uint8 *bitmap,
    evoke_filter_scope_trace *trace
)
{
    const evoke_scope_column_entry *column = &reader->columns[column_index];
    uint8 kind;

    if (evoke_filter_string_equals(operation, "ilike") ||
        evoke_filter_string_equals(operation, "ilike_any"))
    {
        return evoke_filter_scope_add_ilike(
            reader,
            heap_relation,
            column_index,
            operand,
            evoke_filter_string_equals(operation, "ilike_any"),
            bitmap,
            trace
        );
    }

    if (evoke_filter_string_equals(operation, "eq"))
    {
        if (column->element_type_oid != InvalidOid ||
            operand->type == jbvNull)
        {
            return false;
        }
        return evoke_filter_scope_add_operand(
            reader,
            heap_relation,
            column_index,
            EVOKE_SCOPE_VALUE_SCALAR,
            operand,
            bitmap
        );
    }
    if (evoke_filter_string_equals(operation, "range"))
    {
        return evoke_filter_scope_add_range(
            reader,
            heap_relation,
            column_index,
            operand,
            bitmap
        );
    }
    if (evoke_filter_string_equals(operation, "in"))
    {
        if (column->element_type_oid != InvalidOid)
        {
            return false;
        }
        kind = EVOKE_SCOPE_VALUE_SCALAR;
    }
    else if (evoke_filter_string_equals(operation, "overlap"))
    {
        if (column->element_type_oid == InvalidOid)
        {
            return false;
        }
        kind = EVOKE_SCOPE_VALUE_ARRAY_ELEMENT;
    }
    else
    {
        return false;
    }
    if (operand->type != jbvBinary ||
        !JsonContainerIsArray(operand->val.binary.data) ||
        JsonContainerSize(operand->val.binary.data) > EVOKE_FILTER_MAX_VALUES)
    {
        return false;
    }
    {
        JsonbIterator *iterator = JsonbIteratorInit(
            operand->val.binary.data
        );
        JsonbValue item;
        JsonbIteratorToken token = JsonbIteratorNext(
            &iterator,
            &item,
            true
        );

        if (token != WJB_BEGIN_ARRAY)
        {
            return false;
        }
        while ((token = JsonbIteratorNext(&iterator, &item, true)) !=
               WJB_END_ARRAY)
        {
            if (token != WJB_ELEM)
            {
                return false;
            }
            if (item.type != jbvNull &&
                !evoke_filter_scope_add_operand(
                    reader,
                    heap_relation,
                    column_index,
                    kind,
                    &item,
                    bitmap))
            {
                return false;
            }
        }
    }
    return true;
}

static bool
evoke_filter_scope_parse(
    const evoke_filter_scope_reader *reader,
    Relation heap_relation,
    Jsonb *filters,
    uint8 **bitmap_out,
    uint64 *allowed_count_out,
    bool *fully_resolved_out,
    evoke_filter_scope_trace *trace_out
)
{
    JsonbIterator *iterator;
    JsonbValue item;
    JsonbIteratorToken token;
    Size bitmap_size = Max(
        (Size) ((reader->header.document_count + UINT64_C(7)) / 8U),
        (Size) 1
    );
    uint8 *result = NULL;
    uint32 predicate_count = 0;
    uint32 resolved_predicate_count = 0;

    if (filters == NULL || !JB_ROOT_IS_OBJECT(filters) ||
        JB_ROOT_COUNT(filters) == 0 ||
        JB_ROOT_COUNT(filters) > EVOKE_FILTER_MAX_PREDICATES)
    {
        return false;
    }
    iterator = JsonbIteratorInit(&filters->root);
    token = JsonbIteratorNext(&iterator, &item, true);
    if (token != WJB_BEGIN_OBJECT)
    {
        return false;
    }
    while ((token = JsonbIteratorNext(&iterator, &item, true)) !=
           WJB_END_OBJECT)
    {
        JsonbValue predicate;
        JsonbIterator *predicate_iterator;
        JsonbValue operation_value;
        JsonbValue operand;
        char *column_name = NULL;
        char *operation = NULL;
        uint8 *predicate_bitmap = NULL;
        int column_index;
        bool supported = false;

        if (token != WJB_KEY)
        {
            goto unsupported;
        }
        column_name = evoke_filter_json_key(&item);
        token = JsonbIteratorNext(&iterator, &predicate, true);
        if (token != WJB_VALUE || predicate.type != jbvBinary ||
            !JsonContainerIsObject(predicate.val.binary.data) ||
            JsonContainerSize(predicate.val.binary.data) != 1)
        {
            goto unsupported;
        }
        predicate_iterator = JsonbIteratorInit(predicate.val.binary.data);
        if (JsonbIteratorNext(
                &predicate_iterator,
                &operation_value,
                true) != WJB_BEGIN_OBJECT ||
            JsonbIteratorNext(
                &predicate_iterator,
                &operation_value,
                true) != WJB_KEY)
        {
            goto unsupported;
        }
        operation = evoke_filter_json_key(&operation_value);
        if (JsonbIteratorNext(
                &predicate_iterator,
                &operand,
                true) != WJB_VALUE)
        {
            goto unsupported;
        }
        column_index = evoke_filter_scope_find_column(reader, column_name);
        if (column_index < 0)
        {
            goto unsupported;
        }
        predicate_bitmap = palloc0(bitmap_size);
        supported = evoke_filter_scope_build_predicate(
            reader,
            heap_relation,
            (uint32) column_index,
            operation,
            &operand,
            predicate_bitmap,
            trace_out
        );
        if (!supported)
        {
            goto unsupported;
        }
        if (result == NULL)
        {
            result = predicate_bitmap;
            predicate_bitmap = NULL;
        }
        else
        {
            for (Size byte = 0; byte < bitmap_size; byte++)
            {
                result[byte] &= predicate_bitmap[byte];
            }
        }
unsupported:
        if (predicate_bitmap != NULL)
        {
            pfree(predicate_bitmap);
        }
        if (operation != NULL)
        {
            pfree(operation);
        }
        if (column_name != NULL)
        {
            pfree(column_name);
        }
        if (!supported)
        {
            predicate_count++;
            continue;
        }
        resolved_predicate_count++;
        predicate_count++;
    }
    if (predicate_count == 0 || resolved_predicate_count == 0 ||
        result == NULL)
    {
        if (result != NULL)
        {
            pfree(result);
        }
        return false;
    }
    *allowed_count_out = 0;
    for (uint32 document = 0;
         document < reader->header.document_count;
         document++)
    {
        if ((result[document >> 3] &
             (UINT8_C(1) << (document & 7U))) != 0)
        {
            (*allowed_count_out)++;
        }
    }
    *bitmap_out = result;
    *fully_resolved_out =
        resolved_predicate_count == predicate_count;
    if (trace_out != NULL)
    {
        trace_out->predicate_count = predicate_count;
        trace_out->resolved_predicate_count = resolved_predicate_count;
    }
    return true;
}

bool
evoke_filter_try_scope_bitmap(
    Relation index_relation,
    Relation heap_relation,
    const evoke_segment_read_root *root,
    Jsonb *filters,
    uint8 **bitmap_out,
    uint64 *document_slot_count_out,
    uint64 *allowed_document_count_out,
    bool *fully_resolved_out,
    evoke_filter_scope_trace *trace_out
)
{
    evoke_filter_scope_reader reader;
    bool resolved = false;

    if (index_relation == NULL || heap_relation == NULL || root == NULL ||
        filters == NULL || bitmap_out == NULL ||
        document_slot_count_out == NULL ||
        allowed_document_count_out == NULL || fully_resolved_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke scope filter request")));
    }
    *bitmap_out = NULL;
    *document_slot_count_out = 0;
    *allowed_document_count_out = 0;
    *fully_resolved_out = false;
    if (trace_out != NULL)
    {
        memset(trace_out, 0, sizeof(*trace_out));
        if (JB_ROOT_IS_OBJECT(filters))
        {
            trace_out->predicate_count = JB_ROOT_COUNT(filters);
        }
    }
    evoke_filter_scope_reader_init(&reader);
    PG_TRY();
    {
        if (evoke_filter_scope_reader_open(&reader, index_relation, root))
        {
            resolved = evoke_filter_scope_parse(
                &reader,
                heap_relation,
                filters,
                bitmap_out,
                allowed_document_count_out,
                fully_resolved_out,
                trace_out
            );
            if (resolved)
            {
                *document_slot_count_out = reader.header.document_count;
                if (reader.header.document_count !=
                        root->next_document_slot ||
                    root->active_l0.record_count > 0 ||
                    root->pending_l0.record_count > 0 ||
                    evoke_segment_manifest_semantic_accelerator_baseline_sequence(
                        &reader.context.manifest
                    ) < reader.context.manifest.max_sequence)
                {
                    /*
                     * Scope postings remain authoritative for their sealed
                     * baseline, but linked or sealed successors require a
                     * heap/TID residual for complete filter membership.
                     */
                    *fully_resolved_out = false;
                }
            }
        }
    }
    PG_FINALLY();
    {
        evoke_filter_scope_reader_free(&reader);
    }
    PG_END_TRY();
    return resolved;
}
