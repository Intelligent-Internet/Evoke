#include "postgres.h"

#include "access/genam.h"
#include "access/table.h"
#include "access/tableam.h"
#include "access/sysattr.h"
#include "catalog/dependency.h"
#include "catalog/namespace.h"
#include "catalog/pg_namespace_d.h"
#include "catalog/pg_proc.h"
#include "commands/defrem.h"
#include "commands/explain.h"
#if PG_VERSION_NUM >= 180000
#include "commands/explain_format.h"
#endif
#include "commands/extension.h"
#include "executor/executor.h"
#include "executor/tuptable.h"
#include "fmgr.h"
#include "nodes/extensible.h"
#include "nodes/bitmapset.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/clauses.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/planner.h"
#include "optimizer/optimizer.h"
#include "optimizer/restrictinfo.h"
#include "optimizer/tlist.h"
#include "parser/parsetree.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/guc.h"
#include "utils/jsonb.h"
#include "utils/jsonfuncs.h"
#include "utils/lsyscache.h"
#include "utils/snapmgr.h"

#include "evoke_am_query.h"
#include "evoke_am_options.h"
#include "evoke_filter.h"
#include "evoke_planner.h"

#define EVOKE_PLANNER_NAME "Evoke Search"
#define EVOKE_PLANNER_SCOPE_OVERFETCH_MULTIPLIER 4

typedef enum evoke_planner_scope_operation
{
    EVOKE_PLANNER_SCOPE_EQ = 1,
    EVOKE_PLANNER_SCOPE_OVERLAP,
    EVOKE_PLANNER_SCOPE_ILIKE,
    EVOKE_PLANNER_SCOPE_GT,
    EVOKE_PLANNER_SCOPE_GTE,
    EVOKE_PLANNER_SCOPE_LT,
    EVOKE_PLANNER_SCOPE_LTE
} evoke_planner_scope_operation;

typedef struct evoke_planner_scope_predicate
{
    const char *column_name;
    evoke_planner_scope_operation operation;
    ExprState *operand_expression;
    Oid operand_type;
} evoke_planner_scope_predicate;

typedef struct evoke_planner_marker
{
    FuncExpr *function;
    TargetEntry *target;
    Oid index_oid;
} evoke_planner_marker;

typedef struct evoke_planner_state
{
    CustomScanState custom;
    ExprState *query_expression;
    ExprState *field_names_expression;
    ExprState *field_weights_expression;
    ExprState *limit_expression;
    ExprState *filter_qual;
    evoke_planner_scope_predicate *scope_predicates;
    size_t scope_predicate_count;
    PlanState *filter_plan;
    TupleTableSlot *heap_slot;
    ItemPointerData *tids;
    float *scores;
    uint64 hit_count;
    uint64 next_hit;
    uint64 allowed_tid_count;
    uint64 scope_candidate_count;
    uint64 scope_match_count;
    Oid index_oid;
    Oid relation_oid;
    bool scope_attempted;
    bool scope_used;
    bool scope_fallback;
    bool explain_only;
} evoke_planner_state;

static bool evoke_enable_planner_native = true;
static bool evoke_planner_initialized = false;
static create_upper_paths_hook_type evoke_previous_upper_paths_hook = NULL;

static Plan *evoke_plan_custom_path(
    PlannerInfo *root,
    RelOptInfo *rel,
    CustomPath *best_path,
    List *target_list,
    List *clauses,
    List *custom_plans
);
static Node *evoke_create_custom_scan_state(CustomScan *scan);
static void evoke_begin_custom_scan(
    CustomScanState *state,
    EState *estate,
    int flags
);
static TupleTableSlot *evoke_exec_custom_scan(CustomScanState *state);
static void evoke_end_custom_scan(CustomScanState *state);
static void evoke_rescan_custom_scan(CustomScanState *state);
static void evoke_explain_custom_scan(
    CustomScanState *state,
    List *ancestors,
    ExplainState *explain_state
);
static TupleTableSlot *evoke_custom_scan_next(ScanState *state);
static bool evoke_custom_scan_recheck(ScanState *state, TupleTableSlot *slot);

static const CustomPathMethods evoke_custom_path_methods = {
    .CustomName = EVOKE_PLANNER_NAME,
    .PlanCustomPath = evoke_plan_custom_path
};

static const CustomScanMethods evoke_custom_scan_methods = {
    .CustomName = EVOKE_PLANNER_NAME,
    .CreateCustomScanState = evoke_create_custom_scan_state
};

static const CustomExecMethods evoke_custom_exec_methods = {
    .CustomName = EVOKE_PLANNER_NAME,
    .BeginCustomScan = evoke_begin_custom_scan,
    .ExecCustomScan = evoke_exec_custom_scan,
    .EndCustomScan = evoke_end_custom_scan,
    .ReScanCustomScan = evoke_rescan_custom_scan,
    .ExplainCustomScan = evoke_explain_custom_scan
};

static bool
evoke_is_rank_marker(FuncExpr *function)
{
    Oid extension_oid;
    char *function_name;
    bool matches;

    int argument_count;

    if (function == NULL)
    {
        return false;
    }
    argument_count = list_length(function->args);
    if (function->funcresulttype != FLOAT4OID ||
        (argument_count != 2 && argument_count != 4) ||
        exprType(linitial(function->args)) != REGCLASSOID ||
        exprType(lsecond(function->args)) != TEXTOID ||
        (argument_count == 4 &&
         (exprType(list_nth(function->args, 2)) != TEXTARRAYOID ||
          exprType(list_nth(function->args, 3)) != FLOAT4ARRAYOID)))
    {
        return false;
    }

    extension_oid = get_extension_oid("evoke", true);
    if (!OidIsValid(extension_oid) ||
        getExtensionOfObject(ProcedureRelationId, function->funcid) !=
            extension_oid)
    {
        return false;
    }

    function_name = get_func_name(function->funcid);
    matches = function_name != NULL &&
        strcmp(function_name, "evoke_query") == 0;
    if (function_name != NULL)
    {
        pfree(function_name);
    }
    return matches;
}

static bool
evoke_extract_rank_marker_from_target_list(
    List *target_list,
    List *sort_clauses,
    evoke_planner_marker *marker
)
{
    ListCell *sort_cell;

    foreach(sort_cell, sort_clauses)
    {
        SortGroupClause *sort_clause = lfirst_node(
            SortGroupClause,
            sort_cell
        );
        ListCell *target_cell;

        foreach(target_cell, target_list)
        {
            TargetEntry *target = lfirst_node(TargetEntry, target_cell);
            FuncExpr *function;
            Node *index_argument;
            char *sort_operator_name;

            if (target->ressortgroupref != sort_clause->tleSortGroupRef ||
                !IsA(target->expr, FuncExpr))
            {
                continue;
            }
            function = castNode(FuncExpr, target->expr);
            if (!evoke_is_rank_marker(function))
            {
                continue;
            }
            sort_operator_name = get_opname(sort_clause->sortop);
            if (sort_operator_name == NULL ||
                strcmp(sort_operator_name, ">") != 0)
            {
                if (sort_operator_name != NULL)
                {
                    pfree(sort_operator_name);
                }
                return false;
            }
            pfree(sort_operator_name);
            index_argument = linitial(function->args);
            if (!IsA(index_argument, Const) ||
                castNode(Const, index_argument)->constisnull)
            {
                return false;
            }
            marker->function = function;
            marker->target = target;
            marker->index_oid = DatumGetObjectId(
                castNode(Const, index_argument)->constvalue
            );
            return true;
        }
    }
    return false;
}

static bool
evoke_extract_rank_marker(
    PlannerInfo *root,
    evoke_planner_marker *marker
)
{
    memset(marker, 0, sizeof(*marker));
    if (evoke_extract_rank_marker_from_target_list(
            root->processed_tlist,
            root->parse->sortClause,
            marker
        ))
    {
        return true;
    }
    return evoke_extract_rank_marker_from_target_list(
        root->parse->targetList,
        root->parse->sortClause,
        marker
    );
}

static bool
evoke_relation_owns_index(RelOptInfo *relation, Oid index_oid)
{
    ListCell *index_cell;
    Oid evoke_am_oid = get_am_oid("evoke", true);

    if (!OidIsValid(evoke_am_oid))
    {
        return false;
    }
    foreach(index_cell, relation->indexlist)
    {
        IndexOptInfo *index = lfirst_node(IndexOptInfo, index_cell);

        if (index->indexoid == index_oid && index->relam == evoke_am_oid)
        {
            Relation index_relation = index_open(
                index_oid,
                AccessShareLock
            );
            bool semantic = evoke_am_sae_enabled(index_relation);

            /*
             * The Evoke root, not the heap relation, defines the searchable
             * corpus.  An allowed-TID query maps heap TIDs through that
             * root's document directory, so rows outside a partial index
             * predicate are discarded by the same scorer used by the
             * explicit-hit API.  Requiring predOK here would instead force
             * applications to repeat an index visibility predicate in every
             * natural search query.
             */

            index_close(index_relation, AccessShareLock);
            return semantic;
        }
    }
    return false;
}

static bool
evoke_marker_arguments_supported(FuncExpr *marker)
{
    ListCell *argument_cell;
    int argument_number = 0;

    foreach(argument_cell, marker->args)
    {
        Node *argument = lfirst(argument_cell);

        if (argument_number > 0 &&
            (contain_var_clause(argument) ||
             contain_volatile_functions(argument)))
        {
            return false;
        }
        argument_number++;
    }
    return true;
}

static bool
evoke_query_shape_supported(
    PlannerInfo *root,
    Index range_table_index,
    FuncExpr *marker
)
{
    Query *query = root->parse;
    Node *from_item;

    if (list_length(query->sortClause) != 1 ||
        query->limitOption != LIMIT_OPTION_COUNT ||
        query->hasAggs ||
        query->hasWindowFuncs ||
        query->hasTargetSRFs ||
        query->hasSubLinks ||
        query->hasDistinctOn ||
        query->hasRecursive ||
        query->hasModifyingCTE ||
        query->hasForUpdate ||
        query->hasRowSecurity ||
#if PG_VERSION_NUM >= 180000
        query->hasGroupRTE ||
#endif
        query->cteList != NIL ||
        query->groupClause != NIL ||
        query->groupingSets != NIL ||
        query->havingQual != NULL ||
        query->windowClause != NIL ||
        query->distinctClause != NIL ||
        query->rowMarks != NIL ||
        query->setOperations != NULL ||
        query->jointree == NULL ||
        list_length(query->jointree->fromlist) != 1 ||
        bms_num_members(root->all_baserels) != 1 ||
        !bms_is_member(range_table_index, root->all_baserels) ||
        !evoke_marker_arguments_supported(marker))
    {
        return false;
    }
    from_item = linitial(query->jointree->fromlist);
    return IsA(from_item, RangeTblRef) &&
        castNode(RangeTblRef, from_item)->rtindex == range_table_index;
}

static Path *
evoke_cheapest_unparameterized_path(RelOptInfo *relation)
{
    Path *cheapest = NULL;
    ListCell *path_cell;

    foreach(path_cell, relation->pathlist)
    {
        Path *candidate = lfirst_node(Path, path_cell);
        Path *tid_source = candidate;

        while (IsA(tid_source, ProjectionPath))
        {
            tid_source = castNode(ProjectionPath, tid_source)->subpath;
        }
        if (candidate->param_info != NULL ||
            tid_source->pathtype == T_IndexOnlyScan)
        {
            continue;
        }
        if (cheapest == NULL ||
            compare_path_costs(candidate, cheapest, TOTAL_COST) < 0)
        {
            cheapest = candidate;
        }
    }
    return cheapest;
}

static Path *
evoke_build_filter_tid_path(
    PlannerInfo *root,
    RelOptInfo *relation,
    Index range_table_index
)
{
    Path *source_path;
    Path *filter_tid_path;
    PathTarget *target;
    Var *tid;

    if (relation->baserestrictinfo == NIL)
    {
        return NULL;
    }
    source_path = evoke_cheapest_unparameterized_path(relation);
    if (source_path == NULL)
    {
        return NULL;
    }
    tid = makeVar(
        range_table_index,
        SelfItemPointerAttributeNumber,
        TIDOID,
        -1,
        InvalidOid,
        0
    );
    target = create_empty_pathtarget();
    add_column_to_pathtarget(target, (Expr *) tid, 0);
    filter_tid_path = (Path *) create_projection_path(
        root,
        relation,
        source_path,
        target
    );
    return filter_tid_path;
}

static Node *
evoke_planner_scope_unwrap(Node *expression)
{
    Node *current = expression;

    while (current != NULL)
    {
        /* An explicit collation may not match the scope artifact collation. */
        if (IsA(current, CollateExpr))
        {
            return NULL;
        }
        if (IsA(current, RelabelType))
        {
            RelabelType *relabel = castNode(RelabelType, current);

            if (relabel->resulttype != exprType((Node *) relabel->arg))
            {
                return NULL;
            }
            current = (Node *) relabel->arg;
            continue;
        }
        break;
    }
    return current;
}

static bool
evoke_planner_scope_scalar_type_supported(Oid type_oid)
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
evoke_planner_scope_string_type_supported(Oid type_oid)
{
    char type_category = TYPCATEGORY_INVALID;
    bool type_preferred = false;

    get_type_category_preferred(
        getBaseType(type_oid),
        &type_category,
        &type_preferred
    );
    return type_category == TYPCATEGORY_STRING;
}

static bool
evoke_planner_scope_operation_is_range(
    evoke_planner_scope_operation operation
)
{
    return operation == EVOKE_PLANNER_SCOPE_GT ||
        operation == EVOKE_PLANNER_SCOPE_GTE ||
        operation == EVOKE_PLANNER_SCOPE_LT ||
        operation == EVOKE_PLANNER_SCOPE_LTE;
}

static evoke_planner_scope_operation
evoke_planner_scope_reverse_operation(
    evoke_planner_scope_operation operation
)
{
    switch (operation)
    {
        case EVOKE_PLANNER_SCOPE_GT:
            return EVOKE_PLANNER_SCOPE_LT;
        case EVOKE_PLANNER_SCOPE_GTE:
            return EVOKE_PLANNER_SCOPE_LTE;
        case EVOKE_PLANNER_SCOPE_LT:
            return EVOKE_PLANNER_SCOPE_GT;
        case EVOKE_PLANNER_SCOPE_LTE:
            return EVOKE_PLANNER_SCOPE_GTE;
        default:
            return operation;
    }
}

static bool
evoke_planner_scope_attribute_is_included(
    Relation index_relation,
    AttrNumber heap_attribute
)
{
    for (int attribute = index_relation->rd_index->indnkeyatts;
         attribute < index_relation->rd_index->indnatts;
         attribute++)
    {
        if (index_relation->rd_index->indkey.values[attribute] ==
            heap_attribute)
        {
            return true;
        }
    }
    return false;
}

static bool
evoke_planner_scope_spec_conflicts(
    List *specifications,
    const char *column_name,
    evoke_planner_scope_operation operation
)
{
    ListCell *specification_cell;

    foreach(specification_cell, specifications)
    {
        List *specification = castNode(List, lfirst(specification_cell));
        const char *existing_column = strVal(linitial(specification));
        evoke_planner_scope_operation existing_operation =
            (evoke_planner_scope_operation) intVal(lsecond(specification));

        if (strcmp(existing_column, column_name) != 0)
        {
            continue;
        }
        if (!evoke_planner_scope_operation_is_range(operation) ||
            !evoke_planner_scope_operation_is_range(existing_operation) ||
            operation == existing_operation)
        {
            return true;
        }
    }
    return false;
}

static bool
evoke_planner_build_scope_filter_spec(
    Oid index_oid,
    Index range_table_index,
    List *filter_quals,
    List **specifications_out,
    List **operands_out
)
{
    Relation index_relation;
    List *specifications = NIL;
    List *operands = NIL;
    ListCell *qual_cell;
    bool supported = filter_quals != NIL;

    *specifications_out = NIL;
    *operands_out = NIL;
    if (filter_quals == NIL)
    {
        return false;
    }
    index_relation = index_open(index_oid, AccessShareLock);
    foreach(qual_cell, filter_quals)
    {
        Node *clause = lfirst(qual_cell);
        OpExpr *operation_expression;
        Node *left;
        Node *right;
        Node *operand;
        Var *variable;
        Oid variable_type;
        Oid operand_type;
        Oid element_type;
        char *operator_name;
        char *column_name;
        evoke_planner_scope_operation operation;
        bool variable_on_right = false;

        if (!IsA(clause, OpExpr))
        {
            supported = false;
            break;
        }
        operation_expression = castNode(OpExpr, clause);
        if (list_length(operation_expression->args) != 2 ||
            get_func_namespace(get_opcode(operation_expression->opno)) !=
                PG_CATALOG_NAMESPACE)
        {
            supported = false;
            break;
        }
        left = evoke_planner_scope_unwrap(
            linitial(operation_expression->args)
        );
        right = evoke_planner_scope_unwrap(
            lsecond(operation_expression->args)
        );
        if (left != NULL && IsA(left, Var))
        {
            variable = castNode(Var, left);
            operand = lsecond(operation_expression->args);
        }
        else if (right != NULL && IsA(right, Var))
        {
            variable = castNode(Var, right);
            operand = linitial(operation_expression->args);
            variable_on_right = true;
        }
        else
        {
            supported = false;
            break;
        }
        if (variable->varno != range_table_index ||
            variable->varlevelsup != 0 || variable->varattno <= 0 ||
            evoke_planner_scope_unwrap(operand) == NULL ||
            contain_var_clause(operand) ||
            contain_volatile_functions(operand) ||
            !evoke_planner_scope_attribute_is_included(
                index_relation,
                variable->varattno
            ))
        {
            supported = false;
            break;
        }
        variable_type = get_atttype(
            index_relation->rd_index->indrelid,
            variable->varattno
        );
        operator_name = get_opname(operation_expression->opno);
        if (operator_name == NULL)
        {
            supported = false;
            break;
        }
        operand_type = exprType(operand);
        if (!OidIsValid(variable_type) || !OidIsValid(operand_type) ||
            (strcmp(operator_name, "~~*") != 0 &&
             variable_type != operand_type))
        {
            pfree(operator_name);
            supported = false;
            break;
        }
        element_type = get_element_type(getBaseType(variable_type));
        if (strcmp(operator_name, "=") == 0 &&
            element_type == InvalidOid &&
            evoke_planner_scope_scalar_type_supported(variable_type))
        {
            operation = EVOKE_PLANNER_SCOPE_EQ;
        }
        else if (strcmp(operator_name, "&&") == 0 &&
                 element_type != InvalidOid &&
                 evoke_planner_scope_scalar_type_supported(element_type))
        {
            operation = EVOKE_PLANNER_SCOPE_OVERLAP;
        }
        else if (strcmp(operator_name, "~~*") == 0 &&
                 !variable_on_right &&
                 element_type == InvalidOid &&
                 evoke_planner_scope_string_type_supported(variable_type) &&
                 evoke_planner_scope_string_type_supported(operand_type))
        {
            operation = EVOKE_PLANNER_SCOPE_ILIKE;
        }
        else if (element_type == InvalidOid &&
                 evoke_planner_scope_scalar_type_supported(variable_type) &&
                 strcmp(operator_name, ">") == 0)
        {
            operation = EVOKE_PLANNER_SCOPE_GT;
        }
        else if (element_type == InvalidOid &&
                 evoke_planner_scope_scalar_type_supported(variable_type) &&
                 strcmp(operator_name, ">=") == 0)
        {
            operation = EVOKE_PLANNER_SCOPE_GTE;
        }
        else if (element_type == InvalidOid &&
                 evoke_planner_scope_scalar_type_supported(variable_type) &&
                 strcmp(operator_name, "<") == 0)
        {
            operation = EVOKE_PLANNER_SCOPE_LT;
        }
        else if (element_type == InvalidOid &&
                 evoke_planner_scope_scalar_type_supported(variable_type) &&
                 strcmp(operator_name, "<=") == 0)
        {
            operation = EVOKE_PLANNER_SCOPE_LTE;
        }
        else
        {
            pfree(operator_name);
            supported = false;
            break;
        }
        pfree(operator_name);
        if (variable_on_right)
        {
            operation = evoke_planner_scope_reverse_operation(operation);
        }
        column_name = get_attname(
            index_relation->rd_index->indrelid,
            variable->varattno,
            false
        );
        if (evoke_planner_scope_spec_conflicts(
                specifications,
                column_name,
                operation
            ))
        {
            pfree(column_name);
            supported = false;
            break;
        }
        specifications = lappend(
            specifications,
            list_make2(
                makeString(column_name),
                makeInteger((int) operation)
            )
        );
        operands = lappend(operands, copyObject(operand));
    }
    index_close(index_relation, AccessShareLock);
    if (!supported)
    {
        list_free_deep(specifications);
        list_free_deep(operands);
        return false;
    }

    *specifications_out = specifications;
    *operands_out = operands;
    return true;
}

static void
evoke_create_upper_paths(
    PlannerInfo *root,
    UpperRelationKind stage,
    RelOptInfo *input_relation,
    RelOptInfo *output_relation,
    void *extra
)
{
    evoke_planner_marker marker;
    RelOptInfo *relation;
    RangeTblEntry *range_table_entry;
    CustomPath *path;
    Path *filter_tid_path;
    Index range_table_index;

    if (evoke_previous_upper_paths_hook != NULL)
    {
        evoke_previous_upper_paths_hook(
            root,
            stage,
            input_relation,
            output_relation,
            extra
        );
    }
    if (!evoke_enable_planner_native ||
        stage != UPPERREL_ORDERED ||
        root->parse->commandType != CMD_SELECT ||
        root->parse->limitCount == NULL ||
        root->parse->limitOffset != NULL ||
        bms_num_members(root->all_baserels) != 1 ||
        root->upper_targets[UPPERREL_ORDERED] == NULL)
    {
        return;
    }
    range_table_index = bms_singleton_member(root->all_baserels);
    if (range_table_index <= 0 ||
        range_table_index >= root->simple_rel_array_size)
    {
        return;
    }
    relation = root->simple_rel_array[range_table_index];
    range_table_entry = planner_rt_fetch(range_table_index, root);
    if (relation == NULL ||
        input_relation != relation ||
        range_table_entry->rtekind != RTE_RELATION ||
        relation->reloptkind != RELOPT_BASEREL ||
        range_table_entry->securityQuals != NIL ||
        !evoke_extract_rank_marker(root, &marker) ||
        !evoke_query_shape_supported(root, range_table_index, marker.function) ||
        !evoke_relation_owns_index(relation, marker.index_oid))
    {
        return;
    }
    filter_tid_path = evoke_build_filter_tid_path(
        root,
        relation,
        range_table_index
    );
    if (relation->baserestrictinfo != NIL && filter_tid_path == NULL)
    {
        return;
    }

    path = makeNode(CustomPath);
    path->path.pathtype = T_CustomScan;
    path->path.parent = output_relation;
    path->path.pathtarget = root->upper_targets[UPPERREL_ORDERED];
    path->path.param_info = NULL;
    path->path.parallel_aware = false;
    path->path.parallel_safe = false;
    path->path.parallel_workers = 0;
    path->path.rows = Max(1.0, Min(relation->rows, 100.0));
#if PG_VERSION_NUM >= 180000
    path->path.disabled_nodes = 0;
#endif
    path->path.startup_cost = 1.0 +
        (filter_tid_path == NULL ? 0.0 : filter_tid_path->startup_cost);
    path->path.total_cost = 1.0 + path->path.rows * cpu_tuple_cost +
        (filter_tid_path == NULL ? 0.0 : filter_tid_path->total_cost);
    path->path.pathkeys = root->sort_pathkeys;
    path->flags = CUSTOMPATH_SUPPORT_PROJECTION;
    path->custom_paths = filter_tid_path == NULL
        ? NIL
        : list_make1(filter_tid_path);
    path->custom_restrictinfo = list_copy(relation->baserestrictinfo);
    path->custom_private = list_make5(
        makeConst(
            OIDOID,
            -1,
            InvalidOid,
            sizeof(Oid),
            ObjectIdGetDatum(marker.index_oid),
            false,
            true
        ),
        makeConst(
            OIDOID,
            -1,
            InvalidOid,
            sizeof(Oid),
            ObjectIdGetDatum(range_table_entry->relid),
            false,
            true
        ),
        makeConst(
            INT4OID,
            -1,
            InvalidOid,
            sizeof(int32),
            Int32GetDatum((int32) range_table_index),
            false,
            true
        ),
        copyObject(marker.function),
        copyObject(root->parse->limitCount)
    );
    path->methods = &evoke_custom_path_methods;
    add_path(output_relation, &path->path);
}

static List *
evoke_build_custom_scan_tuple_list(
    Oid relation_oid,
    Index scan_relation_id,
    FuncExpr *marker,
    AttrNumber *ctid_attribute_out,
    AttrNumber *tableoid_attribute_out,
    AttrNumber *score_attribute_out
)
{
    Relation relation;
    TupleDesc descriptor;
    List *target_list = NIL;
    AttrNumber attribute_number;

    relation = table_open(relation_oid, NoLock);
    descriptor = RelationGetDescr(relation);
    for (attribute_number = 1;
         attribute_number <= descriptor->natts;
         attribute_number++)
    {
        Form_pg_attribute attribute = TupleDescAttr(
            descriptor,
            attribute_number - 1
        );
        Oid attribute_type = attribute->attisdropped
            ? TEXTOID
            : attribute->atttypid;
        int32 attribute_typmod = attribute->attisdropped
            ? -1
            : attribute->atttypmod;
        Oid attribute_collation = attribute->attisdropped
            ? InvalidOid
            : attribute->attcollation;

        target_list = lappend(
            target_list,
            makeTargetEntry(
                (Expr *) makeVar(
                    scan_relation_id,
                    attribute_number,
                    attribute_type,
                    attribute_typmod,
                    attribute_collation,
                    0
                ),
                attribute_number,
                pstrdup(NameStr(attribute->attname)),
                false
            )
        );
    }
    *ctid_attribute_out = descriptor->natts + 1;
    target_list = lappend(
        target_list,
        makeTargetEntry(
            (Expr *) makeVar(
                scan_relation_id,
                SelfItemPointerAttributeNumber,
                TIDOID,
                -1,
                InvalidOid,
                0
            ),
            *ctid_attribute_out,
            pstrdup("ctid"),
            false
        )
    );
    *tableoid_attribute_out = descriptor->natts + 2;
    target_list = lappend(
        target_list,
        makeTargetEntry(
            (Expr *) makeVar(
                scan_relation_id,
                TableOidAttributeNumber,
                OIDOID,
                -1,
                InvalidOid,
                0
            ),
            *tableoid_attribute_out,
            pstrdup("tableoid"),
            false
        )
    );
    *score_attribute_out = descriptor->natts + 3;
    target_list = lappend(
        target_list,
        makeTargetEntry(
            (Expr *) copyObject(marker),
            *score_attribute_out,
            pstrdup("evoke_score"),
            false
        )
    );
    table_close(relation, NoLock);
    return target_list;
}

static Plan *
evoke_plan_custom_path(
    PlannerInfo *root,
    RelOptInfo *relation,
    CustomPath *best_path,
    List *target_list,
    List *clauses,
    List *custom_plans
)
{
    CustomScan *scan = makeNode(CustomScan);
    FuncExpr *marker = castNode(
        FuncExpr,
        list_nth(best_path->custom_private, 3)
    );
    Node *query_expression = copyObject(lsecond(marker->args));
    Node *field_names_expression;
    Node *field_weights_expression;
    Node *limit_expression = copyObject(
        list_nth(best_path->custom_private, 4)
    );
    List *scope_specifications = NIL;
    List *scope_operands = NIL;
    List *filter_quals = extract_actual_clauses(
        best_path->custom_restrictinfo,
        false
    );
    Const *relation_oid_constant = castNode(
        Const,
        lsecond(best_path->custom_private)
    );
    Const *index_oid_constant = castNode(
        Const,
        linitial(best_path->custom_private)
    );
    Const *scan_relation_id_constant = castNode(
        Const,
        list_nth(best_path->custom_private, 2)
    );
    Oid index_oid = DatumGetObjectId(index_oid_constant->constvalue);
    Index scan_relation_id = (Index) DatumGetInt32(
        scan_relation_id_constant->constvalue
    );
    AttrNumber ctid_attribute;
    AttrNumber tableoid_attribute;
    AttrNumber score_attribute;

    (void) relation;
    (void) clauses;
    if (list_length(marker->args) == 4)
    {
        field_names_expression = copyObject(list_nth(marker->args, 2));
        field_weights_expression = copyObject(list_nth(marker->args, 3));
    }
    else
    {
        field_names_expression = (Node *) makeNullConst(
            TEXTARRAYOID,
            -1,
            InvalidOid
        );
        field_weights_expression = (Node *) makeNullConst(
            FLOAT4ARRAYOID,
            -1,
            InvalidOid
        );
    }
    (void) evoke_planner_build_scope_filter_spec(
        index_oid,
        scan_relation_id,
        filter_quals,
        &scope_specifications,
        &scope_operands
    );
    scan->custom_scan_tlist = evoke_build_custom_scan_tuple_list(
        DatumGetObjectId(relation_oid_constant->constvalue),
        scan_relation_id,
        marker,
        &ctid_attribute,
        &tableoid_attribute,
        &score_attribute
    );
    scan->scan.plan.targetlist = copyObject(target_list);
    /*
     * Keep the relation predicate in PostgreSQL's standard qual pipeline.
     * Returned rows retain PostgreSQL's normal executor recheck semantics.
     */
    scan->scan.plan.qual = copyObject(filter_quals);
    scan->scan.scanrelid = scan_relation_id;
    scan->flags = best_path->flags;
    scan->custom_plans = custom_plans;
    scan->custom_exprs = list_make4(
        query_expression,
        field_names_expression,
        field_weights_expression,
        limit_expression
    );
    scan->custom_exprs = list_concat(
        scan->custom_exprs,
        scope_operands
    );
    scan->custom_private = list_make3(
        copyObject(linitial(best_path->custom_private)),
        copyObject(lsecond(best_path->custom_private)),
        scope_specifications
    );
    if (!list_member_oid(root->glob->relationOids, index_oid))
    {
        root->glob->relationOids = lappend_oid(
            root->glob->relationOids,
            index_oid
        );
    }
    scan->methods = &evoke_custom_scan_methods;
    return &scan->scan.plan;
}

static Node *
evoke_create_custom_scan_state(CustomScan *scan)
{
    evoke_planner_state *state = palloc0(sizeof(*state));

    (void) scan;
    NodeSetTag(state, T_CustomScanState);
    state->custom.methods = &evoke_custom_exec_methods;
    return (Node *) state;
}

static int32
evoke_planner_limit_value(Datum value, Oid value_type)
{
    int64 limit;

    if (value_type == INT8OID)
    {
        limit = DatumGetInt64(value);
    }
    else if (value_type == INT4OID)
    {
        limit = DatumGetInt32(value);
    }
    else
    {
        ereport(
            ERROR,
            (errmsg("evoke planner LIMIT must be int4 or int8"))
        );
    }
    if (limit < 0 || limit > INT_MAX)
    {
        ereport(
            ERROR,
            (errmsg("evoke planner LIMIT must be between 0 and %d", INT_MAX))
        );
    }
    return (int32) limit;
}

static const char *
evoke_planner_scope_operation_name(
    evoke_planner_scope_operation operation
)
{
    switch (operation)
    {
        case EVOKE_PLANNER_SCOPE_EQ:
            return "eq";
        case EVOKE_PLANNER_SCOPE_OVERLAP:
            return "overlap";
        case EVOKE_PLANNER_SCOPE_ILIKE:
            return "ilike";
        case EVOKE_PLANNER_SCOPE_GT:
            return "gt";
        case EVOKE_PLANNER_SCOPE_GTE:
            return "gte";
        case EVOKE_PLANNER_SCOPE_LT:
            return "lt";
        case EVOKE_PLANNER_SCOPE_LTE:
            return "lte";
        default:
            ereport(ERROR, (errmsg("invalid evoke planner scope operation")));
    }
    return NULL;
}

static void
evoke_planner_push_json_key(
    JsonbParseState **parse_state,
    const char *key
)
{
    JsonbValue value;

    value.type = jbvString;
    value.val.string.val = unconstify(char *, key);
    value.val.string.len = strlen(key);
    (void) pushJsonbValue(parse_state, WJB_KEY, &value);
}

static void
evoke_planner_push_json_datum(
    JsonbParseState **parse_state,
    Datum datum,
    Oid type_oid
)
{
    JsonTypeCategory category;
    Oid output_function;
    Datum jsonb_datum;
    Jsonb *jsonb;
    JsonbValue value;

    json_categorize_type(type_oid, true, &category, &output_function);
    jsonb_datum = datum_to_jsonb(datum, category, output_function);
    jsonb = DatumGetJsonbP(jsonb_datum);
    if (JB_ROOT_IS_SCALAR(jsonb))
    {
        if (!JsonbExtractScalar(&jsonb->root, &value))
        {
            ereport(ERROR, (errmsg("invalid evoke planner scope operand")));
        }
    }
    else
    {
        JsonbToJsonbValue(jsonb, &value);
    }
    (void) pushJsonbValue(parse_state, WJB_VALUE, &value);
}

static bool
evoke_planner_scope_column_emitted(
    const evoke_planner_scope_predicate *predicates,
    size_t predicate_index
)
{
    for (size_t previous = 0; previous < predicate_index; previous++)
    {
        if (strcmp(
                predicates[previous].column_name,
                predicates[predicate_index].column_name
            ) == 0)
        {
            return true;
        }
    }
    return false;
}

static Jsonb *
evoke_planner_build_scope_filter(evoke_planner_state *state)
{
    JsonbParseState *parse_state = NULL;
    JsonbValue *result;

    if (state->scope_predicate_count == 0)
    {
        return NULL;
    }
    (void) pushJsonbValue(&parse_state, WJB_BEGIN_OBJECT, NULL);
    for (size_t predicate_index = 0;
         predicate_index < state->scope_predicate_count;
         predicate_index++)
    {
        evoke_planner_scope_predicate *predicate =
            &state->scope_predicates[predicate_index];
        bool range = evoke_planner_scope_operation_is_range(
            predicate->operation
        );

        if (evoke_planner_scope_column_emitted(
                state->scope_predicates,
                predicate_index
            ))
        {
            continue;
        }
        evoke_planner_push_json_key(&parse_state, predicate->column_name);
        (void) pushJsonbValue(&parse_state, WJB_BEGIN_OBJECT, NULL);
        if (range)
        {
            evoke_planner_push_json_key(&parse_state, "range");
            (void) pushJsonbValue(&parse_state, WJB_BEGIN_OBJECT, NULL);
        }
        for (size_t value_index = predicate_index;
             value_index < state->scope_predicate_count;
             value_index++)
        {
            evoke_planner_scope_predicate *value_predicate =
                &state->scope_predicates[value_index];
            bool is_null;
            Datum value;

            if (strcmp(
                    value_predicate->column_name,
                    predicate->column_name
                ) != 0)
            {
                continue;
            }
            value = ExecEvalExprSwitchContext(
                value_predicate->operand_expression,
                state->custom.ss.ps.ps_ExprContext,
                &is_null
            );
            if (is_null)
            {
                return NULL;
            }
            evoke_planner_push_json_key(
                &parse_state,
                evoke_planner_scope_operation_name(
                    value_predicate->operation
                )
            );
            evoke_planner_push_json_datum(
                &parse_state,
                value,
                value_predicate->operand_type
            );
        }
        if (range)
        {
            (void) pushJsonbValue(&parse_state, WJB_END_OBJECT, NULL);
        }
        (void) pushJsonbValue(&parse_state, WJB_END_OBJECT, NULL);
    }
    result = pushJsonbValue(&parse_state, WJB_END_OBJECT, NULL);
    return JsonbValueToJsonb(result);
}

static int32
evoke_planner_scope_candidate_limit(int32 requested_limit)
{
    if (requested_limit >
        INT_MAX / EVOKE_PLANNER_SCOPE_OVERFETCH_MULTIPLIER)
    {
        return INT_MAX;
    }
    return requested_limit * EVOKE_PLANNER_SCOPE_OVERFETCH_MULTIPLIER;
}

static bool
evoke_planner_store_hit(
    evoke_planner_state *state,
    const ItemPointer tid,
    float score
)
{
    TupleTableSlot *slot = state->custom.ss.ss_ScanTupleSlot;
    int relation_attribute_count =
        RelationGetDescr(state->custom.ss.ss_currentRelation)->natts;

    ExecClearTuple(state->heap_slot);
    if (!table_tuple_fetch_row_version(
            state->custom.ss.ss_currentRelation,
            tid,
            state->custom.ss.ps.state->es_snapshot,
            state->heap_slot
        ))
    {
        return false;
    }
    slot_getallattrs(state->heap_slot);
    ExecClearTuple(slot);
    for (int attribute = 0;
         attribute < relation_attribute_count;
         attribute++)
    {
        slot->tts_values[attribute] = state->heap_slot->tts_values[attribute];
        slot->tts_isnull[attribute] =
            state->heap_slot->tts_isnull[attribute];
    }
    slot->tts_values[relation_attribute_count] = ItemPointerGetDatum(tid);
    slot->tts_isnull[relation_attribute_count] = false;
    slot->tts_values[relation_attribute_count + 1] =
        ObjectIdGetDatum(state->heap_slot->tts_tableOid);
    slot->tts_isnull[relation_attribute_count + 1] = false;
    slot->tts_values[relation_attribute_count + 2] = Float4GetDatum(score);
    slot->tts_isnull[relation_attribute_count + 2] = false;
    slot->tts_tid = state->heap_slot->tts_tid;
    slot->tts_tableOid = state->heap_slot->tts_tableOid;
    ExecStoreVirtualTuple(slot);
    return true;
}

static size_t
evoke_planner_filter_scope_hits(
    evoke_planner_state *state,
    evoke_am_query_hits *hits
)
{
    ExprContext *expression_context = state->custom.ss.ps.ps_ExprContext;
    TupleTableSlot *slot = state->custom.ss.ss_ScanTupleSlot;
    size_t retained = 0;

    if (state->filter_qual == NULL)
    {
        return 0;
    }
    for (size_t rank = 0; rank < hits->len; rank++)
    {
        ResetExprContext(expression_context);
        if (!evoke_planner_store_hit(
                state,
                &hits->tids[rank],
                hits->scores[rank]
            ))
        {
            continue;
        }
        expression_context->ecxt_scantuple = slot;
        if (!ExecQual(state->filter_qual, expression_context))
        {
            continue;
        }
        hits->tids[retained] = hits->tids[rank];
        hits->scores[retained] = hits->scores[rank];
        retained++;
    }
    expression_context->ecxt_scantuple = NULL;
    ExecClearTuple(slot);
    ExecClearTuple(state->heap_slot);
    hits->len = retained;
    return retained;
}

static void
evoke_planner_collect_allowed_tid_keys(
    evoke_planner_state *state,
    uint64 **keys_out,
    size_t *tid_count_out
)
{
    MemoryContext old_context;
    uint64 *keys = NULL;
    size_t count = 0;
    size_t capacity = 0;
    TupleTableSlot *slot;

    *keys_out = NULL;
    *tid_count_out = 0;
    if (state->filter_plan == NULL)
    {
        return;
    }
    old_context = MemoryContextSwitchTo(
        state->custom.ss.ps.state->es_query_cxt
    );
    for (;;)
    {
        Datum tid_value;
        ItemPointer tid;
        bool is_null;

        slot = ExecProcNode(state->filter_plan);
        if (TupIsNull(slot))
        {
            break;
        }
        tid_value = slot_getattr(slot, 1, &is_null);
        if (is_null)
        {
            ereport(ERROR, (errmsg("evoke filter path returned a NULL TID")));
        }
        tid = DatumGetItemPointer(tid_value);
        if (!ItemPointerIsValid(tid))
        {
            ereport(ERROR, (errmsg("evoke filter path returned an invalid TID")));
        }
        if (count == capacity)
        {
            size_t next_capacity = capacity == 0 ? 1024 : capacity * 2;

            if (next_capacity < capacity ||
                next_capacity > MaxAllocSize / sizeof(*keys))
            {
                ereport(ERROR, (errmsg("evoke filter result is too large")));
            }
            keys = keys == NULL
                ? palloc(sizeof(*keys) * next_capacity)
                : repalloc(keys, sizeof(*keys) * next_capacity);
            capacity = next_capacity;
        }
        keys[count++] = evoke_filter_tid_key(tid);
    }
    if (keys == NULL)
    {
        /* A non-NULL pointer distinguishes an empty filter from no filter. */
        keys = palloc(sizeof(*keys));
    }
    count = evoke_filter_sort_unique_tid_keys(keys, count);
    MemoryContextSwitchTo(old_context);
    *keys_out = keys;
    *tid_count_out = count;
}

static void
evoke_planner_load_hits(evoke_planner_state *state)
{
    ExprContext *expression_context = state->custom.ss.ps.ps_ExprContext;
    MemoryContext old_context;
    evoke_am_query_hits hits;
    bool query_is_null;
    bool field_names_is_null;
    bool field_weights_is_null;
    bool limit_is_null;
    Datum query_value;
    Datum field_names_value;
    Datum field_weights_value;
    Datum limit_value;
    text *query_text;
    ArrayType *field_names;
    ArrayType *field_weights;
    Jsonb *scope_filter = NULL;
    uint64 *allowed_tid_keys = NULL;
    size_t allowed_tid_count = 0;
    int32 limit;

    memset(&hits, 0, sizeof(hits));
    state->scope_attempted = false;
    state->scope_used = false;
    state->scope_fallback = false;
    state->scope_candidate_count = 0;
    state->scope_match_count = 0;
    state->custom.ss.ps.qual = state->filter_qual;
    limit_value = ExecEvalExprSwitchContext(
        state->limit_expression,
        expression_context,
        &limit_is_null
    );
    if (limit_is_null)
    {
        ereport(ERROR, (errmsg("evoke planner LIMIT cannot be NULL")));
    }
    limit = evoke_planner_limit_value(
        limit_value,
        exprType((Node *) state->limit_expression->expr)
    );
    if (limit == 0)
    {
        state->allowed_tid_count = 0;
        state->hit_count = 0;
        state->next_hit = 0;
        return;
    }
    query_value = ExecEvalExprSwitchContext(
        state->query_expression,
        expression_context,
        &query_is_null
    );
    field_names_value = ExecEvalExprSwitchContext(
        state->field_names_expression,
        expression_context,
        &field_names_is_null
    );
    field_weights_value = ExecEvalExprSwitchContext(
        state->field_weights_expression,
        expression_context,
        &field_weights_is_null
    );
    if (query_is_null)
    {
        ereport(ERROR, (errmsg("evoke planner query cannot be NULL")));
    }
    /*
     * Scope qualification resets the per-tuple expression context. Keep
     * independent query arguments for the exact fallback.
     */
    old_context = MemoryContextSwitchTo(
        state->custom.ss.ps.state->es_query_cxt
    );
    query_text = DatumGetTextPCopy(query_value);
    field_names = field_names_is_null
        ? NULL
        : DatumGetArrayTypePCopy(field_names_value);
    field_weights = field_weights_is_null
        ? NULL
        : DatumGetArrayTypePCopy(field_weights_value);
    scope_filter = evoke_planner_build_scope_filter(state);
    MemoryContextSwitchTo(old_context);
    if (scope_filter != NULL)
    {
        state->scope_attempted = true;
        if (evoke_am_query_text(
            state->index_oid,
            query_text,
            field_names,
            field_weights,
            scope_filter,
            NULL,
            0,
            evoke_planner_scope_candidate_limit(limit),
            state->custom.ss.ps.state->es_query_cxt,
            &hits
        ))
        {
            state->scope_candidate_count = (uint64) hits.len;
            state->scope_match_count = (uint64)
                evoke_planner_filter_scope_hits(state, &hits);
        }
        if (hits.len >= (size_t) limit)
        {
            hits.len = (size_t) limit;
            state->scope_used = true;
            state->custom.ss.ps.qual = NULL;
            state->hit_count = hits.len;
            state->tids = hits.tids;
            state->scores = hits.scores;
            state->next_hit = 0;
            pfree(scope_filter);
            pfree(query_text);
            if (field_names != NULL)
            {
                pfree(field_names);
            }
            if (field_weights != NULL)
            {
                pfree(field_weights);
            }
            return;
        }
        state->scope_fallback = true;
        if (hits.tids != NULL)
        {
            pfree(hits.tids);
        }
        if (hits.scores != NULL)
        {
            pfree(hits.scores);
        }
        memset(&hits, 0, sizeof(hits));
        pfree(scope_filter);
        scope_filter = NULL;
    }
    evoke_planner_collect_allowed_tid_keys(
        state,
        &allowed_tid_keys,
        &allowed_tid_count
    );
    state->allowed_tid_count = (uint64) allowed_tid_count;
    (void) evoke_am_query_text(
        state->index_oid,
        query_text,
        field_names,
        field_weights,
        NULL,
        state->filter_plan == NULL ? NULL : allowed_tid_keys,
        allowed_tid_count,
        limit,
        state->custom.ss.ps.state->es_query_cxt,
        &hits
    );
    if (allowed_tid_keys != NULL)
    {
        pfree(allowed_tid_keys);
    }
    state->hit_count = hits.len;
    state->tids = hits.tids;
    state->scores = hits.scores;
    state->next_hit = 0;
    pfree(query_text);
    if (field_names != NULL)
    {
        pfree(field_names);
    }
    if (field_weights != NULL)
    {
        pfree(field_weights);
    }
}

static void
evoke_begin_custom_scan(
    CustomScanState *state,
    EState *estate,
    int flags
)
{
    evoke_planner_state *planner_state = (evoke_planner_state *) state;
    CustomScan *scan = castNode(CustomScan, state->ss.ps.plan);
    Const *index_oid_constant = linitial_node(
        Const,
        scan->custom_private
    );
    Const *relation_oid_constant = lsecond_node(
        Const,
        scan->custom_private
    );
    List *scope_specifications = (List *) list_nth(
        scan->custom_private,
        2
    );
    (void) flags;
    planner_state->explain_only = (flags & EXEC_FLAG_EXPLAIN_ONLY) != 0;
    planner_state->index_oid = DatumGetObjectId(
        index_oid_constant->constvalue
    );
    planner_state->relation_oid = DatumGetObjectId(
        relation_oid_constant->constvalue
    );
    planner_state->query_expression = ExecInitExpr(
        linitial_node(Expr, scan->custom_exprs),
        &state->ss.ps
    );
    planner_state->field_names_expression = ExecInitExpr(
        list_nth_node(Expr, scan->custom_exprs, 1),
        &state->ss.ps
    );
    planner_state->field_weights_expression = ExecInitExpr(
        list_nth_node(Expr, scan->custom_exprs, 2),
        &state->ss.ps
    );
    planner_state->limit_expression = ExecInitExpr(
        list_nth_node(Expr, scan->custom_exprs, 3),
        &state->ss.ps
    );
    planner_state->scope_predicate_count = (size_t) list_length(
        scope_specifications
    );
    if (planner_state->scope_predicate_count > 0)
    {
        planner_state->scope_predicates = palloc0(
            planner_state->scope_predicate_count *
            sizeof(*planner_state->scope_predicates)
        );
        for (size_t predicate = 0;
             predicate < planner_state->scope_predicate_count;
             predicate++)
        {
            List *specification = castNode(
                List,
                list_nth(scope_specifications, (int) predicate)
            );
            Expr *operand = list_nth_node(
                Expr,
                scan->custom_exprs,
                4 + (int) predicate
            );

            planner_state->scope_predicates[predicate].column_name =
                strVal(linitial(specification));
            planner_state->scope_predicates[predicate].operation =
                (evoke_planner_scope_operation)
                    intVal(lsecond(specification));
            planner_state->scope_predicates[predicate].operand_expression =
                ExecInitExpr(operand, &state->ss.ps);
            planner_state->scope_predicates[predicate].operand_type =
                exprType((Node *) operand);
        }
    }
    if (scan->custom_plans != NIL)
    {
        ListCell *plan_cell;

        foreach(plan_cell, scan->custom_plans)
        {
            PlanState *child = ExecInitNode(
                lfirst_node(Plan, plan_cell),
                estate,
                flags
            );

            state->custom_ps = lappend(state->custom_ps, child);
        }
        planner_state->filter_plan = linitial_node(
            PlanState,
            state->custom_ps
        );
    }
    planner_state->heap_slot = ExecInitExtraTupleSlot(
        estate,
        RelationGetDescr(state->ss.ss_currentRelation),
        table_slot_callbacks(state->ss.ss_currentRelation)
    );
    planner_state->filter_qual = state->ss.ps.qual;
    if (!planner_state->explain_only)
    {
        evoke_planner_load_hits(planner_state);
    }
}

static TupleTableSlot *
evoke_exec_custom_scan(CustomScanState *state)
{
    return ExecScan(
        &state->ss,
        evoke_custom_scan_next,
        evoke_custom_scan_recheck
    );
}

static TupleTableSlot *
evoke_custom_scan_next(ScanState *scan_state)
{
    evoke_planner_state *state = (evoke_planner_state *) scan_state;
    TupleTableSlot *slot = scan_state->ss_ScanTupleSlot;

    while (state->next_hit < state->hit_count)
    {
        uint64 hit = state->next_hit++;

        if (!evoke_planner_store_hit(
                state,
                &state->tids[hit],
                state->scores[hit]
            ))
        {
            continue;
        }
        return slot;
    }
    return ExecClearTuple(slot);
}

static bool
evoke_custom_scan_recheck(ScanState *state, TupleTableSlot *slot)
{
    (void) state;
    (void) slot;
    return true;
}

static void
evoke_end_custom_scan(CustomScanState *state)
{
    evoke_planner_state *planner_state = (evoke_planner_state *) state;
    ListCell *plan_state_cell;

    if (planner_state->tids != NULL)
    {
        pfree(planner_state->tids);
    }
    if (planner_state->scores != NULL)
    {
        pfree(planner_state->scores);
    }
    planner_state->tids = NULL;
    planner_state->scores = NULL;
    planner_state->hit_count = 0;
    planner_state->next_hit = 0;
    planner_state->allowed_tid_count = 0;
    if (planner_state->scope_predicates != NULL)
    {
        pfree(planner_state->scope_predicates);
        planner_state->scope_predicates = NULL;
        planner_state->scope_predicate_count = 0;
    }
    foreach(plan_state_cell, state->custom_ps)
    {
        ExecEndNode(lfirst_node(PlanState, plan_state_cell));
    }
    planner_state->filter_plan = NULL;
    state->custom_ps = NIL;
}

static void
evoke_rescan_custom_scan(CustomScanState *state)
{
    evoke_planner_state *planner_state = (evoke_planner_state *) state;
    ListCell *plan_state_cell;

    if (planner_state->tids != NULL)
    {
        pfree(planner_state->tids);
    }
    if (planner_state->scores != NULL)
    {
        pfree(planner_state->scores);
    }
    planner_state->tids = NULL;
    planner_state->scores = NULL;
    planner_state->hit_count = 0;
    planner_state->next_hit = 0;
    planner_state->allowed_tid_count = 0;
    foreach(plan_state_cell, state->custom_ps)
    {
        ExecReScan(lfirst_node(PlanState, plan_state_cell));
    }
    evoke_planner_load_hits(planner_state);
}

static void
evoke_explain_custom_scan(
    CustomScanState *state,
    List *ancestors,
    ExplainState *explain_state
)
{
    evoke_planner_state *planner_state = (evoke_planner_state *) state;
    char *index_name = get_rel_name(planner_state->index_oid);

    (void) ancestors;
    ExplainPropertyText(
        "Evoke Index",
        index_name == NULL ? "<dropped>" : index_name,
        explain_state
    );
    ExplainPropertyBool(
        "Filtered",
        planner_state->filter_plan != NULL,
        explain_state
    );
    ExplainPropertyBool(
        "Scope Filter Eligible",
        planner_state->scope_predicate_count > 0,
        explain_state
    );
    if (explain_state->analyze)
    {
        ExplainPropertyUInteger(
            "Allowed TIDs",
            NULL,
            planner_state->allowed_tid_count,
            explain_state
        );
        ExplainPropertyUInteger(
            "Ranked Hits",
            NULL,
            planner_state->hit_count,
            explain_state
        );
        ExplainPropertyUInteger(
            "Scope Filter Probes",
            NULL,
            planner_state->scope_attempted ? 1 : 0,
            explain_state
        );
        ExplainPropertyUInteger(
            "Scope Filter Candidates",
            NULL,
            planner_state->scope_candidate_count,
            explain_state
        );
        ExplainPropertyUInteger(
            "Scope Filter Matches",
            NULL,
            planner_state->scope_match_count,
            explain_state
        );
        ExplainPropertyBool(
            "Scope Filter Complete",
            planner_state->scope_used,
            explain_state
        );
        ExplainPropertyBool(
            "Scope Filter Fallback",
            planner_state->scope_fallback,
            explain_state
        );
    }
    if (index_name != NULL)
    {
        pfree(index_name);
    }
}

PG_FUNCTION_INFO_V1(evoke_query_marker);
Datum
evoke_query_marker(PG_FUNCTION_ARGS)
{
    ereport(
        ERROR,
        (
            errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
            errmsg(
                "scalar evoke_query requires the planner-native Evoke path"
            ),
            errhint(
                "Use a k-bearing evoke_query overload for explicit hits, "
                "or use a supported natural SQL query shape."
            )
        )
    );
    PG_RETURN_NULL();
}

void
evoke_planner_init(void)
{
    if (evoke_planner_initialized)
    {
        return;
    }
    DefineCustomBoolVariable(
        "evoke.enable_planner_native",
        "Enables planner-native Evoke ranked scans.",
        NULL,
        &evoke_enable_planner_native,
        true,
        PGC_USERSET,
        0,
        NULL,
        NULL,
        NULL
    );
    RegisterCustomScanMethods(&evoke_custom_scan_methods);
    evoke_previous_upper_paths_hook = create_upper_paths_hook;
    create_upper_paths_hook = evoke_create_upper_paths;
    evoke_planner_initialized = true;
}

void
evoke_planner_fini(void)
{
    if (!evoke_planner_initialized)
    {
        return;
    }
    if (create_upper_paths_hook == evoke_create_upper_paths)
    {
        create_upper_paths_hook = evoke_previous_upper_paths_hook;
    }
    evoke_planner_initialized = false;
}
