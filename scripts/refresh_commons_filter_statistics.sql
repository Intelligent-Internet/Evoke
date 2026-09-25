\set ON_ERROR_STOP on

-- Run only after restoring the production autovacuum policy. This refreshes
-- planner statistics for the metadata predicates used by the Commons query
-- matrix without scanning Evoke relations or changing scorer settings.
DO $preflight$
DECLARE
    required_column record;
    generated_column record;
    actual_type text;
    actual_generated "char";
    actual_expression text;
BEGIN
    IF current_setting('autovacuum') <> 'on' THEN
        RAISE EXCEPTION
            'refusing statistics refresh while autovacuum is disabled';
    END IF;
    IF current_setting('track_counts') <> 'on' THEN
        RAISE EXCEPTION
            'refusing statistics refresh while track_counts is disabled';
    END IF;

    FOR required_column IN
        SELECT *
        FROM (VALUES
            ('data_arxiv', 'publish_date'),
            ('data_arxiv', 'categories'),
            ('data_arxiv', 'organizations'),
            ('data_pubmed', 'publish_date'),
            ('data_pubmed', 'publish_date_start_bound'),
            ('data_pubmed', 'publish_date_end_bound'),
            ('data_pubmed', 'publish_date_has_day'),
            ('data_pubmed', 'categories'),
            ('data_pubmed', 'journal_title'),
            ('data_pubmed', 'nlm_ta'),
            ('data_policy_ca_chunks', 'policy_ca_doc_id'),
            ('data_policy_tx_chunks', 'policy_tx_doc_id'),
            ('data_policy_wa_chunks', 'policy_wa_doc_id'),
            ('sys_chunks', 'document_id')
        ) AS expected(table_name, column_name)
    LOOP
        IF NOT EXISTS (
            SELECT 1
            FROM pg_attribute AS attribute
            JOIN pg_class AS relation
              ON relation.oid = attribute.attrelid
            JOIN pg_namespace AS namespace
              ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'commons'
              AND relation.relname = required_column.table_name
              AND attribute.attname = required_column.column_name
              AND attribute.attnum > 0
              AND NOT attribute.attisdropped
        ) THEN
            RAISE EXCEPTION
                'missing required Commons filter column %.%',
                required_column.table_name,
                required_column.column_name;
        END IF;
    END LOOP;

    FOR generated_column IN
        SELECT *
        FROM (VALUES
            (
                'publish_date_start_bound',
                'integer',
                'CASE WHEN (((publish_date / 100) % 100) = 0) '
                'THEN (((publish_date / 10000) * 10000) + 101) '
                'WHEN ((publish_date % 100) = 0) '
                'THEN (((publish_date / 100) * 100) + 1) '
                'ELSE publish_date END'
            ),
            (
                'publish_date_end_bound',
                'integer',
                'CASE WHEN (((publish_date / 100) % 100) = 0) '
                'THEN (((publish_date / 10000) * 10000) + 1231) '
                'WHEN ((publish_date % 100) = 0) '
                'THEN (((publish_date / 100) * 100) + 31) '
                'ELSE publish_date END'
            ),
            (
                'publish_date_has_day',
                'boolean',
                '((publish_date % 100) <> 0)'
            )
        ) AS expected(column_name, type_name, expression)
    LOOP
        SELECT
            format_type(attribute.atttypid, attribute.atttypmod),
            attribute.attgenerated,
            pg_get_expr(definition.adbin, definition.adrelid)
        INTO actual_type, actual_generated, actual_expression
        FROM pg_attribute AS attribute
        JOIN pg_class AS relation
          ON relation.oid = attribute.attrelid
        JOIN pg_namespace AS namespace
          ON namespace.oid = relation.relnamespace
        LEFT JOIN pg_attrdef AS definition
          ON definition.adrelid = attribute.attrelid
         AND definition.adnum = attribute.attnum
        WHERE namespace.nspname = 'commons'
          AND relation.relname = 'data_pubmed'
          AND attribute.attname = generated_column.column_name
          AND attribute.attnum > 0
          AND NOT attribute.attisdropped;

        IF actual_type IS DISTINCT FROM generated_column.type_name OR
           actual_generated IS DISTINCT FROM 's'::"char" OR
           regexp_replace(
               actual_expression,
               '[[:space:]]',
               '',
               'g'
           ) IS DISTINCT FROM regexp_replace(
               generated_column.expression,
               '[[:space:]]',
               '',
               'g'
           ) THEN
            RAISE EXCEPTION
                'invalid generated-column contract for '
                'commons.data_pubmed.%',
                generated_column.column_name;
        END IF;
    END LOOP;
END
$preflight$;

ANALYZE commons.data_arxiv (
    publish_date,
    categories,
    organizations
);
ANALYZE commons.data_pubmed (
    publish_date,
    publish_date_start_bound,
    publish_date_end_bound,
    publish_date_has_day,
    categories,
    journal_title,
    nlm_ta
);
ANALYZE commons.data_policy_ca_chunks (policy_ca_doc_id);
ANALYZE commons.data_policy_tx_chunks (policy_tx_doc_id);
ANALYZE commons.data_policy_wa_chunks (policy_wa_doc_id);
ANALYZE commons.sys_chunks (document_id);

SELECT
    stats.relname,
    stats.n_live_tup,
    stats.last_analyze,
    stats.last_autoanalyze
FROM pg_stat_user_tables AS stats
WHERE stats.schemaname = 'commons'
  AND stats.relname IN (
      'data_arxiv',
      'data_pubmed',
      'data_policy_ca_chunks',
      'data_policy_tx_chunks',
      'data_policy_wa_chunks',
      'sys_chunks'
  )
ORDER BY stats.relname;
