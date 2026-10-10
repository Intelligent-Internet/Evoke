from pathlib import Path

import pytest

from scripts.test_runtime_service_privilege_smoke import privilege_sql


ROOT = Path(__file__).resolve().parents[1]


def test_only_trusted_contract_lookup_changes_identity() -> None:
    source = (ROOT / 'src/evoke_am.c').read_text(encoding='utf-8')
    start = source.index('static void\nevoke_am_semantic_builder_load_contract(')
    end = source.index('\nstatic ', start + 1)
    contract = source[start:end]
    owner_check = contract.index('if (!superuser_arg(index_owner))')
    save = contract.index('GetUserIdAndSecContext(')
    switch = contract.index('SetUserIdAndSecContext(\n        index_owner,')
    execute = contract.index('ret = SPI_execute_with_args(')
    finally_block = contract.index('PG_FINALLY();')
    restore = contract.index(
        'SetUserIdAndSecContext(saved_userid, saved_sec_context);'
    )
    parse_result = contract.index('runtime_abi = SPI_getvalue(')

    assert owner_check < save < switch < execute
    assert execute < finally_block < restore < parse_result
    assert 'SECURITY_RESTRICTED_OPERATION' in contract[switch:execute]
    assert 'quote_identifier(schema_name)' in contract[:switch]
    assert 'builder->index_relation->rd_rel->relowner' in contract
    assert contract.count('SPI_execute_with_args(') == 1


def test_raw_model_path_entrypoints_keep_superuser_guard() -> None:
    source = (ROOT / 'src/evoke_semantic.c').read_text(encoding='utf-8')
    for function in (
        'evoke_index_checkout_manifest_internal_c',
        'evoke_index_checkout_validate_internal_c',
    ):
        start = source.index(f'{function}(PG_FUNCTION_ARGS)')
        end = source.index('\n}', start)
        body = source[start:end]
        assert body.index('evoke_checkout_require_superuser();') < body.index(
            'evoke_checkout_read_manifest_text('
        )


@pytest.mark.parametrize('schema', ['public', 'search_ext', 'quoted"schema'])
def test_privilege_smoke_targets_installed_extension_schema(schema: str) -> None:
    sql = privilege_sql(Path("/model/check'out"), schema)
    identifier = '"' + schema.replace('"', '""') + '"'

    assert f'CREATE EXTENSION evoke WITH SCHEMA {identifier};' in sql
    assert (
        f'CREATE OR REPLACE FUNCTION {identifier}.'
        'evoke_index_runtime_options_internal('
    ) in sql
    assert "model_path = '/model/check''out'" in sql
    assert 'GRANT INSERT (id, body) ON write_docs' in sql
    assert 'model lookup error leaked privileged identity' in sql
    assert 'writer can read raw server-local model paths' in sql
    assert 'index maintenance bypassed INSERT RLS' in sql
