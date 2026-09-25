#!/usr/bin/env bash

set -euo pipefail

create_extension() {
    local db_name="$1"
    psql \
        --username "$POSTGRES_USER" \
        --dbname "$db_name" \
        --set ON_ERROR_STOP=1 \
        --command 'CREATE EXTENSION IF NOT EXISTS evoke;'
}

create_extension postgres
create_extension template1

# The official entrypoint restarts PostgreSQL after init scripts finish, so
# this cluster-wide setting activates the unified semantic runtime worker for
# the final server without maintaining a second lifecycle entrypoint.
psql \
    --username "$POSTGRES_USER" \
    --dbname postgres \
    --set ON_ERROR_STOP=1 \
    --command "ALTER SYSTEM SET shared_preload_libraries = 'evoke';" \
    --command \
        "ALTER SYSTEM SET evoke.shared_runtime_size = '64MB';"

if [ "${POSTGRES_DB:-postgres}" != 'postgres' ]; then
    create_extension "${POSTGRES_DB}"
fi
