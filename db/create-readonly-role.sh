#!/bin/bash
# Creates the least-privilege role the API connects as.
#
# Runs once, after 01-init.sql (numeric prefix forces ordering in
# docker-entrypoint-initdb.d), on a fresh data volume. The API only ever reads,
# so api_reader gets SELECT on the three data tables and nothing else — no
# INSERT/UPDATE/DELETE/DDL and no other tables. This is the real enforcement
# behind the read-only guarantee: even if a generated SQL string slipped past
# the application-level validator, the database role physically cannot mutate
# data. statement_timeout caps runaway/`pg_sleep`-style queries.
set -euo pipefail

if [ -z "${API_DB_PASSWORD:-}" ]; then
  echo "ERROR: API_DB_PASSWORD is not set; cannot create api_reader role." >&2
  exit 1
fi

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
  DO \$\$
  BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'api_reader') THEN
      CREATE ROLE api_reader LOGIN PASSWORD '${API_DB_PASSWORD}';
    END IF;
  END
  \$\$;

  -- Least privilege: connect + read the three data tables, nothing more.
  GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO api_reader;
  GRANT USAGE ON SCHEMA public TO api_reader;
  GRANT SELECT ON documents, text_chunks, table_rows TO api_reader;

  -- Defence in depth: never hand the reader write/DDL ability, and cap query time.
  REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER
    ON documents, text_chunks, table_rows FROM api_reader;
  ALTER ROLE api_reader SET statement_timeout = '15s';
EOSQL

echo "api_reader role created with SELECT-only privileges."
