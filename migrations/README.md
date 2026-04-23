# Migrations

Plain SQL schema migrations. No Alembic.

- Files are named `NNNN_description.sql` and applied in lexicographic order.
- `util/migrate.py` walks this directory and executes each file that has not already
  been recorded in the `schema_migrations` table (`name text primary key, applied_at timestamptz`).
- Each file runs inside a single transaction; its filename is inserted into
  `schema_migrations` after a successful apply.
- 0002_pg_trgm_street_index.sql — enables pg_trgm and adds trigram indexes for address/street fuzzy matching

Run with:

```
make migrate
```

or directly:

```
python util/migrate.py
```
