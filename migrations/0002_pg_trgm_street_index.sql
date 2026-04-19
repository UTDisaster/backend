-- 0002: enable pg_trgm and index locations.street / full_address for fuzzy match
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE INDEX IF NOT EXISTS ix_locations_street_trgm
  ON locations
  USING gin (street gin_trgm_ops);

CREATE INDEX IF NOT EXISTS ix_locations_full_address_trgm
  ON locations
  USING gin (full_address gin_trgm_ops);
