-- 0001: add reverse-geocoded address columns to locations
ALTER TABLE locations
  ADD COLUMN IF NOT EXISTS street            text,
  ADD COLUMN IF NOT EXISTS city              text,
  ADD COLUMN IF NOT EXISTS county            text,
  ADD COLUMN IF NOT EXISTS full_address      text,
  ADD COLUMN IF NOT EXISTS address_source    text,
  ADD COLUMN IF NOT EXISTS address_fetched_at timestamptz;
