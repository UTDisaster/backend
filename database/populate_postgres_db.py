import os
import json
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

# ── DB Connection ──────────────────────────────────────────────────────────
conn = psycopg2.connect(
    host     = os.getenv("DB_HOST", "localhost"),
    port     = os.getenv("DB_PORT", "5432"),
    dbname   = os.getenv("DB_NAME", "hurricane_db"),
    user     = os.getenv("DB_USER", "postgres"),
    password = os.getenv("DB_PASSWORD", ""),
)
conn.autocommit = False
cur = conn.cursor()

# ── Paths ──────────────────────────────────────────────────────────────────
TILES_PATH  = r"C:\Users\furkm\OneDrive\Desktop\database_setup\CS4485_Hurricane_Crop_Tiles\CS4485_Hurricane_Crop_Tiles"
HOUSES_PATH = r"C:\Users\furkm\OneDrive\Desktop\database_setup\CS4485_Hurricane_Cursor_House\CS4485_Hurricane_Cursor_House"

# ── Create Tables ──────────────────────────────────────────────────────────
print("🔧 Setting up tables...")

cur.execute("DROP TABLE IF EXISTS tile_index CASCADE;")
cur.execute("DROP TABLE IF EXISTS tile_full CASCADE;")
cur.execute("DROP TABLE IF EXISTS house_index CASCADE;")
cur.execute("DROP TABLE IF EXISTS house_full CASCADE;")

cur.execute("""
CREATE TABLE tile_index (
    id                SERIAL PRIMARY KEY,
    image_uid         TEXT NOT NULL,
    tile_uid          TEXT NOT NULL,
    disaster_id       TEXT,
    pair_id           TEXT,
    tile_id           TEXT,
    disaster_type     TEXT,
    capture_date      TEXT,
    building_count    INTEGER,
    classification    TEXT DEFAULT 'unknown',
    prediction        TEXT,
    pre_image_path    TEXT,
    post_image_path   TEXT,
    target_image_path TEXT,
    full_record_key   TEXT UNIQUE,
    UNIQUE(image_uid, tile_uid)
);
""")

cur.execute("""
CREATE TABLE tile_full (
    id                  SERIAL PRIMARY KEY,
    image_uid           TEXT NOT NULL,
    tile_uid            TEXT NOT NULL,
    disaster_id         TEXT,
    pair_id             TEXT,
    tile_id             TEXT,
    sensor              TEXT,
    provider_asset_type TEXT,
    gsd                 TEXT,
    capture_date        TEXT,
    off_nadir_angle     TEXT,
    pan_resolution      TEXT,
    sun_azimuth         TEXT,
    sun_elevation       TEXT,
    target_azimuth      TEXT,
    disaster_type       TEXT,
    catalog_id          TEXT,
    original_width      INTEGER,
    original_height     INTEGER,
    crop_bounds         JSONB,
    buildings           JSONB,
    json_path           TEXT,
    UNIQUE(image_uid, tile_uid)
);
""")

cur.execute("""
CREATE TABLE house_index (
    id                SERIAL PRIMARY KEY,
    image_uid         TEXT NOT NULL,
    house_uid         TEXT NOT NULL,
    disaster_id       TEXT,
    pair_id           TEXT,
    house_id          TEXT,
    disaster_type     TEXT,
    capture_date      TEXT,
    classification    TEXT DEFAULT 'unknown',
    prediction        TEXT,
    pre_image_path    TEXT,
    post_image_path   TEXT,
    target_image_path TEXT,
    full_record_key   TEXT UNIQUE,
    UNIQUE(image_uid, house_uid)
);
""")

cur.execute("""
CREATE TABLE house_full (
    id                  SERIAL PRIMARY KEY,
    image_uid           TEXT NOT NULL,
    house_uid           TEXT NOT NULL,
    disaster_id         TEXT,
    pair_id             TEXT,
    house_id            TEXT,
    sensor              TEXT,
    provider_asset_type TEXT,
    gsd                 TEXT,
    capture_date        TEXT,
    off_nadir_angle     TEXT,
    pan_resolution      TEXT,
    sun_azimuth         TEXT,
    sun_elevation       TEXT,
    target_azimuth      TEXT,
    disaster_type       TEXT,
    catalog_id          TEXT,
    original_width      INTEGER,
    original_height     INTEGER,
    crop_bounds         JSONB,
    points              JSONB,
    classification      TEXT DEFAULT 'unknown',
    prediction          TEXT,
    json_path           TEXT,
    UNIQUE(image_uid, house_uid)
);
""")

conn.commit()
print("  ✅ All tables created.\n")

# ── Load TILES ─────────────────────────────────────────────────────────────
print("📦 Loading tiles...")
tiles_loaded = 0

for image_folder in os.listdir(TILES_PATH):
    folder_path = os.path.join(TILES_PATH, image_folder)
    if not os.path.isdir(folder_path):
        continue

    json_path = os.path.join(folder_path, "master_grid_labels.json")
    if not os.path.exists(json_path):
        print(f"  ⚠️  No master_grid_labels.json in {image_folder}, skipping.")
        continue

    with open(json_path, 'r') as f:
        data = json.load(f)

    meta        = data.get("metadata", {})
    disaster_id = data.get("disasterId") or meta.get("disaster", "hurricane-florence")
    pair_id     = data.get("pairId", "unknown")
    image_uid   = image_folder  # unique per folder e.g. "hurricane-florence_00000543_post_disaster"

    for tile_id, tile_info in data.get("tiles", {}).items():
        tile_uid    = tile_id
        buildings   = tile_info.get("buildings", [])
        tile_folder = os.path.join(folder_path, tile_id)
        record_key  = f"{image_uid}#{tile_uid}"

        # FAST INDEX
        cur.execute("""
            INSERT INTO tile_index (
                image_uid, tile_uid, disaster_id, pair_id, tile_id,
                disaster_type, capture_date, building_count, classification,
                prediction, pre_image_path, post_image_path, target_image_path,
                full_record_key
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (image_uid, tile_uid) DO NOTHING;
        """, (
            image_uid,
            tile_uid,
            disaster_id,
            pair_id,
            tile_id,
            meta.get("disaster_type"),
            meta.get("capture_date"),
            len(buildings),
            "unknown",
            None,
            os.path.join(tile_folder, "pre.png"),
            os.path.join(tile_folder, "post.png"),
            os.path.join(tile_folder, "target.png"),
            record_key,
        ))

        # FULL DATA
        cur.execute("""
            INSERT INTO tile_full (
                image_uid, tile_uid, disaster_id, pair_id, tile_id,
                sensor, provider_asset_type, gsd, capture_date,
                off_nadir_angle, pan_resolution, sun_azimuth, sun_elevation,
                target_azimuth, disaster_type, catalog_id,
                original_width, original_height, crop_bounds, buildings, json_path
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (image_uid, tile_uid) DO NOTHING;
        """, (
            image_uid,
            tile_uid,
            disaster_id,
            pair_id,
            tile_id,
            meta.get("sensor"),
            meta.get("provider_asset_type"),
            str(meta.get("gsd", "")),
            meta.get("capture_date"),
            str(meta.get("off_nadir_angle", "")),
            str(meta.get("pan_resolution", "")),
            str(meta.get("sun_azimuth", "")),
            str(meta.get("sun_elevation", "")),
            str(meta.get("target_azimuth", "")),
            meta.get("disaster_type"),
            meta.get("catalog_id"),
            meta.get("original_width"),
            meta.get("original_height"),
            json.dumps(tile_info.get("crop_bounds", {})),
            json.dumps(buildings),
            json_path,
        ))

        tiles_loaded += 1

    conn.commit()
    print(f"  🚀 {image_folder} → {len(data.get('tiles', {}))} tiles")

print(f"\n  ✅ Tiles done: {tiles_loaded} records loaded")

# ── Load HOUSES ────────────────────────────────────────────────────────────
print("\n🏠 Loading houses...")
houses_loaded = 0

for image_folder in os.listdir(HOUSES_PATH):
    folder_path = os.path.join(HOUSES_PATH, image_folder)
    if not os.path.isdir(folder_path):
        continue

    json_path = os.path.join(folder_path, "labels.json")
    if not os.path.exists(json_path):
        print(f"  ⚠️  No labels.json in {image_folder}, skipping.")
        continue

    with open(json_path, 'r') as f:
        data = json.load(f)

    meta        = data.get("metadata", {})
    disaster_id = data.get("disasterId") or meta.get("disaster", "hurricane-florence")
    pair_id     = data.get("pairId", "unknown")
    image_uid   = image_folder  # unique per folder

    for house_id, house_info in data.get("houses", {}).items():
        house_uid    = house_id
        house_folder = os.path.join(folder_path, house_id)
        record_key   = f"{image_uid}#{house_uid}"

        # FAST INDEX
        cur.execute("""
            INSERT INTO house_index (
                image_uid, house_uid, disaster_id, pair_id, house_id,
                disaster_type, capture_date, classification, prediction,
                pre_image_path, post_image_path, target_image_path, full_record_key
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (image_uid, house_uid) DO NOTHING;
        """, (
            image_uid,
            house_uid,
            disaster_id,
            pair_id,
            house_id,
            meta.get("disaster_type"),
            meta.get("capture_date"),
            house_info.get("classification", "unknown"),
            house_info.get("prediction"),
            os.path.join(house_folder, "pre.png"),
            os.path.join(house_folder, "post.png"),
            os.path.join(house_folder, "target.png"),
            record_key,
        ))

        # FULL DATA
        cur.execute("""
            INSERT INTO house_full (
                image_uid, house_uid, disaster_id, pair_id, house_id,
                sensor, provider_asset_type, gsd, capture_date,
                off_nadir_angle, pan_resolution, sun_azimuth, sun_elevation,
                target_azimuth, disaster_type, catalog_id,
                original_width, original_height, crop_bounds, points,
                classification, prediction, json_path
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (image_uid, house_uid) DO NOTHING;
        """, (
            image_uid,
            house_uid,
            disaster_id,
            pair_id,
            house_id,
            meta.get("sensor"),
            meta.get("provider_asset_type"),
            str(meta.get("gsd", "")),
            meta.get("capture_date"),
            str(meta.get("off_nadir_angle", "")),
            str(meta.get("pan_resolution", "")),
            str(meta.get("sun_azimuth", "")),
            str(meta.get("sun_elevation", "")),
            str(meta.get("target_azimuth", "")),
            meta.get("disaster_type"),
            meta.get("catalog_id"),
            meta.get("original_width"),
            meta.get("original_height"),
            json.dumps(house_info.get("crop_bounds", {})),
            json.dumps(house_info.get("points", {})),
            house_info.get("classification", "unknown"),
            house_info.get("prediction"),
            json_path,
        ))

        houses_loaded += 1

    conn.commit()
    print(f"  🚀 {image_folder} → {len(data.get('houses', {}))} houses")

print(f"\n  ✅ Houses done: {houses_loaded} records loaded")

# ── Cleanup ────────────────────────────────────────────────────────────────
cur.close()
conn.close()
print("\n✨ All done!")