from models_2 import TileIndex, TileFull, HouseIndex, HouseFull
import os
import json

TILES_PATH  = r"C:\Users\furkm\OneDrive\Desktop\database_setup\CS4485_Hurricane_Crop_Tiles\CS4485_Hurricane_Crop_Tiles"
HOUSES_PATH = r"C:\Users\furkm\OneDrive\Desktop\database_setup\CS4485_Hurricane_Cursor_House\CS4485_Hurricane_Cursor_House"

# ── Create / reset all 4 tables ────────────────────────────────────────────
print("🔧 Setting up tables...")
tables = [(TileIndex,"tile_cursor_index"),(TileFull,"tile_cursor_full"),(HouseIndex,"house_cursor_index"),(HouseFull,"house_cursor_full")]

for ModelClass, name in tables:
    if ModelClass.exists():
        ModelClass.delete_table()
        print(f"  🗑️  Cleared: {name}")
    ModelClass.create_table(wait=True, read_capacity_units=1, write_capacity_units=1)
    print(f"  ✅ Created: {name}")

# ── Load TILES ─────────────────────────────────────────────────────────────
print("\n📦 Loading tiles...")
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
    image_uid   = image_folder

    for tile_id, tile_info in data["tiles"].items():
        tile_uid    = tile_id
        buildings   = tile_info.get("buildings", [])
        tile_folder = os.path.join(folder_path, tile_id)
        record_key  = f"{image_uid}#{tile_uid}"

        # FAST INDEX
        TileIndex(
            image_uid         = image_uid,
            tile_uid          = tile_uid,
            disaster_id       = disaster_id,
            pair_id           = pair_id,
            tile_id           = tile_id,
            disaster_type     = meta.get("disaster_type"),
            capture_date      = meta.get("capture_date"),
            building_count    = len(buildings),
            classification    = "unknown",
            prediction        = None,
            pre_image_path    = os.path.join(tile_folder, "pre.png"),
            post_image_path   = os.path.join(tile_folder, "post.png"),
            target_image_path = os.path.join(tile_folder, "target.png"),
            full_record_key   = record_key,
        ).save()

        # FULL DATA
        TileFull(
            image_uid           = image_uid,
            tile_uid            = tile_uid,
            disaster_id         = disaster_id,
            pair_id             = pair_id,
            tile_id             = tile_id,
            sensor              = meta.get("sensor"),
            provider_asset_type = meta.get("provider_asset_type"),
            gsd                 = str(meta.get("gsd", "")),
            capture_date        = meta.get("capture_date"),
            off_nadir_angle     = str(meta.get("off_nadir_angle", "")),
            pan_resolution      = str(meta.get("pan_resolution", "")),
            sun_azimuth         = str(meta.get("sun_azimuth", "")),
            sun_elevation       = str(meta.get("sun_elevation", "")),
            target_azimuth      = str(meta.get("target_azimuth", "")),
            disaster_type       = meta.get("disaster_type"),
            catalog_id          = meta.get("catalog_id"),
            original_width      = meta.get("original_width"),
            original_height     = meta.get("original_height"),
            crop_bounds         = tile_info.get("crop_bounds", {}),
            buildings           = buildings,
            json_path           = json_path,
        ).save()

        tiles_loaded += 1

    print(f"  🚀 {image_folder} → {len(data['tiles'])} tiles")

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
    image_uid   = image_folder 
    for house_id, house_info in data["houses"].items():
        house_uid    = house_id
        house_folder = os.path.join(folder_path, house_id)
        record_key   = f"{image_uid}#{house_uid}"

        # FAST INDEX
        HouseIndex(
            image_uid         = image_uid,
            house_uid         = house_uid,
            disaster_id       = disaster_id,
            pair_id           = pair_id,
            house_id          = house_id,
            disaster_type     = meta.get("disaster_type"),
            capture_date      = meta.get("capture_date"),
            classification    = "unknown",
            prediction        = None,
            pre_image_path    = os.path.join(house_folder, "pre.png"),
            post_image_path   = os.path.join(house_folder, "post.png"),
            target_image_path = os.path.join(house_folder, "target.png"),
            full_record_key   = record_key,
        ).save()

        # FULL DATA
        HouseFull(
            image_uid           = image_uid,
            house_uid           = house_uid,
            disaster_id         = disaster_id,
            pair_id             = pair_id,
            house_id            = house_id,
            sensor              = meta.get("sensor"),
            provider_asset_type = meta.get("provider_asset_type"),
            gsd                 = str(meta.get("gsd", "")),
            capture_date        = meta.get("capture_date"),
            off_nadir_angle     = str(meta.get("off_nadir_angle", "")),
            pan_resolution      = str(meta.get("pan_resolution", "")),
            sun_azimuth         = str(meta.get("sun_azimuth", "")),
            sun_elevation       = str(meta.get("sun_elevation", "")),
            target_azimuth      = str(meta.get("target_azimuth", "")),
            disaster_type       = meta.get("disaster_type"),
            catalog_id          = meta.get("catalog_id"),
            original_width      = meta.get("original_width"),
            original_height     = meta.get("original_height"),
            crop_bounds         = house_info.get("crop_bounds", {}),
            points              = house_info.get("points", {}),
            classification      = house_info.get("classification", "unknown"),
            prediction          = house_info.get("prediction"),
            json_path           = json_path,
        ).save()

        houses_loaded += 1

    print(f"  🚀 {image_folder} → {len(data['houses'])} houses")

print(f"\n  ✅ Houses done: {houses_loaded} records loaded")
print("\n✨ All done!")