import os
import json
from models import TileIndex, TileFull, HouseIndex, HouseFull

# ✏️ UPDATE THESE to your actual folder paths
TILES_PATH  = r"C:\Users\furkm\OneDrive\Desktop\database_setup\CS4485_Hurricane_Grid_Final\CS4485_Hurricane_Grid_Final"
HOUSES_PATH = r"C:\Users\furkm\OneDrive\Desktop\database_setup\CS4485_Hurricane_Org_final\CS4485_Hurricane_Org_final"




# ── Create all 4 tables ────────────────────────────────────────────────────
print("🔧 Setting up tables...")
tables = [
    (TileIndex,  "TileIndex"),
    (TileFull,   "TileFull"),
    (HouseIndex, "HouseIndex"),
    (HouseFull,  "HouseFull"),
]
for ModelClass, name in tables:
    if not ModelClass.exists():
        ModelClass.create_table(wait=True, read_capacity_units=1, write_capacity_units=1)
        print(f"  ✅ Created: {name}")
    else:
        print(f"  ✅ Already exists: {name}")


# ── Load ALL TILES ─────────────────────────────────────────────────────────
print("\n📦 Loading tiles dataset...")
tiles_loaded  = 0
tiles_skipped = 0

for image_folder in os.listdir(TILES_PATH):
    image_folder_path = os.path.join(TILES_PATH, image_folder)

    if not os.path.isdir(image_folder_path):
        continue

    json_path = os.path.join(image_folder_path, "master_grid_labels.json")

    if not os.path.exists(json_path):
        print(f"  ⚠️  No master_grid_labels.json in {image_folder}, skipping.")
        tiles_skipped += 1
        continue

    with open(json_path, 'r') as f:
        data = json.load(f)

    image_id = data["source_image"].replace(".png", "")
    meta     = data.get("metadata", {})

    for tile_id, tile_info in data["tiles"].items():
        buildings   = tile_info.get("buildings", [])
        tile_folder = os.path.join(image_folder_path, tile_id)
        record_key  = f"{image_id}#{tile_id}"

        # ── Save to FAST INDEX (lightweight) ──────────────────────────────
        TileIndex(
            image_id          = image_id,
            tile_id           = tile_id,
            disaster          = meta.get("disaster"),
            disaster_type     = meta.get("disaster_type"),
            capture_date      = meta.get("capture_date"),
            building_count    = len(buildings),
            max_damage_level  = "pending",
            pre_image_path    = os.path.join(tile_folder, "pre.png"),
            post_image_path   = os.path.join(tile_folder, "post.png"),
            target_image_path = os.path.join(tile_folder, "target.png"),
            full_record_key   = record_key,
        ).save()

        # ── Save to FULL DATA (everything) ────────────────────────────────
        TileFull(
            image_id             = image_id,
            tile_id              = tile_id,
            sensor               = meta.get("sensor"),
            provider_asset_type  = meta.get("provider_asset_type"),
            gsd                  = str(meta.get("gsd", "")),
            capture_date         = meta.get("capture_date"),
            off_nadir_angle      = str(meta.get("off_nadir_angle", "")),
            pan_resolution       = str(meta.get("pan_resolution", "")),
            sun_azimuth          = str(meta.get("sun_azimuth", "")),
            sun_elevation        = str(meta.get("sun_elevation", "")),
            target_azimuth       = str(meta.get("target_azimuth", "")),
            disaster             = meta.get("disaster"),
            disaster_type        = meta.get("disaster_type"),
            catalog_id           = meta.get("catalog_id"),
            original_width       = meta.get("original_width"),
            original_height      = meta.get("original_height"),
            crop_bounds          = tile_info.get("crop_bounds", {}),
            buildings            = buildings,
            json_path            = json_path,
        ).save()

        tiles_loaded += 1

    print(f"  🚀 {image_folder} → {len(data['tiles'])} tiles")

print(f"\n  ✅ Tiles done: {tiles_loaded} records loaded, {tiles_skipped} skipped")


# ── Load ALL HOUSES ────────────────────────────────────────────────────────
print("\n🏠 Loading houses dataset...")
houses_loaded  = 0
houses_skipped = 0

for image_folder in os.listdir(HOUSES_PATH):
    image_folder_path = os.path.join(HOUSES_PATH, image_folder)

    if not os.path.isdir(image_folder_path):
        continue

    json_path = os.path.join(image_folder_path, "labels.json")

    if not os.path.exists(json_path):
        print(f"  ⚠️  No labels.json in {image_folder}, skipping.")
        houses_skipped += 1
        continue

    with open(json_path, 'r') as f:
        data = json.load(f)

    image_id = data["source_image"].replace(".png", "")
    meta     = data.get("metadata", {})

    for house_id, house_info in data["houses"].items():
        house_folder = os.path.join(image_folder_path, house_id)
        record_key   = f"{image_id}#{house_id}"

        # ── Save to FAST INDEX (lightweight) ──────────────────────────────
        HouseIndex(
            image_id          = image_id,
            house_id          = house_id,
            disaster          = meta.get("disaster"),
            disaster_type     = meta.get("disaster_type"),
            capture_date      = meta.get("capture_date"),
            damage_level      = "pending",
            pre_image_path    = os.path.join(house_folder, "pre.png"),
            post_image_path   = os.path.join(house_folder, "post.png"),
            target_image_path = os.path.join(house_folder, "target.png"),
            full_record_key   = record_key,
        ).save()

        # ── Save to FULL DATA (everything) ────────────────────────────────
        HouseFull(
            image_id             = image_id,
            house_id             = house_id,
            sensor               = meta.get("sensor"),
            provider_asset_type  = meta.get("provider_asset_type"),
            gsd                  = str(meta.get("gsd", "")),
            capture_date         = meta.get("capture_date"),
            off_nadir_angle      = str(meta.get("off_nadir_angle", "")),
            pan_resolution       = str(meta.get("pan_resolution", "")),
            sun_azimuth          = str(meta.get("sun_azimuth", "")),
            sun_elevation        = str(meta.get("sun_elevation", "")),
            target_azimuth       = str(meta.get("target_azimuth", "")),
            disaster             = meta.get("disaster"),
            disaster_type        = meta.get("disaster_type"),
            catalog_id           = meta.get("catalog_id"),
            original_width       = meta.get("original_width"),
            original_height      = meta.get("original_height"),
            local_wkt            = house_info.get("local_wkt"),
            original_wkt         = house_info.get("original_wkt"),
            crop_bounds          = house_info.get("crop_bounds", {}),
            damage_level         = house_info.get("damage_level"),
            json_path            = json_path,
        ).save()

        houses_loaded += 1

    print(f"  🚀 {image_folder} → {len(data['houses'])} houses")

print(f"\n  ✅ Houses done: {houses_loaded} records loaded, {houses_skipped} skipped")

print("\n✨ All done! Both index and full tables are loaded.")