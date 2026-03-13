from models import TileIndex, TileFull, HouseIndex, HouseFull

# ──────────────────────────────────────────────────────────────────────────────
# TWO-TABLE SYSTEM
# Query the INDEX table (fast, lightweight)
# Only fetch FULL table if you need WKT / polygons / full metadata
# ──────────────────────────────────────────────────────────────────────────────


#Get all tiles for one image (INDEX only) 
print("=== All tiles for one image ===")
image_id = "hurricane-florence_00000004_post_disaster"

for tile in TileIndex.query(image_id):
    print(f"  {tile.tile_id} | buildings: {tile.building_count} | damage: {tile.max_damage_level}")
    print(f"    post → {tile.post_image_path}")


# Get all destroyed tiles across ALL images 
print("\n=== All destroyed tiles ===")
for tile in TileIndex.scan(TileIndex.max_damage_level == "destroyed"):
    print(f"  {tile.image_id} / {tile.tile_id} → {tile.post_image_path}")


# Get full polygon data for a specific tile 
print("\n=== Full data for one tile ===")
full = TileFull.get("hurricane-florence_00000004_post_disaster", "tile_001")
print(f"  Sensor: {full.sensor}")
print(f"  Crop bounds: {full.crop_bounds}")
print(f"  Buildings: {len(full.buildings)} found")
for b in full.buildings:
    print(f"    → damage: {b['damage_level']} | wkt: {b['local_wkt'][:40]}...")


# Use INDEX pointer to load FULL data 
print("\n=== Index → Full data lookup ===")
tile = TileIndex.get("hurricane-florence_00000004_post_disaster", "tile_002")
print(f"  Index record: {tile.tile_id}, damage={tile.max_damage_level}")

# Use the full_record_key pointer to fetch full record
img_id, tile_id = tile.full_record_key.split("#")
full = TileFull.get(img_id, tile_id)
print(f"  Full record has {len(full.buildings)} buildings")
print(f"  GSD: {full.gsd}, Sensor: {full.sensor}")


#  Training loop pattern 
print("\n=== Training loop example ===")
from PIL import Image  # pip install Pillow

for tile in TileIndex.scan(TileIndex.disaster_type == "flooding"):
    # Load the actual images using the paths stored in the index
    post_img = Image.open(tile.post_image_path)
    pre_img  = Image.open(tile.pre_image_path)
    label    = tile.max_damage_level

    print(f"  Loaded {tile.tile_id}: {post_img.size}, label={label}")

    # Only fetch full data if you need polygon details
    if label == "destroyed":
        full = TileFull.get(tile.image_id, tile.tile_id)
        print(f"    → {len(full.buildings)} building polygons available")


#Example 6: Houses queries
print("\n=== All un-classified houses ===")
for house in HouseIndex.scan(HouseIndex.damage_level == "un-classified"):
    print(f"  {house.image_id} / {house.house_id} → {house.post_image_path}")