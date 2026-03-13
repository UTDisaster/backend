from models import TileIndex, TileFull, HouseIndex, HouseFull

print("=== TABLE RECORD COUNTS ===")
tile_index_count  = sum(1 for _ in TileIndex.scan())
tile_full_count   = sum(1 for _ in TileFull.scan())
house_index_count = sum(1 for _ in HouseIndex.scan())
house_full_count  = sum(1 for _ in HouseFull.scan())

print(f"  TileIndex  : {tile_index_count} records")
print(f"  TileFull   : {tile_full_count} records")
print(f"  HouseIndex : {house_index_count} records")
print(f"  HouseFull  : {house_full_count} records")

print("\n=== SAMPLE TILE INDEX RECORD ===")
sample_tile = next(TileIndex.scan())
print(f"  image_id        : {sample_tile.image_id}")
print(f"  tile_id         : {sample_tile.tile_id}")
print(f"  disaster        : {sample_tile.disaster}")
print(f"  building_count  : {sample_tile.building_count}")
print(f"  damage_level    : {sample_tile.max_damage_level}")
print(f"  post_image_path : {sample_tile.post_image_path}")
print(f"  full_record_key : {sample_tile.full_record_key}")

print("\n=== SAMPLE HOUSE INDEX RECORD ===")
sample_house = next(HouseIndex.scan())
print(f"  image_id        : {sample_house.image_id}")
print(f"  house_id        : {sample_house.house_id}")
print(f"  damage_level    : {sample_house.damage_level}")
print(f"  post_image_path : {sample_house.post_image_path}")

print("\n✅ All tables verified!")