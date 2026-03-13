from pynamodb.models import Model
from pynamodb.attributes import UnicodeAttribute, NumberAttribute, JSONAttribute


# FAST INDEX TABLES

# Only holds labels, counts, damage levels, and file paths


class TileIndex(Model):
    """
    Fast index for tiles dataset.
    Query this to find tiles by damage level, disaster type, etc.
    Use tile.full_record_key to fetch full data from TileFull if needed.
    """
    class Meta:
        table_name  = "TileIndex"
        host        = "http://localhost:8000"
        region      = "us-west-2"
        aws_access_key_id     = "fake"
        aws_secret_access_key = "fake"

    # Keys
    image_id         = UnicodeAttribute(hash_key=True)   
    tile_id          = UnicodeAttribute(range_key=True)  

    # Quick-filter labels
    disaster         = UnicodeAttribute(null=True)        
    disaster_type    = UnicodeAttribute(null=True)        
    capture_date     = UnicodeAttribute(null=True)
    building_count   = NumberAttribute(default=0)
    max_damage_level = UnicodeAttribute(null=True)        

    # File paths 
    pre_image_path    = UnicodeAttribute(null=True)
    post_image_path   = UnicodeAttribute(null=True)
    target_image_path = UnicodeAttribute(null=True)

    # Pointer to full data record
    full_record_key  = UnicodeAttribute(null=True)       


class HouseIndex(Model):
    """
    Fast index for houses dataset.
    Query this to find houses by damage level, disaster type, etc.
    Use house.full_record_key to fetch full data from HouseFull if needed.
    """
    class Meta:
        table_name  = "HouseIndex"
        host        = "http://localhost:8000"
        region      = "us-west-2"
        aws_access_key_id     = "fake"
        aws_secret_access_key = "fake"

    # Keys
    image_id         = UnicodeAttribute(hash_key=True)
    house_id         = UnicodeAttribute(range_key=True)  

    # Quick-filter labels
    disaster         = UnicodeAttribute(null=True)
    disaster_type    = UnicodeAttribute(null=True)
    capture_date     = UnicodeAttribute(null=True)
    damage_level     = UnicodeAttribute(null=True)

    # File paths
    pre_image_path    = UnicodeAttribute(null=True)
    post_image_path   = UnicodeAttribute(null=True)
    target_image_path = UnicodeAttribute(null=True)

    # Pointer to full data record
    full_record_key  = UnicodeAttribute(null=True)        



# FULL DATA TABLES
# Heavy records with all WKT polygons, buildings arrays, and complete metadata
# Only load these when you actually need the polygon/geometry details


class TileFull(Model):
    """
    Full data record for a tile.
    Contains all building polygons, WKT geometry, crop bounds, and complete metadata.
    Access via: TileFull.get(image_id, tile_id)
    """
    class Meta:
        table_name  = "TileFull"
        host        = "http://localhost:8000"
        region      = "us-west-2"
        aws_access_key_id     = "fake"
        aws_secret_access_key = "fake"

    # Keys (same as TileIndex so you can cross-reference easily)
    image_id  = UnicodeAttribute(hash_key=True)
    tile_id   = UnicodeAttribute(range_key=True)

    # Complete metadata from JSON
    sensor               = UnicodeAttribute(null=True)
    provider_asset_type  = UnicodeAttribute(null=True)
    gsd                  = UnicodeAttribute(null=True)   
    capture_date         = UnicodeAttribute(null=True)
    off_nadir_angle      = UnicodeAttribute(null=True)
    pan_resolution       = UnicodeAttribute(null=True)
    sun_azimuth          = UnicodeAttribute(null=True)
    sun_elevation        = UnicodeAttribute(null=True)
    target_azimuth       = UnicodeAttribute(null=True)
    disaster             = UnicodeAttribute(null=True)
    disaster_type        = UnicodeAttribute(null=True)
    catalog_id           = UnicodeAttribute(null=True)
    original_width       = NumberAttribute(null=True)
    original_height      = NumberAttribute(null=True)

    # Tile geometry
    crop_bounds = JSONAttribute()   # {x1, y1, x2, y2}
    buildings   = JSONAttribute()   # full list of {local_wkt, damage_level}

    # Path to raw JSON on disk (load this if you need anything else)
    json_path   = UnicodeAttribute(null=True)


class HouseFull(Model):
    """
    Full data record for a house crop.
    Contains WKT polygons, crop bounds, and complete metadata.
    Access via: HouseFull.get(image_id, house_id)
    """
    class Meta:
        table_name  = "HouseFull"
        host        = "http://localhost:8000"
        region      = "us-west-2"
        aws_access_key_id     = "fake"
        aws_secret_access_key = "fake"

    # Keys
    image_id  = UnicodeAttribute(hash_key=True)
    house_id  = UnicodeAttribute(range_key=True)

    # Complete metadata
    sensor               = UnicodeAttribute(null=True)
    provider_asset_type  = UnicodeAttribute(null=True)
    gsd                  = UnicodeAttribute(null=True)
    capture_date         = UnicodeAttribute(null=True)
    off_nadir_angle      = UnicodeAttribute(null=True)
    pan_resolution       = UnicodeAttribute(null=True)
    sun_azimuth          = UnicodeAttribute(null=True)
    sun_elevation        = UnicodeAttribute(null=True)
    target_azimuth       = UnicodeAttribute(null=True)
    disaster             = UnicodeAttribute(null=True)
    disaster_type        = UnicodeAttribute(null=True)
    catalog_id           = UnicodeAttribute(null=True)
    original_width       = NumberAttribute(null=True)
    original_height      = NumberAttribute(null=True)

    # House geometry
    local_wkt    = UnicodeAttribute(null=True)
    original_wkt = UnicodeAttribute(null=True)
    crop_bounds  = JSONAttribute()
    damage_level = UnicodeAttribute(null=True)

    # Path to raw JSON on disk
    json_path    = UnicodeAttribute(null=True)