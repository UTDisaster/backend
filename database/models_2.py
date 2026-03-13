from pynamodb.models import Model
from pynamodb.attributes import UnicodeAttribute, NumberAttribute, JSONAttribute

class TileIndex(Model):
    class Meta:
        table_name            = "tile_cursor_index"
        host                  = "http://localhost:8000"
        region                = "us-west-2"
        aws_access_key_id     = "fake"
        aws_secret_access_key = "fake"

    image_uid         = UnicodeAttribute(hash_key=True)
    tile_uid          = UnicodeAttribute(range_key=True)
    disaster_id       = UnicodeAttribute(null=True)
    pair_id           = UnicodeAttribute(null=True)
    tile_id           = UnicodeAttribute(null=True)
    disaster_type     = UnicodeAttribute(null=True)
    capture_date      = UnicodeAttribute(null=True)
    building_count    = NumberAttribute(default=0)
    classification    = UnicodeAttribute(null=True)
    prediction        = UnicodeAttribute(null=True)
    pre_image_path    = UnicodeAttribute(null=True)
    post_image_path   = UnicodeAttribute(null=True)
    target_image_path = UnicodeAttribute(null=True)
    full_record_key   = UnicodeAttribute(null=True)

class HouseIndex(Model):
    class Meta:
        table_name            = "house_cursor_index"
        host                  = "http://localhost:8000"
        region                = "us-west-2"
        aws_access_key_id     = "fake"
        aws_secret_access_key = "fake"

    image_uid         = UnicodeAttribute(hash_key=True)
    house_uid         = UnicodeAttribute(range_key=True)
    disaster_id       = UnicodeAttribute(null=True)
    pair_id           = UnicodeAttribute(null=True)
    house_id          = UnicodeAttribute(null=True)
    disaster_type     = UnicodeAttribute(null=True)
    capture_date      = UnicodeAttribute(null=True)
    classification    = UnicodeAttribute(null=True)
    prediction        = UnicodeAttribute(null=True)
    pre_image_path    = UnicodeAttribute(null=True)
    post_image_path   = UnicodeAttribute(null=True)
    target_image_path = UnicodeAttribute(null=True)
    full_record_key   = UnicodeAttribute(null=True)

class TileFull(Model):
    class Meta:
        table_name            = "tile_cursor_full"
        host                  = "http://localhost:8000"
        region                = "us-west-2"
        aws_access_key_id     = "fake"
        aws_secret_access_key = "fake"

    image_uid           = UnicodeAttribute(hash_key=True)
    tile_uid            = UnicodeAttribute(range_key=True)
    disaster_id         = UnicodeAttribute(null=True)
    pair_id             = UnicodeAttribute(null=True)
    tile_id             = UnicodeAttribute(null=True)
    sensor              = UnicodeAttribute(null=True)
    provider_asset_type = UnicodeAttribute(null=True)
    gsd                 = UnicodeAttribute(null=True)
    capture_date        = UnicodeAttribute(null=True)
    off_nadir_angle     = UnicodeAttribute(null=True)
    pan_resolution      = UnicodeAttribute(null=True)
    sun_azimuth         = UnicodeAttribute(null=True)
    sun_elevation       = UnicodeAttribute(null=True)
    target_azimuth      = UnicodeAttribute(null=True)
    disaster_type       = UnicodeAttribute(null=True)
    catalog_id          = UnicodeAttribute(null=True)
    original_width      = NumberAttribute(null=True)
    original_height     = NumberAttribute(null=True)
    crop_bounds         = JSONAttribute()
    buildings           = JSONAttribute()
    json_path           = UnicodeAttribute(null=True)

class HouseFull(Model):
    class Meta:
        table_name            = "house_cursor_full"
        host                  = "http://localhost:8000"
        region                = "us-west-2"
        aws_access_key_id     = "fake"
        aws_secret_access_key = "fake"

    image_uid           = UnicodeAttribute(hash_key=True)
    house_uid           = UnicodeAttribute(range_key=True)
    disaster_id         = UnicodeAttribute(null=True)
    pair_id             = UnicodeAttribute(null=True)
    house_id            = UnicodeAttribute(null=True)
    sensor              = UnicodeAttribute(null=True)
    provider_asset_type = UnicodeAttribute(null=True)
    gsd                 = UnicodeAttribute(null=True)
    capture_date        = UnicodeAttribute(null=True)
    off_nadir_angle     = UnicodeAttribute(null=True)
    pan_resolution      = UnicodeAttribute(null=True)
    sun_azimuth         = UnicodeAttribute(null=True)
    sun_elevation       = UnicodeAttribute(null=True)
    target_azimuth      = UnicodeAttribute(null=True)
    disaster_type       = UnicodeAttribute(null=True)
    catalog_id          = UnicodeAttribute(null=True)
    original_width      = NumberAttribute(null=True)
    original_height     = NumberAttribute(null=True)
    crop_bounds         = JSONAttribute()
    points              = JSONAttribute()
    classification      = UnicodeAttribute(null=True)
    prediction          = UnicodeAttribute(null=True)
    json_path           = UnicodeAttribute(null=True)