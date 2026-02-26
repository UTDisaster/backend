from fastapi import FastAPI, HTTPException, status

app = FastAPI()

MOCK_LOCATIONS = {
    "disaster-123": [
        {
            "image_id": "IMG_001",
            "image_url": "https://storage.aws.com/disaster/drone_shot_north.jpg",
            "houses": [
                {"house_id": "H1", "lat": 34.0522, "lng": -118.2437, "damage": "Damaged"},
                {"house_id": "H2", "lat": 34.0525, "lng": -118.2440, "damage": "Minor Damage"}
            ]
        },
        {
            "image_id": "IMG_002",
            "image_url": "https://storage.aws.com/disaster/drone_shot_south.jpg",
            "houses": [
                {"house_id": "H3", "lat": 34.0610, "lng": -118.2510, "damage": "Moderate"},
                {"house_id": "H4", "lat": 34.0615, "lng": -118.2520, "damage": "Safe"}
            ]
        }
    ]
}

@app.get("/")
async def root():
    return {"message": "UTD Disaster Assessment Project"}

@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    return {"status": "OK", "status_code": status.HTTP_200_OK}

@app.get("/location/{disaster_id}")
async def get_location(disaster_id: str):
    disaster_data = MOCK_LOCATIONS.get(disaster_id)
    
    if not disaster_data:
        raise HTTPException(status_code=404, detail="Disaster not found")

    features = []
    for img in disaster_data:
        for house in img["houses"]:
            features.append({
                "geometry": {
                    "type": "Point",
                    "coordinates": [house["lng"], house["lat"]]
                },
                "properties": {
                    "house_id": house["house_id"],
                    "parent_image": img["image_id"],
                    "image_url": img["image_url"],
                    "damage_level": house["damage"]
                }
            })

    return {"features": features}



