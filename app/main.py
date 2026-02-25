from fastapi import FastAPI, HTTPException, status

app = FastAPI()

MOCK_LOCATIONS = {
    "disaster-123": [
        {
            "id": "p1",
            "lat": 34.0522,
            "lng": -118.2437,
            "damage": "High",
            "img": "thumb_01.jpg",
        },
        {
            "id": "p2",
            "lat": 34.0622,
            "lng": -118.2537,
            "damage": "Medium",
            "img": "thumb_02.jpg",
        },
    ]
}


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    return {"status": "OK", "status_code": status.HTTP_200_OK}


@app.get("/location/{disaster_id}")
async def get_location(disaster_id: str):
    data = MOCK_LOCATIONS.get(disaster_id)
    if data:
        return data

    raise HTTPException(status_code=404, detail="Disaster not found")
