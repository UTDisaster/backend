from fastapi import FastAPI, HTTPException, status
from typing import List, Optional, Dict

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

MOCK_CHATS = {
    "chat_01": {
        "id": "chat_01",
        "title": "Florence Sector 4 Damage",
        "timestamp": "2026-02-26T10:00:00Z",
        "messages": [
            {"role": "user", "content": "How many buildings are un-classified?"},
            {"role": "assistant", "content": "I found 1 un-classified building in this view."}
        ]
    },
    "chat_02": {
        "id": "chat_02",
        "title": "Evacuation Routes",
        "timestamp": "2026-02-25T14:30:00Z",
        "messages": [
            {"role": "user", "content": "Is the main road clear?"},
            {"role": "assistant", "content": "Satellite data shows minor debris on Main St."}
        ]
    }
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

@app.get("/chat/conversations")
async def list_conversations(search: Optional[str] = None):
    chat_list = list(MOCK_CHATS.values())
    
    if search:
        chat_list = [c for c in chat_list if search.lower() in c["title"].lower()]
    
    chat_list.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return [{"id": c["id"], "title": c["title"], "timestamp": c["timestamp"]} for c in chat_list]

@app.get("/chat/conversations/{chat_id}")
async def get_chat(chat_id: str):
    chat = MOCK_CHATS.get(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat

