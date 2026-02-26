from fastapi import FastAPI, HTTPException, status

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    return {"status": "OK", "status_code": status.HTTP_200_OK}



