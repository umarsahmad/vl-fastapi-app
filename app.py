from fastapi import FastAPI
from datetime import datetime
import os

app = FastAPI(title="Simple FastAPI Test")

@app.get("/")
def home():
    return {
        "message": "Hello from FastAPI!",
        "status": "running",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/health")
def health():
    # Check if model file exists
    model_path = "models/best.pt"
    model_exists = os.path.exists(model_path)
    model_size = os.path.getsize(model_path) if model_exists else 0
    
    return {
        "status": "healthy",
        "service": "fastapi-render-test",
        "model_loaded": model_exists,
        "model_size_mb": round(model_size / (1024 * 1024), 2) if model_exists else 0
    }

@app.get("/test")
def test():
    return {
        "test": "successful",
        "data": [1, 2, 3, 4, 5]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)