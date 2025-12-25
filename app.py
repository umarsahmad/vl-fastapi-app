from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from utils import initialize_model, generate_answer
from datetime import datetime
import os
import cv2
import numpy as np
from ultralytics import YOLO
import json
from io import BytesIO
import asyncio


app = FastAPI(title="Simple FastAPI Test")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Serve HTML pages
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )

# Add after your home() function
@app.head("/")
async def head_home():
    return {}

@app.head("/health")
async def head_health():
    return {}

@app.get("/defect_gen", response_class=HTMLResponse)
def defect_gen_page(request: Request):
    return templates.TemplateResponse("defect_gen.html", {"request": request})

@app.get("/llm_page", response_class=HTMLResponse)
def llm_page(request: Request):
    return templates.TemplateResponse("llm_page.html", {"request": request})

# API endpoints for processing
@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    # Load model inside the endpoint
    try:
        model = YOLO('models/best.pt')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    # Read uploaded image
    image_bytes = await file.read()
    
    # Convert to OpenCV format
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    try:
        # Run YOLO inference
        results = model.predict(source=img, conf=0.50, verbose=False)
        
        # Annotate image
        annotated_img = results[0].plot()
        
        # Encode to JPEG
        success, encoded_image = cv2.imencode('.jpg', annotated_img)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image")
        
        # Return as streaming response
        return StreamingResponse(BytesIO(encoded_image), media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# Global variables for model and tokenizer
model = None
enc = None
device = None


@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    try:
        initialize_model()
    except Exception as e:
        print(f"Warning: Could not initialize LLM model: {e}")
        print("LLM endpoints will not work, but app will continue running")


@app.post("/process_text")
async def process_text(data: dict):
    """
    Process text input using the fine-tuned model
    """
    global model, enc, device
    
    if model is None or enc is None:
        raise HTTPException(status_code=503, detail="LLM model not initialized. Check logs.")
    
    input_text = data.get("text", "")
    max_tokens = min(data.get("max_tokens", 20), 20)  # Cap at 20 tokens
    
    if not input_text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    try:
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        processed_text = await loop.run_in_executor(
            None, 
            generate_answer, 
            model, enc, input_text, max_tokens, device
        )
        
        return {
            "processed_text": processed_text,
            "original_text": input_text,
            "original_length": len(input_text),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")


@app.get("/health")
def health():
    # Check if model file exists
    model_path = "models/best.pt"
    model_exists = os.path.exists(model_path)
    model_size = os.path.getsize(model_path) if model_exists else 0
    
    return {
        "status": "healthy",
        "service": "vl-fastapi-app",
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