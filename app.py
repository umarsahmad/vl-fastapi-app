from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from utils import apply_lora_to_model
from slm_arc import GPT, GPTConfig
from datetime import datetime
import os
import cv2
import numpy as np
from ultralytics import YOLO
import json
from io import BytesIO
import asyncio
import torch
import tiktoken


app = FastAPI(title="Simple FastAPI Test")

# Serve HTML pages
@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FastAPI App - Home</title>
    </head>
    <body>
        <h1>Welcome to FastAPI App</h1>
        <p>Status: Running</p>
        <p>Timestamp: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        
        <h2>Available Pages:</h2>
        <ul>
            <li><a href="/defect_gen">Defect Detection Page</a></li>
            <li><a href="/llm_page">LLM Processing Page</a></li>
            <li><a href="/health">Health Check (JSON)</a></li>
        </ul>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Add after your home() function
@app.head("/")
async def head_home():
    return {}

@app.head("/health")
async def head_health():
    return {}

@app.get("/defect_gen", response_class=HTMLResponse)
def defect_gen_page():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Defect Detection</title>
    </head>
    <body>
        <h1>Defect Detection</h1>
        <p><a href="/">← Back to Home</a></p>
        
        <h2>Upload Image for Detection</h2>
        <form id="uploadForm">
            <input type="file" id="imageInput" accept="image/*" required>
            <br><br>
            <button type="submit">Submit</button>
        </form>
        
        <br>
        <div id="result" style="display:none;">
            <h2>Results:</h2>
            <div style="display: flex; gap: 20px;">
                <div>
                    <h3>Original Image:</h3>
                    <img id="originalImage" style="max-width: 400px; border: 1px solid black;">
                </div>
                <div>
                    <h3>Processed Image:</h3>
                    <img id="processedImage" style="max-width: 400px; border: 1px solid black;">
                </div>
            </div>
        </div>
        
        <div id="loading" style="display:none;">
            <p>Processing image...</p>
        </div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const fileInput = document.getElementById('imageInput');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select an image');
                    return;
                }
                
                // Show original image
                const reader = new FileReader();
                reader.onload = (e) => {
                    document.getElementById('originalImage').src = e.target.result;
                };
                reader.readAsDataURL(file);
                
                // Show loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                
                // Upload image
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/process_image', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        const blob = await response.blob();
                        const imageUrl = URL.createObjectURL(blob);
                        document.getElementById('processedImage').src = imageUrl;
                        document.getElementById('result').style.display = 'block';
                    } else {
                        alert('Error processing image');
                    }
                } catch (error) {
                    alert('Error: ' + error.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/llm_page", response_class=HTMLResponse)
def llm_page():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM Processing</title>
    </head>
    <body>
        <h1>LLM Text Processing</h1>
        <p><a href="/">← Back to Home</a></p>
        
        <h2>Enter Text for Processing</h2>
        <form id="textForm">
            <textarea id="textInput" rows="5" cols="50" placeholder="Enter your text here..." required></textarea>
            <br><br>
            <button type="submit">Submit</button>
        </form>
        
        <br>
        <div id="result" style="display:none;">
            <h2>Processed Result:</h2>
            <div style="border: 1px solid black; padding: 10px; background-color: #f0f0f0;">
                <p id="resultText"></p>
            </div>
        </div>
        
        <div id="loading" style="display:none;">
            <p>Processing text...</p>
        </div>

        <script>
            document.getElementById('textForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const textInput = document.getElementById('textInput').value;
                
                if (!textInput.trim()) {
                    alert('Please enter some text');
                    return;
                }
                
                // Show loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                
                try {
                    const response = await fetch('/process_text', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text: textInput })
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        document.getElementById('resultText').textContent = data.processed_text;
                        document.getElementById('result').style.display = 'block';
                    } else {
                        alert('Error processing text');
                    }
                } catch (error) {
                    alert('Error: ' + error.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

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



def load_lora_weights(model, lora_weights_path, device='cpu'):
    """Load LoRA weights into the model"""
    lora_state_dict = torch.load(lora_weights_path, map_location=device)  # Add map_location
    
    # Load only LoRA parameters
    model_state_dict = model.state_dict()
    model_state_dict.update(lora_state_dict)
    model.load_state_dict(model_state_dict)
    
    print(f"Loaded LoRA weights from {lora_weights_path}")
    return model

def generate_answer(model, enc, question, max_tokens=20, device='cpu'):
    """Generate answer for a question"""
    
    # Format the prompt (same format as training)
    prompt = f"Question: {question}\nAnswer:"
    
    # Tokenize
    tokens = enc.encode_ordinary(prompt)
    context = torch.tensor(tokens).unsqueeze(0).to(device)
    
    # Generate
    model.eval()
    with torch.no_grad():
        output = model.generate(context, max_tokens)
    
    # Decode
    generated_text = enc.decode(output.squeeze().tolist())
    
    # Extract only the answer part
    answer_start = generated_text.find("Answer:") + len("Answer:")
    answer = generated_text[answer_start:].strip()
    
    return answer

def initialize_model():
    """Initialize model once at startup"""
    global model, enc, device
    
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Load tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Load base model
    config = GPTConfig(
        vocab_size=50257,
        block_size=128,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.1,
        bias=True
    )
    model = GPT(config)
    model.load_state_dict(torch.load('models/best_model_params.pt', map_location=device))
    
    # Apply LoRA architecture
    model = apply_lora_to_model(model, rank=8, alpha=16)
    
    # Load LoRA weights
    model = load_lora_weights(model, 'lora_weights.pt', device=device)
    model = model.to(device)
    model.eval()
    
    print("Model initialized successfully!")


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
        # Get max_tokens from request or use default
        # max_tokens = data.get("max_tokens", 100)
        
        # Generate answer using the model
        # processed_text = generate_answer(model, enc, input_text, max_tokens=max_tokens, device=device)
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




