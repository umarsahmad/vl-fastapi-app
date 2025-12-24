from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from datetime import datetime
import os
from io import BytesIO

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
    """
    Process uploaded image and return modified image
    For now, just returns the same image as placeholder
    """
    # Read the uploaded image
    image_bytes = await file.read()
    
    # TODO: Add your YOLO model processing here
    # For now, just return the same image
    processed_image = image_bytes
    
    return StreamingResponse(BytesIO(processed_image), media_type="image/jpeg")

@app.post("/process_text")
async def process_text(data: dict):
    """
    Process text input and return processed result
    For now, just returns modified text as placeholder
    """
    input_text = data.get("text", "")
    
    # TODO: Add your LLM processing here
    # For now, just return a simple processed version
    processed_text = f"Processed: {input_text.upper()} [Length: {len(input_text)} characters]"
    
    return {
        "processed_text": processed_text,
        "original_length": len(input_text),
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