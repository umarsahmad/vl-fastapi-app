FROM python:3.10-slim

WORKDIR /app

# Install git, git-lfs, AND the libraries required for OpenCV
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
# Use the CPU-only index for torch as discussed before to save space
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL your project files (app.py, utils1.py, utils2.py, models/)
COPY . .

# Expose port (Render usually uses 10000, but 8000 is fine if configured)
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]