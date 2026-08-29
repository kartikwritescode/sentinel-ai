# Production Dockerfile for Smart CCTV Suspicious Behavior Detection System
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies required by OpenCV, FFmpeg, and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU PyTorch and torchvision first for lightweight container size
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source files
COPY config.py .
COPY app.py .
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create data and models directories
RUN mkdir -p data/evidence_clips data/uploads models

# Expose FastAPI server port
EXPOSE 8000

# Run FastAPI backend with Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
