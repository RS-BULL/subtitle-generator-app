FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone and build whisper.cpp
RUN git clone https://github.com/ggerganov/whisper.cpp.git
WORKDIR /app/whisper.cpp
RUN make -j

# Create model directory and download ggml-tiny.bin
RUN mkdir -p /app/whisper_model
RUN wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin -O /app/whisper_model/ggml-tiny.bin

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ONLY the app code (no whisper_model from local!)
COPY main.py .

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]