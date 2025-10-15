FROM python:3.11-slim

WORKDIR /app

# Install build tools and ffmpeg
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone whisper.cpp
RUN git clone https://github.com/ggerganov/whisper.cpp.git
WORKDIR /app/whisper.cpp

# Build only the 'tiny' model support (minimal memory)
RUN make -j1 CC=gcc CXX=g++ LDFLAGS="-s" WHISPER_NOCOMPUTE=1

# Download ggml-tiny.bin model
RUN mkdir -p /app/whisper_model
RUN wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin -O /app/whisper_model/ggml-tiny.bin

# Install Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY backend/main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]