FROM python:3.11-slim

WORKDIR /app

# Install runtime deps (no build tools needed!)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download prebuilt whisper.cpp binary for Linux (x86_64)
RUN wget https://github.com/ggerganov/whisper.cpp/releases/latest/download/whisper-linux-x64-bin.tar.gz
RUN tar -xzf whisper-linux-x64-bin.tar.gz
RUN mv whisper-linux-x64-bin/main whisper.cpp-main
RUN chmod +x whisper.cpp-main

# Download ggml-tiny.bin model
RUN mkdir -p /app/whisper_model
RUN wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin -O /app/whisper_model/ggml-tiny.bin

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]