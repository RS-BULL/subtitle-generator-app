FROM python:3.11-slim

WORKDIR /app

# Install system deps
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

# Download ggml-tiny.bin directly from Hugging Face (75 MB)
RUN wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin -O /app/whisper_model/ggml-tiny.bin

# Install Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code (no need to include model in Git)
COPY . .

# Ensure whisper_model dir exists
RUN mkdir -p /app/whisper_model

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]