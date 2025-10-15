FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Clone and build whisper.cpp
RUN git clone https://github.com/ggerganov/whisper.cpp.git
WORKDIR /app/whisper.cpp
RUN make -j && make tiny

# Copy your tiny model (from your local machine)
COPY whisper_model/ggml-tiny.bin /app/whisper_model/

# Install Python
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port (Render uses 10000)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]