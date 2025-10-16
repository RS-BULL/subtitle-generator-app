FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (order matters for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY backend/main.py .

# Reduce torch CPU threads (saves RAM)
ENV OMP_NUM_THREADS=1
ENV TORCH_NUM_THREADS=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]