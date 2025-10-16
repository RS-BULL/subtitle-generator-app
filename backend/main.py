import os
import tempfile
import subprocess
import numpy as np
import librosa
from transformers import pipeline
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# Allow frontend (Live Server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup (Render keeps instance warm)
print("Loading Distil-Whisper model...")
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="distil-whisper/distil-large-v2",
    torch_dtype=np.float32,
    device="cpu",
    chunk_length_s=30,
    return_timestamps=True
)
print("Model loaded!")

@app.post("/generate")
async def generate_subtitled_video(
    video: UploadFile = File(...),
    font: str = Form("Arial"),
    textColor: str = Form("#ffffff"),
    outlineColor: str = Form("#000000"),
    wordsPerLine: int = Form(4)
):
    try:
        with tempfile.TemporaryDirectory() as tmp:
            # Save video
            video_path = os.path.join(tmp, "input.mp4")
            with open(video_path, "wb") as f:
                f.write(await video.read())

            # Extract audio as WAV
            audio_path = os.path.join(tmp, "audio.wav")
            subprocess.run([
                "ffmpeg", "-i", video_path,
                "-ar", "16000", "-ac", "1", "-y", audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Load audio
            audio, _ = librosa.load(audio_path, sr=16000)

            # Transcribe
            result = asr_pipeline(audio)

            return {
                "message": "Success",
                "subtitles": result["chunks"],  # [{"timestamp": (start, end), "text": "..."}, ...]
                "font": font,
                "textColor": textColor,
                "outlineColor": outlineColor,
                "wordsPerLine": wordsPerLine
            }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})