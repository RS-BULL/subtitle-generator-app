import os
import subprocess
import tempfile
import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# Allow frontend (Live Server) to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            # Save uploaded video
            video_path = os.path.join(tmp, "input.mp4")
            with open(video_path, "wb") as f:
                f.write(await video.read())

            # Extract audio → 16kHz mono WAV
            audio_path = os.path.join(tmp, "audio.wav")
            subprocess.run([
                "/app/whisper.cpp-main",
                "-m", "/app/whisper_model/ggml-tiny.bin",
                "-f", audio_path,
                "-otxt"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Run whisper.cpp (tiny model)
            # Output: /app/whisper.cpp/audio.wav.txt
            subprocess.run([
                "/app/whisper.cpp/main",
                "-m", "/app/whisper_model/ggml-tiny.bin",
                "-f", audio_path,
                "-otxt"
            ], check=True, cwd="/app/whisper.cpp", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Read transcription
            txt_file = "/app/whisper.cpp/audio.wav.txt"
            if not os.path.exists(txt_file):
                return JSONResponse(status_code=500, content={"error": "Transcription failed"})

            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read().strip()

            # Clean up
            os.remove(txt_file)

            return {
                "message": "Success",
                "transcript": text,
                "font": font,
                "textColor": textColor,
                "outlineColor": outlineColor,
                "wordsPerLine": wordsPerLine
            }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})