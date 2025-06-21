# main.py
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
import torch
import torchaudio
import soundfile as sf
import io
import os
import sys

# Ensure MeloTTS is installable via requirements.txt
# If you cloned the MeloTTS repo directly into your project, you might need to adjust
# the Python path, but for pip installs, it should be fine.
try:
    from melo.api import TTS
except ImportError:
    # Fallback/Error if melo.api is not found, good for local testing before pip install
    print("MeloTTS not found. Please ensure it's installed via requirements.txt or locally.")
    sys.exit(1)

app = FastAPI(
    title="MeloTTS API",
    description="Text-to-Speech API powered by MeloTTS, deployed on Render.",
    version="1.0.0"
)

# --- Global Model Loading ---
# This part runs once when your FastAPI application starts up.
# It will download the necessary model weights if they are not already present.
# This can take time and consume memory during startup.
# CRITICAL: Render's free tier is NOT enough. You will need a paid instance
# with at least 2GB RAM (e.g., Standard plan) for the model to load successfully.
# Even on a CPU instance, MeloTTS can run, but it will be slower than GPU.
model_instance = None
loading_error = None

@app.on_event("startup")
async def load_model():
    global model_instance, loading_error
    try:
        print(f"[{os.getenv('RENDER_INSTANCE_ID', 'local')}] Starting model loading...")
        
        # Determine device: 'cuda' for GPU (if Render offers GPU and you use that plan), else 'cpu'
        # For typical Render CPU instances, 'cpu' is correct.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[{os.getenv('RENDER_INSTANCE_ID', 'local')}] Attempting to load MeloTTS model on device: {device}")

        # Initialize TTS model for English. You can change 'en' to other supported languages.
        # This call handles downloading models if not cached.
        model_instance = TTS(language='en', device=device)
        
        print(f"[{os.getenv('RENDER_INSTANCE_ID', 'local')}] MeloTTS model loaded successfully.")

        # Optionally, get available speaker IDs if you plan to allow specific voice selection
        # speaker_ids = model_instance.hps.data.spk_ids
        # print(f"Available speaker IDs: {speaker_ids}")

    except Exception as e:
        loading_error = f"Failed to load MeloTTS model: {e}"
        print(f"[{os.getenv('RENDER_INSTANCE_ID', 'local')}] ERROR: {loading_error}")
        # In a production setup, you might want to exit here if the model is critical
        # sys.exit(1) # Uncomment if you want the service to fail immediately on model load error

class TTSRequest(BaseModel):
    text: str
    voice_name: str = "EN-US" # Default voice, MeloTTS uses names like 'EN-US', 'EN-BR', 'EN-IN', etc.

@app.get("/")
async def read_root():
    if loading_error:
        return {"status": "error", "message": f"API is running but model failed to load: {loading_error}"}
    if not model_instance:
        return {"status": "loading", "message": "MeloTTS model is still loading or failed to load. Please try again shortly."}
    return {"status": "ready", "message": "MeloTTS API is running and model is loaded."}

@app.post("/tts")
async def generate_speech(request: TTSRequest):
    if loading_error:
        raise HTTPException(status_code=503, detail=f"Service Unavailable: Model failed to load: {loading_error}")
    if not model_instance:
        raise HTTPException(status_code=503, detail="Service Unavailable: MeloTTS model is still loading. Please try again shortly.")

    if not request.text:
        raise HTTPException(status_code=400, detail="Text input is required.")

    try:
        print(f"[{os.getenv('RENDER_INSTANCE_ID', 'local')}] Received TTS request for voice: {request.voice_name}, text: '{request.text[:50]}...'")

        # MeloTTS's `synthesize` method typically expects a speaker_id.
        # The `TTS` class's `hps.data.spk_ids` maps names to internal IDs.
        # You'll need to ensure `request.voice_name` maps to a valid ID.
        # For simplicity, we'll try to use voice_name directly if it's a known string ID (e.g., 'EN-US')
        # If MeloTTS requires a numeric ID, you'd need to convert:
        # speaker_id_value = model_instance.hps.data.spk_ids.get(request.voice_name, model_instance.hps.data.spk_ids['EN-US'])
        
        # Calling the synthesis function
        # This returns a NumPy array of audio data
        audio_data_np = model_instance.synthesize(request.text, request.voice_name) # Assuming voice_name directly corresponds to a valid speaker

        # Convert the NumPy array to WAV bytes
        output_buffer = io.BytesIO()
        sf.write(output_buffer, audio_data_np, model_instance.hps.data.sampling_rate, format='WAV')
        output_buffer.seek(0)

        print(f"[{os.getenv('RENDER_INSTANCE_ID', 'local')}] Successfully generated audio.")
        return Response(content=output_buffer.getvalue(), media_type="audio/wav")

    except Exception as e:
        # Log the full traceback for debugging on Render
        import traceback
        print(f"[{os.getenv('RENDER_INSTANCE_ID', 'local')}] ERROR during TTS generation: {e}")
        traceback.print_exc() # This will print the full error stack to Render logs
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {e}. Check server logs for details.")
