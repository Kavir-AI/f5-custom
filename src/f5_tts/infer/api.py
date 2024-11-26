from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import soundfile as sf
from pathlib import Path
import base64
import io
from dataclasses import dataclass
from cached_path import cached_path
import torch
import os
import sys
from contextlib import asynccontextmanager
from asyncio import Semaphore

# Improve path handling for resemble-enhance
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent  # Go up 3 levels from /src/f5_tts/infer/
resemble_path = project_root / "src" /"third_party" / "resemble_custom" / "resemble_enhance"

if resemble_path.exists():
    sys.path.append(str(resemble_path))
else:
    raise ImportError(f"Required directory not found: {resemble_path}")

# Optional: Debug logging
print(f"Added to system path: {resemble_path}")
for path in sys.path:
    print(f"- {path}")

print("Loading resemble enhance")
from third_party.resemble_custom.resemble_enhance.enhancer.inference import load_enhancer, enhance

print("Loading utils_infer")
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)

print("Loading model")
from f5_tts.model import DiT, UNetT

print("Loading Pydantic models")
# Pydantic models for request validation
class Voice(BaseModel):
    voice_name: str  # Changed from ref_audio and ref_text to just voice_name

class TTSRequest(BaseModel):
    model: str = "F5-TTS"
    gen_text: str
    voices: Optional[Dict[str, Voice]] = None
    main_voice: Voice
    remove_silence: bool = False
    vocoder_name: str = "vocos"
    speed: float = 1.0
    output_format: str = "wav"
    enhance_audio: bool = False

@dataclass
class ProcessedVoice:
    ref_audio: Any  # Processed reference audio
    ref_text: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading vocoder models")
    vocoders["vocos"] = load_vocoder(
        vocoder_name="vocos",
        is_local=False,
        local_path="../checkpoints/vocos-mel-24khz"
    )
    print("Vocos loaded successfully")
    vocoders["bigvgan"] = load_vocoder(
        vocoder_name="bigvgan",
        is_local=False,
        local_path="../checkpoints/bigvgan_v2_24khz_100band_256x"
    )
    print("Bigvgan loaded successfully")

    # Load F5-TTS model
    model_cfg_f5 = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    print("Fetching F5-TTS model")
    f5_model_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
    print("F5-TTS model path:", f5_model_path)
    models["F5-TTS"] = load_model(
        DiT,
        model_cfg_f5,
        f5_model_path,
        mel_spec_type="vocos"
    )
    
    # Load enhancer model
    print("Loading enhancer model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models["enhancer"] = load_enhancer(run_dir=None, device=device)
    print("Enhancer model loaded successfully")
    
    # Load E2-TTS model
    model_cfg_e2 = dict(dim=1024, depth=24, heads=16, ff_mult=4)
    print("Fetching E2-TTS model")
    e2_model_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
    print("E2-TTS model path:", e2_model_path)
    models["E2-TTS"] = load_model(
        UNetT,
        model_cfg_e2,
        e2_model_path,
        mel_spec_type="vocos"
    )

    # Add voice preprocessing
    voices_dir = Path(__file__).parent / "voices"
    voice_files = {
        path.stem: path for path in voices_dir.glob("*.wav")
    }
    
    # Process each known voice
    for voice_name, voice_path in voice_files.items():
        print(f"Processing voice file: {voice_path}")
        # Assuming each voice file has a corresponding .txt file with the reference text
        text_path = voice_path.with_suffix('.txt')
        if not text_path.exists():
            print(f"Warning: No reference text found for voice {voice_name}")
            continue
            
        print(f"Found reference text file: {text_path}")
        ref_text = text_path.read_text().strip()
        processed_audio, processed_text = preprocess_ref_audio_text(str(voice_path), ref_text)
        processed_voice_cache[voice_name] = ProcessedVoice(processed_audio, processed_text)
        print(f"Successfully loaded and processed voice: {voice_name}")

    yield  # Server is running and ready to accept requests
    
    # Cleanup (if needed) when the server shuts down
    # For example: models.clear(), vocoders.clear()
    print("Clearing models and vocoders")

# Update FastAPI initialization to use lifespan
print("Initializing FastAPI")
app = FastAPI(title="F5/E2 TTS API", lifespan=lifespan)
print("FastAPI initialized")

# Global variables for models
models = {}
vocoders = {}
processed_voice_cache = {}

# Initialize semaphore to allow only 1 request at a time
tts_semaphore = Semaphore(1)

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    # Acquire semaphore before processing
    async with tts_semaphore:
        if request.model not in models:
            raise HTTPException(status_code=400, detail=f"Model {request.model} not supported")
        
        if request.vocoder_name not in vocoders:
            raise HTTPException(status_code=400, detail=f"Vocoder {request.vocoder_name} not supported")
        
        if request.model == "E2-TTS" and request.vocoder_name != "vocos":
            raise HTTPException(status_code=400, detail="E2-TTS only supports vocos vocoder")

        try:
            # Modified voice processing
            processed_voices = {}
            
            # Get main voice from cache
            if request.main_voice.voice_name not in processed_voice_cache:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Voice '{request.main_voice.voice_name}' not found. Available voices: {list(processed_voice_cache.keys())}"
                )
            
            processed_voices["main"] = processed_voice_cache[request.main_voice.voice_name]

            # Process additional voices if present
            if request.voices:
                for voice_name, voice in request.voices.items():
                    if voice.voice_name not in processed_voice_cache:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Voice '{voice.voice_name}' not found. Available voices: {list(processed_voice_cache.keys())}"
                        )
                    processed_voices[voice_name] = processed_voice_cache[voice.voice_name]

            # Generate audio segments
            generated_audio_segments = []
            for text_chunk in request.gen_text.split("["):
                if not text_chunk.strip():
                    continue
                    
                if "]" in text_chunk:
                    voice_name, text = text_chunk.split("]", 1)
                    if voice_name not in processed_voices:
                        voice_name = "main"
                else:
                    voice_name, text = "main", text_chunk

                voice = processed_voices[voice_name]
                audio, sample_rate, _ = infer_process(
                    voice.ref_audio,
                    voice.ref_text,
                    text.strip(),
                    models[request.model],
                    vocoders[request.vocoder_name],
                    mel_spec_type=request.vocoder_name,
                    speed=request.speed
                )
                generated_audio_segments.append(audio)

            # Combine audio segments
            final_wave = np.concatenate(generated_audio_segments)

            # Add enhancement if requested
            if request.enhance_audio:
                # Convert numpy array to torch tensor
                wav_tensor = torch.from_numpy(final_wave).float()
                # Enhance the audio using resemble enhance
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                enhanced_wave, sample_rate = enhance(wav_tensor, sample_rate, device)
                # Convert back to numpy array
                final_wave = enhanced_wave.cpu().numpy()

            # Instead of converting to base64, return the audio file directly
            buffer = io.BytesIO()
            sf.write(buffer, final_wave, sample_rate, format=request.output_format)
            buffer.seek(0)

            # Create appropriate content type based on format
            content_type = "audio/wav" if request.output_format == "wav" else "audio/mpeg"
            
            return Response(
                content=buffer.read(),
                media_type=content_type,
                headers={
                    "Content-Disposition": f'attachment; filename="generated_audio.{request.output_format}"'
                }
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) 

@app.get("/available-voices")
async def list_voices():
    return {"voices": list(processed_voice_cache.keys())} 