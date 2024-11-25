from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import Optional, Dict, List, Tuple, Any
import os
import numpy as np
import soundfile as sf
from pathlib import Path
import base64
import io
from dataclasses import dataclass

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT, UNetT

# Pydantic models for request validation
class Voice(BaseModel):
    ref_audio: str  # Base64 encoded audio file
    ref_text: str

class TTSRequest(BaseModel):
    model: str = "F5-TTS"
    gen_text: str
    voices: Optional[Dict[str, Voice]] = None
    main_voice: Voice
    remove_silence: bool = False
    vocoder_name: str = "vocos"
    speed: float = 1.0
    output_format: str = "wav"

@dataclass
class ProcessedVoice:
    ref_audio: Any  # Processed reference audio
    ref_text: str

app = FastAPI(title="F5/E2 TTS API")

# Global variables for models
models = {}
vocoders = {}
processed_voice_cache = {}

@app.on_event("startup")
async def load_models():
    # Load vocoder models
    vocoders["vocos"] = load_vocoder(
        vocoder_name="vocos",
        is_local=False,
        local_path="../checkpoints/vocos-mel-24khz"
    )
    vocoders["bigvgan"] = load_vocoder(
        vocoder_name="bigvgan",
        is_local=False,
        local_path="../checkpoints/bigvgan_v2_24khz_100band_256x"
    )
    
    # Load F5-TTS model
    model_cfg_f5 = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    models["F5-TTS"] = load_model(
        DiT,
        model_cfg_f5,
        "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors",
        mel_spec_type="vocos"
    )
    
    # Load E2-TTS model
    model_cfg_e2 = dict(dim=1024, depth=24, heads=16, ff_mult=4)
    models["E2-TTS"] = load_model(
        UNetT,
        model_cfg_e2,
        "hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors",
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

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    if request.model not in models:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not supported")
    
    if request.vocoder_name not in vocoders:
        raise HTTPException(status_code=400, detail=f"Vocoder {request.vocoder_name} not supported")
    
    if request.model == "E2-TTS" and request.vocoder_name != "vocos":
        raise HTTPException(status_code=400, detail="E2-TTS only supports vocos vocoder")

    try:
        # Modified voice processing
        processed_voices = {}
        
        # Process main voice if it's not in cache
        main_voice_key = None
        for cached_name, cached_voice in processed_voice_cache.items():
            if (base64.b64decode(request.main_voice.ref_audio) == Path(f"voices/{cached_name}.wav").read_bytes() and 
                request.main_voice.ref_text == cached_voice.ref_text):
                main_voice_key = cached_name
                break
                
        if main_voice_key:
            processed_voices["main"] = processed_voice_cache[main_voice_key]
        else:
            # Process uncached voice as before
            audio_data = base64.b64decode(request.main_voice.ref_audio)
            temp_audio_path = "temp_main.wav"
            with open(temp_audio_path, "wb") as f:
                f.write(audio_data)
            
            processed_audio, processed_text = preprocess_ref_audio_text(
                temp_audio_path, 
                request.main_voice.ref_text
            )
            processed_voices["main"] = ProcessedVoice(processed_audio, processed_text)
            os.remove(temp_audio_path)

        # Process additional voices if present
        if request.voices:
            for voice_name, voice in request.voices.items():
                # Check if voice is in cache
                for cached_name, cached_voice in processed_voice_cache.items():
                    if (base64.b64decode(voice.ref_audio) == Path(f"voices/{cached_name}.wav").read_bytes() and 
                        voice.ref_text == cached_voice.ref_text):
                        processed_voices[voice_name] = cached_voice
                        break
                else:
                    # Process uncached voice as before
                    audio_data = base64.b64decode(voice.ref_audio)
                    temp_audio_path = f"temp_{voice_name}.wav"
                    with open(temp_audio_path, "wb") as f:
                        f.write(audio_data)
                    
                    processed_audio, processed_text = preprocess_ref_audio_text(
                        temp_audio_path, 
                        voice.ref_text
                    )
                    processed_voices[voice_name] = ProcessedVoice(processed_audio, processed_text)
                    os.remove(temp_audio_path)

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
                voice["ref_audio"],
                voice["ref_text"],
                text.strip(),
                models[request.model],
                vocoders[request.vocoder_name],
                mel_spec_type=request.vocoder_name,
                speed=request.speed
            )
            generated_audio_segments.append(audio)

        # Combine audio segments
        final_wave = np.concatenate(generated_audio_segments)

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