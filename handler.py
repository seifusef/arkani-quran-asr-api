import os
import base64
import tempfile
import runpod
import torch
import librosa
import soundfile as sf
import nemo.collections.asr as nemo_asr
from recitation_analyzer import RecitationAnalyzer

# ─── تحميل الموديل ───
print("📥 جاري تحميل موديل NVIDIA Arabic Quran...")

MODEL_NAME = os.environ.get(
    "MODEL_NAME", 
    "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()

print(f"✅ Model loaded on: {device}")
print(f"📊 Decoder: RNNT (الأدق)")


def prepare_audio(audio_bytes: bytes, target_sr: int = 16000) -> str:
    """
    تجهيز الصوت: تحويل لـ WAV + 16kHz + mono باستخدام librosa
    """
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
        f.write(audio_bytes)
        input_path = f.name
    
    output_path = input_path + ".wav"
    
    try:
        # librosa بيقرأ أي صيغة (WAV, MP3, M4A, WebM, etc.)
        # mono=True بيحول لـ mono تلقائي
        # sr=target_sr بيعمل resample لـ 16kHz
        waveform, _ = librosa.load(input_path, sr=target_sr, mono=True)
        
        # حفظ كـ WAV باستخدام soundfile (موجود في NeMo image)
        sf.write(output_path, waveform, target_sr, subtype='PCM_16')
        
        return output_path
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)


def transcribe_audio_bytes(audio_bytes: bytes) -> dict:
    """تحويل الصوت لنص مع التشكيل"""
    wav_path = prepare_audio(audio_bytes)
    
    try:
        results = model.transcribe([wav_path], batch_size=1)
        result = results[0]
        text = result.text if hasattr(result, 'text') else str(result)
        text = text.strip()
        
        words = text.split() if text else []
        
        return {
            "text": text,
            "words": words,
            "has_diacritics": True,
        }
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


def handle_full_mode(input_data: dict) -> dict:
    """Full audio inference."""
    audio_base64 = input_data.get("audio_base64")
    if not audio_base64:
        return {"error": "Missing 'audio_base64' in input", "status": "error"}
    
    audio_bytes = base64.b64decode(audio_base64)
    result = transcribe_audio_bytes(audio_bytes)
    
    return {
        "text": result["text"],
        "has_diacritics": result["has_diacritics"],
        "status": "success"
    }


def handle_chunked_mode(input_data: dict) -> dict:
    """Chunked inference."""
    audio_base64 = input_data.get("audio_base64")
    session_id = input_data.get("session_id", "unknown")
    chunk_index = input_data.get("chunk_index", 0)
    previous_text = input_data.get("previous_text", "")
    
    if not audio_base64:
        return {
            "error": "Missing 'audio_base64' in chunked input",
            "status": "error",
            "session_id": session_id,
            "chunk_index": chunk_index,
        }
    
    try:
        audio_bytes = base64.b64decode(audio_base64)
        result = transcribe_audio_bytes(audio_bytes)
        
        chunk_text = result["text"]
        full_text = (previous_text + " " + chunk_text).strip() if previous_text else chunk_text
        
        return {
            "mode": "chunked",
            "session_id": session_id,
            "chunk_index": chunk_index,
            "text": chunk_text,
            "full_text": full_text,
            "words": result["words"],
            "duration_ms": 1500,
            "status": "success"
        }
    except Exception as e:
        return {
            "mode": "chunked",
            "session_id": session_id,
            "chunk_index": chunk_index,
            "error": str(e),
            "status": "error"
        }


def handle_analyze_mode(input_data: dict) -> dict:
    """Analyze mode with detailed comparison."""
    audio_base64 = input_data.get("audio_base64")
    expected_text = input_data.get("expected_text")
    surah_number = input_data.get("surah_number")
    ayah_number = input_data.get("ayah_number")

    if not audio_base64:
        return {"error": "Missing 'audio_base64' in input", "status": "error"}
    
    if expected_text is None:
        return {"error": "Missing 'expected_text' in input", "status": "error"}
    
    try:
        print(f"🔍 Analyzing Surah {surah_number}, Ayah {ayah_number}")
        print(f"📝 Expected: {expected_text}")
        
        audio_bytes = base64.b64decode(audio_base64)
        result = transcribe_audio_bytes(audio_bytes)
        
        transcription = result["text"]
        print(f"🎤 Transcribed: {transcription}")
        
        analyzer = RecitationAnalyzer()
        analysis = analyzer.analyze(transcription, expected_text)
        
        return {
            "transcription": transcription,
            "has_diacritics": result["has_diacritics"],
            "analysis": analysis,
            "status": "success"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "status": "error"
        }


def handler(event):
    """Main RunPod handler."""
    try:
        if event.get("httpMethod") == "GET" and event.get("path") == "/health":
            return {"status": "success", "message": "Model is loaded and ready."}
            
        input_data = event.get("input", {})
        mode = input_data.get("mode", "full")
        
        if mode == "chunked":
            return handle_chunked_mode(input_data)
        elif mode == "analyze":
            return handle_analyze_mode(input_data)
        elif mode == "health":
            return {"status": "success", "message": "Model is loaded and ready."}
        else:
            return handle_full_mode(input_data)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": f"Handler exception: {str(e)}",
            "status": "error"
        }


print("🚀 RunPod handler ready (NVIDIA Arabic Quran Model - Fixed Version)")
runpod.serverless.start({"handler": handler})
