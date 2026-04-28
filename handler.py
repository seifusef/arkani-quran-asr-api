import os
import base64
import tempfile
import runpod
import torch
import nemo.collections.asr as nemo_asr
from recitation_analyzer import RecitationAnalyzer

# ─── تحميل الموديل من NVIDIA مباشرة ───
print("📥 جاري تحميل موديل NVIDIA FastConformer Arabic Quran...")

# الموديل بيتحمل من Hugging Face Hub تلقائياً
MODEL_NAME = os.environ.get(
    "MODEL_NAME", 
    "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
    MODEL_NAME
)
model = model.to(device)
model.eval()

if hasattr(model, 'cur_decoder'):
    model.cur_decoder = 'ctc'

print(f"✅ Model loaded on: {device}")
print(f"📊 Model: {MODEL_NAME}")


def transcribe_audio_bytes(audio_bytes: bytes) -> dict:
    """
    Transcribe audio bytes and return text + word list.
    الموديل الجديد بيرجع نص بتشكيل كامل.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name
    
    try:
        results = model.transcribe([temp_path], batch_size=1)
        result = results[0]
        text = result.text if hasattr(result, 'text') else str(result)
        text = text.strip()
        
        words = text.split() if text else []
        
        return {
            "text": text,
            "words": words,
            "has_diacritics": True,  # الموديل الجديد دايماً بيدي تشكيل
        }
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


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
    """Chunked inference for live word highlighting."""
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
    """Analyze mode with Needleman-Wunsch alignment."""
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
        audio_bytes = base64.b64decode(audio_base64)
        result = transcribe_audio_bytes(audio_bytes)
        
        transcription = result["text"]
        
        analyzer = RecitationAnalyzer()
        analysis = analyzer.analyze(transcription, expected_text)
        
        return {
            "transcription": transcription,
            "has_diacritics": result["has_diacritics"],
            "analysis": analysis,
            "status": "success"
        }
    except Exception as e:
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
        return {
            "error": f"Handler exception: {str(e)}",
            "status": "error"
        }


print("🚀 RunPod handler ready (NVIDIA Arabic Quran Model)")
runpod.serverless.start({"handler": handler})
