import os
import base64
import tempfile
import runpod
import torch
import nemo.collections.asr as nemo_asr
from huggingface_hub import hf_hub_download
from recitation_analyzer import RecitationAnalyzer

# ─── Model loading (UNCHANGED) ───
print("📥 جاري تحميل الموديل من HuggingFace...")

REPO_ID = os.environ.get("HF_REPO_ID", "seifelshaer/arkani-quran-asr")
FILENAME = os.environ.get("HF_FILENAME", "arkani_quran_full.nemo")

model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    model_path, map_location=device
)
model.eval()
if hasattr(model, 'cur_decoder'):
    model.cur_decoder = 'ctc'

print(f"✅ Model loaded on: {device}")


def transcribe_audio_bytes(audio_bytes: bytes) -> dict:
    """
    Transcribe audio bytes and return text + word list.
    Used by both full and chunked modes.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name
    
    try:
        results = model.transcribe([temp_path], batch_size=1)
        result = results[0]
        text = result.text if hasattr(result, 'text') else str(result)
        text = text.strip()
        
        # Split into words for word-level highlighting
        words = text.split() if text else []
        
        return {
            "text": text,
            "words": words,
        }
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def handle_full_mode(input_data: dict) -> dict:
    """
    Original full-audio inference (UNCHANGED behavior).
    """
    audio_base64 = input_data.get("audio_base64")
    if not audio_base64:
        return {"error": "Missing 'audio_base64' in input", "status": "error"}
    
    audio_bytes = base64.b64decode(audio_base64)
    result = transcribe_audio_bytes(audio_bytes)
    
    return {
        "text": result["text"],
        "status": "success"
    }


def handle_chunked_mode(input_data: dict) -> dict:
    """
    NEW: Chunked inference for live word highlighting.
    Receives a short audio chunk (~1.5s) and returns transcription
    that the client can append to its running text.
    """
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
    """
    NEW: Analyze mode for word-level recitation comparison using Needleman-Wunsch.
    """
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
            "analysis": analysis,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }


def handler(event):
    """
    Main RunPod handler. Routes to full, chunked, or analyze mode based on input.
    """
    try:
        # Simple health check mode
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


print("🚀 RunPod handler ready (full, chunked, and analyze modes)")
runpod.serverless.start({"handler": handler})
