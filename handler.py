import os
import io
import base64
import tempfile
import runpod
import torch
import numpy as np
import soundfile as sf
import librosa
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

# Cache للسرعة
_session_cache = {}  # session_id -> last_transcription


def _clean_text(text):
    """تنظيف النص من الـ Python list wrapper"""
    if not text:
        return ""
    text = str(text).strip()
    if text.startswith("['") and text.endswith("']"):
        text = text[2:-2]
    elif text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip("'\"")
    return text.strip()


def _transcribe_audio_fast(audio_bytes: bytes, target_sr: int = 16000) -> str:
    """
    تحويل سريع للصوت لنص — كل العمليات في الذاكرة.
    """
    # حاول نقرأ من الذاكرة مباشرة (أسرع من ملف)
    try:
        # soundfile ممكن يقرأ WAV/FLAC في الذاكرة
        audio_io = io.BytesIO(audio_bytes)
        waveform, sr = sf.read(audio_io, dtype='float32')
        
        # Mono conversion
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        
        # Resample لو محتاج
        if sr != target_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
    except Exception:
        # Fallback: لو soundfile فشل، استخدم temp file
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
            f.write(audio_bytes)
            input_path = f.name
        try:
            waveform, _ = librosa.load(input_path, sr=target_sr, mono=True)
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
    
    # تحقق من طول الصوت — لو قصير جداً، ارجع فاضي
    if len(waveform) < target_sr * 0.3:  # أقل من 0.3 ثانية
        return ""
    
    # تحقق من إن مش كله صمت
    if np.abs(waveform).max() < 0.01:
        return ""
    
    # احفظ كـ WAV temp للموديل (NeMo محتاج file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    
    try:
        sf.write(wav_path, waveform, target_sr, subtype='PCM_16')
        results = model.transcribe([wav_path], batch_size=1)
        text = results[0] if results else ""
        return _clean_text(text)
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


def handle_full_mode(input_data: dict) -> dict:
    """Full audio inference."""
    audio_base64 = input_data.get("audio_base64")
    if not audio_base64:
        return {"error": "Missing 'audio_base64' in input", "status": "error"}
    
    try:
        audio_bytes = base64.b64decode(audio_base64)
        text = _transcribe_audio_fast(audio_bytes)
        
        return {
            "text": text,
            "has_diacritics": True,
            "status": "success"
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "error"
        }


def handle_chunked_mode(input_data: dict) -> dict:
    """
    Chunked inference - محسّن للسرعة.
    """
    audio_base64 = input_data.get("audio_base64", "")
    session_id = input_data.get("session_id", "unknown")
    chunk_index = input_data.get("chunk_index", 0)
    previous_text = input_data.get("previous_text", "")
    
    if not audio_base64:
        return {
            "error": "Missing 'audio_base64'",
            "status": "error",
            "session_id": session_id,
            "chunk_index": chunk_index,
        }
    
    try:
        audio_bytes = base64.b64decode(audio_base64)
        
        # Skip لو الـ chunk صغير جداً (أقل من 0.5 ثانية)
        # WAV mono 16kHz 16-bit = 32000 bytes/sec
        # 0.5 sec = 16000 bytes (+ 44 byte header)
        if len(audio_bytes) < 16044:
            return {
                "mode": "chunked",
                "session_id": session_id,
                "chunk_index": chunk_index,
                "text": "",
                "full_text": previous_text,
                "status": "success"
            }
        
        # Transcribe
        text = _transcribe_audio_fast(audio_bytes)
        
        # Cache آخر transcription لده الـ session
        _session_cache[session_id] = text
        
        return {
            "mode": "chunked",
            "session_id": session_id,
            "chunk_index": chunk_index,
            "text": text,
            "full_text": text,  # في chunked mode بنبعت كل buffer كل مرة
            "status": "success"
        }
        
    except Exception as e:
        import traceback
        return {
            "mode": "chunked",
            "session_id": session_id,
            "chunk_index": chunk_index,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "error"
        }


def handle_analyze_mode(input_data: dict) -> dict:
    """Analyze mode with detailed comparison."""
    audio_base64 = input_data.get("audio_base64")
    expected_text = input_data.get("expected_text")
    surah_number = input_data.get("surah_number")
    ayah_number = input_data.get("ayah_number")

    if not audio_base64:
        return {"error": "Missing 'audio_base64'", "status": "error"}
    
    try:
        audio_bytes = base64.b64decode(audio_base64)
        transcription = _transcribe_audio_fast(audio_bytes)
        
        # لو مفيش expected_text، رجّع الـ transcription بس
        if expected_text is None:
            return {
                "transcription": transcription,
                "has_diacritics": True,
                "status": "success"
            }
        
        # Analyze
        analyzer = RecitationAnalyzer()
        analysis = analyzer.analyze(transcription, expected_text)
        
        return {
            "transcription": transcription,
            "has_diacritics": True,
            "analysis": analysis,
            "status": "success"
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "error"
        }


def handler(event):
    """Main RunPod handler."""
    try:
        input_data = event.get("input", {})
        mode = input_data.get("mode", "full")
        
        # Check task field (Web sometimes sends "task" instead of "mode")
        if "task" in input_data and mode == "full":
            mode = "full"  # web full mode
        
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
        return {
            "error": f"Handler exception: {str(e)}",
            "traceback": traceback.format_exc(),
            "status": "error"
        }


print("🚀 RunPod handler ready (Fast Mode)")
runpod.serverless.start({"handler": handler})
