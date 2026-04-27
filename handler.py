import os
import base64
import tempfile
import runpod
import torch
import nemo.collections.asr as nemo_asr
from huggingface_hub import hf_hub_download

# ─── تحميل الموديل (مرة واحدة فقط) ───
print("📥 جاري تحميل الموديل من HuggingFace...")

REPO_ID = os.environ.get("HF_REPO_ID", "seifelshaer/arkani-quran-asr")
FILENAME = os.environ.get("HF_FILENAME", "arkani_quran_full.nemo")

model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
)

print(f"✅ الموديل: {model_path}")

# تحميل الموديل
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    model_path, 
    map_location=device
)
model.eval()

# استخدام CTC decoder (أدق للقرآن)
if hasattr(model, 'cur_decoder'):
    model.cur_decoder = 'ctc'

print(f"✅ الموديل اتحمّل على: {device}")


# ─── Handler Function ───
def handler(event):
    """
    استقبال طلبات تحويل الصوت لنص.
    
    Input:
    {
        "input": {
            "audio_base64": "...base64-encoded-wav..."
        }
    }
    
    Output:
    {
        "text": "النص المحوّل"
    }
    """
    try:
        input_data = event.get("input", {})
        audio_base64 = input_data.get("audio_base64")
        
        if not audio_base64:
            return {"error": "Missing 'audio_base64' in input"}
        
        # فك التشفير
        audio_bytes = base64.b64decode(audio_base64)
        
        # حفظ في ملف مؤقت
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        # Transcribe
        results = model.transcribe([temp_path], batch_size=1)
        
        # تنظيف
        os.unlink(temp_path)
        
        # استخراج النص
        result = results[0]
        text = result.text if hasattr(result, 'text') else str(result)
        
        return {
            "text": text.strip(),
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }


# ─── تشغيل RunPod ───
print("🚀 RunPod handler جاهز!")
runpod.serverless.start({"handler": handler})
