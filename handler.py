import runpod
import base64
import tempfile
import os
import torch
import nemo.collections.asr as nemo_asr
from huggingface_hub import hf_hub_download

HF_REPO_ID = os.environ.get("HF_REPO_ID", "seifelshaer/arkani-quran-asr")
HF_FILENAME = os.environ.get("HF_FILENAME", "arkani_quran_full.nemo")
MODEL_CACHE = "/app/model_cache"

def load_model():
    os.makedirs(MODEL_CACHE, exist_ok=True)
    local_path = os.path.join(MODEL_CACHE, HF_FILENAME)
    if not os.path.exists(local_path):
        print(f"Downloading model from HuggingFace: {HF_REPO_ID}/{HF_FILENAME}")
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID, filename=HF_FILENAME, cache_dir=MODEL_CACHE, local_dir=MODEL_CACHE
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(local_path, map_location=device)
    model.eval()
    if hasattr(model, 'cur_decoder'):
        model.cur_decoder = 'ctc'
    return model

MODEL = load_model()

def handler(event):
    try:
        audio_base64 = event.get("input", {}).get("audio")
        if not audio_base64: return {"error": "No audio provided."}
        audio_bytes = base64.b64decode(audio_base64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        try:
            transcriptions = MODEL.transcribe([temp_path])
            if isinstance(transcriptions, list) and len(transcriptions) > 0:
                text = transcriptions[0].text if hasattr(transcriptions[0], 'text') else str(transcriptions[0])
            else:
                text = str(transcriptions)
            return {"text": text.strip(), "status": "success"}
        finally:
            if os.path.exists(temp_path): os.unlink(temp_path)
    except Exception as e:
        return {"error": str(e), "status": "failed"}

runpod.serverless.start({"handler": handler})
