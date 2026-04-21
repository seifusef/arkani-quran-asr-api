# ─── Arkani Quran ASR - RunPod Serverless ───
# Base: NVIDIA NeMo container
FROM nvcr.io/nvidia/nemo:24.01.01

WORKDIR /app

# 1. Fix the broken torchaudio by installing official PyTorch 2.2.0 wheels
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# 2. Install RunPod SDK
RUN pip install --no-cache-dir runpod huggingface_hub

# 3. Copy handler
COPY handler.py .

# 4. Environment variables
ENV HF_REPO_ID="seifelshaer/arkani-quran-asr"
ENV HF_FILENAME="arkani_quran_full.nemo"

# 5. Start handler
CMD ["python", "handler.py"]
