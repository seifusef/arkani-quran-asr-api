# ─── Arkani Quran ASR - RunPod Serverless ───
# Base: NVIDIA NeMo container (has PyTorch + NeMo pre-installed)
FROM nvcr.io/nvidia/nemo:24.01.01

WORKDIR /app

# Remove broken torchaudio to prevent crash
RUN pip uninstall -y torchaudio

# Install RunPod SDK + HuggingFace Hub
RUN pip install --no-cache-dir runpod huggingface_hub

# Copy handler
COPY handler.py .

# Environment variables (override in RunPod dashboard)
ENV HF_REPO_ID="seifelshaer/arkani-quran-asr"
ENV HF_FILENAME="arkani_quran_full.nemo"

# Start handler
CMD ["python", "handler.py"]
