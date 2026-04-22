# ─── Arkani Quran ASR - RunPod Serverless ───
FROM nvcr.io/nvidia/nemo:24.01.01

WORKDIR /app

# مش هنمسح PyTorch المرة دي عشان مكاتب نيفيديا ماتضربش
RUN pip install --no-cache-dir runpod huggingface_hub

COPY handler.py .

ENV HF_REPO_ID="seifelshaer/arkani-quran-asr"
ENV HF_FILENAME="arkani_quran_full.nemo"

CMD ["python", "handler.py"]
