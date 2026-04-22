# ─── Arkani Quran ASR - RunPod Serverless ───
FROM nvcr.io/nvidia/nemo:24.01.01

WORKDIR /app

# الخطوة السحرية: مسح الملفات المكسورة وتنزيل النسخة الرسمية
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# تسطيب باقي المكاتب
RUN pip install --no-cache-dir runpod huggingface_hub

COPY handler.py .

ENV HF_REPO_ID="seifelshaer/arkani-quran-asr"
ENV HF_FILENAME="arkani_quran_full.nemo"

CMD ["python", "handler.py"]
