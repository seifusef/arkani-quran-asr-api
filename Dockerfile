# نستخدم NVIDIA NGC NeMo image (يحتوي على كل الـ dependencies الصحيحة)
FROM nvcr.io/nvidia/nemo:24.05

WORKDIR /app

# تثبيت RunPod SDK فقط (الباقي موجود في الـ image)
RUN pip install --no-cache-dir runpod

# نسخ الـ handler
COPY handler.py .

# Environment variables
ENV HF_REPO_ID="seifelshaer/arkani-quran-asr"
ENV HF_FILENAME="arkani_quran_full.nemo"
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
