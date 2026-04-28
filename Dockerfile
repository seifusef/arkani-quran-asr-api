FROM nvcr.io/nvidia/nemo:24.05

WORKDIR /app

# تثبيت RunPod SDK
RUN pip install --no-cache-dir runpod

# نسخ الملفات
COPY handler.py .
COPY recitation_analyzer.py .

# Environment variables
ENV MODEL_NAME="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
