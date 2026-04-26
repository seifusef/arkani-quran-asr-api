FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# تثبيت الأدوات الأساسية
RUN apt-get update && apt-get install -y \
    ffmpeg sox libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# تثبيت المكتبات (NeMo أصغر بدون كل الـ dependencies)
RUN pip install --no-cache-dir \
    runpod \
    huggingface_hub \
    "nemo_toolkit[asr]==1.23.0" \
    soundfile

COPY handler.py .

ENV HF_REPO_ID="seifelshaer/arkani-quran-asr"
ENV HF_FILENAME="arkani_quran_full.nemo"

CMD ["python", "-u", "handler.py"]
