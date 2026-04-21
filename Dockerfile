# ─── Arkani Quran ASR - RunPod Serverless ───
FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg sox libsndfile1 cmake \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod \
    huggingface_hub \
    Cython \
    soundfile \
    "nemo_toolkit[asr]"

COPY handler.py .

ENV HF_REPO_ID="seifelshaer/arkani-quran-asr"
ENV HF_FILENAME="arkani_quran_full.nemo"

CMD ["python", "-u", "handler.py"]
