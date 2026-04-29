FROM nvcr.io/nvidia/nemo:24.05

WORKDIR /app

# torchaudio موجود في NeMo image - لو ضفنا torchaudio هنكسر CUDA
# soundfile موجود برضه. بنضيف runpod بس
RUN pip install --no-cache-dir runpod

COPY handler.py .
COPY recitation_analyzer.py .

ENV MODEL_NAME="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
