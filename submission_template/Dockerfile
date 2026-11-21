# Submission contract: the container must run
#   <entrypoint> /data/images /data/predictions
# offline (no network). Bundle all model weights in the image.
# Built from the repository root:
#   docker build -t my-tablesalt -f submission_template/Dockerfile .
FROM python:3.12-slim

# Tesseract 5 with German + English data (Debian ships the tessdata_fast models)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# pinned versions (same as the local evaluation environment)
RUN pip install --no-cache-dir numpy==2.5.5 opencv-python-headless==5.1.0.95 \
        pytesseract==0.3.13 onnxruntime==1.32.0 pillow==13.0.0

WORKDIR /app
# code + weights live in the image: the run has no network.
# models/ holds the PP-OCRv5 latin recognizer (ONNX, 7.9 MB, Apache-2.0,
# monkt/paddleocr-onnx@7b02d0a3) and the triage tree.
COPY tableparser/ /app/tableparser/
COPY models/ /app/models/
COPY submission_template/ /app/submission_template/

# OMP_THREAD_LIMIT=1: the pipeline already runs up to 8 Tesseract calls in
# parallel; Debian's OpenMP Tesseract would oversubscribe the CPUs (dev set:
# mean 7.8 -> 3.7 s/table, max 15.4 -> 6.5 s with this setting)
ENV PYTHONUNBUFFERED=1 TABLEPARSER_MODELS=/app/models OMP_THREAD_LIMIT=1
ENTRYPOINT ["python", "submission_template/predict.py", "/data/images", "/data/predictions"]
