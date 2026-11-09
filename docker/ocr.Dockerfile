FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "onnxruntime-gpu[cuda,cudnn]" rapidocr pypdfium2 numpy pillow
WORKDIR /app
# RapidOCR downloads its models into its package folder, so keep them across runs.
ENTRYPOINT ["python", "-m", "lab.ocr"]
