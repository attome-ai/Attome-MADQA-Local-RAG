# ColQwen visual page retrieval (T18), built on the Kev image for its torch + transformers.
FROM madqa-kev
WORKDIR /kev
RUN uv pip install --python /kev/.venv/bin/python --no-deps colpali-engine pypdfium2 \
    && uv pip install --python /kev/.venv/bin/python numpy pillow peft torchvision==0.23.0
ENV HF_HOME=/hf
ENTRYPOINT ["/kev/.venv/bin/python"]
