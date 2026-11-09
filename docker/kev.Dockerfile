# Kev decision-model server: https://github.com/jaredpalmer/kev (Apache-2.0)
# Built on the local Python 3.12 image, since pulling a new base can fail here.
FROM madqa-ocr
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
RUN git clone https://github.com/jaredpalmer/kev.git /kev
WORKDIR /kev
ENV UV_TORCH_BACKEND=auto
RUN uv sync --extra serve
EXPOSE 8009
ENTRYPOINT ["uv", "run", "--no-sync", "--extra", "serve", "python", "-m", "kev.serve"]
