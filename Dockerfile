# MedDoc RAG: one image, two roles (API by default; UI via docker-compose).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# CPU-only PyTorch first: the default Linux wheel bundles CUDA (~2.5 GB) that this app never uses.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Dependencies before the rest of the code, so code edits don't invalidate this layer.
COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
# Editable install: config resolves data/ and eval/ relative to the source tree (/app).
RUN pip install -e ".[ui]"

# Bake model weights into the image so containers start without downloading ~200 MB.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
ENV HF_HUB_OFFLINE=1

COPY scripts ./scripts
COPY eval ./eval

# Never run as root. data/ is mounted at runtime; the API key is passed as an env var, never baked in.
RUN useradd --create-home appuser && mkdir -p /app/data && chown -R appuser /app /opt/hf
USER appuser

EXPOSE 8000 8501
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
