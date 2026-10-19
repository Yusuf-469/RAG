# MedDoc RAG: one image, two roles (API by default; UI via docker-compose).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

# Non-root user created up front, so later layers can be written with the right owner
# (a recursive chown at the end would copy every file into a new layer).
RUN useradd --create-home appuser && mkdir -p /app/data /opt/hf && chown appuser /app /app/data /opt/hf
WORKDIR /app

# CPU-only PyTorch first: the default Linux wheel bundles CUDA (~2.5 GB) that this app never uses.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies from pyproject.toml alone, using an empty stub package. Code edits then
# only invalidate the cheap COPY layers below instead of reinstalling every dependency.
COPY pyproject.toml README.md ./
RUN mkdir -p src/meddoc_rag && touch src/meddoc_rag/__init__.py && pip install -e ".[ui]"

USER appuser
# Bake model weights into the image so containers start without downloading ~200 MB.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
ENV HF_HUB_OFFLINE=1

# Code last. Editable install: the package resolves to /app/src, and config finds data/ and eval/
# relative to it. data/ is mounted at runtime; the API key is passed as an env var, never baked in.
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser scripts ./scripts
COPY --chown=appuser:appuser eval ./eval

EXPOSE 8000 8501
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
