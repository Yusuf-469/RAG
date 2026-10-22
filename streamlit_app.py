"""Streamlit Community Cloud entry point: API and UI in one free app.

Community Cloud runs a single Streamlit process, so this file starts the FastAPI service in a
background thread on loopback, then runs the normal UI (app/ui.py) against it. The index is
downloaded from a private Hugging Face dataset (scripts/upload_index.py) because it contains
text from CC BY-NC-SA documents that this repo does not redistribute.

Secrets (app settings -> Secrets):
    GROQ_API_KEY = "..."
    HF_TOKEN = "..."                                   # read access to the index dataset
    MEDDOC_INDEX_REPO = "yusuf469/meddoc-rag-index"    # optional override
"""

import os
import runpy
import threading
import time
from pathlib import Path

import requests
import streamlit as st

# Secrets must be in the environment before meddoc_rag reads its settings at import time.
try:
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except FileNotFoundError:   # no secrets file: running locally with .env
    pass

API_URL = "http://127.0.0.1:8000"
os.environ["MEDDOC_API_URL"] = API_URL


@st.cache_resource(show_spinner="Starting up: downloading the index and loading models (about a minute)...")
def start_api() -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    from meddoc_rag.config import PROJECT_ROOT, settings

    index_dir = settings.index_dir(settings.chunk_strategy)
    if not (index_dir / "index.faiss").exists():
        repo = os.environ.get("MEDDOC_INDEX_REPO", "yusuf469/meddoc-rag-index")
        token = os.environ.get("HF_TOKEN", "").strip()
        if not token:
            st.error("HF_TOKEN is missing from the app's secrets, so the private index cannot be downloaded.")
            st.stop()
        try:
            snapshot_download(repo, repo_type="dataset", local_dir=index_dir, token=token)
        except RepositoryNotFoundError:
            st.error(f"The HF_TOKEN in the app's secrets cannot read the private dataset `{repo}`. "
                     "Check the token is valid and has read access to it.")
            st.stop()

    import uvicorn

    server = uvicorn.Server(uvicorn.Config("app.api:app", host="127.0.0.1", port=8000, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(180):
        try:
            if requests.get(f"{API_URL}/health", timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError("API did not become healthy within 3 minutes")


start_api()
runpy.run_path(str(Path(__file__).parent / "app" / "ui.py"), run_name="__main__")
