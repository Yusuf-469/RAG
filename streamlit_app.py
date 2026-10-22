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

import logging
import os
import runpy
import socket
import sys
import threading
import time
from pathlib import Path

import requests
import streamlit as st

# Secrets must be in the environment before meddoc_rag reads its settings at import time.
def _export_secrets(section) -> None:
    """Copy string secrets to env vars, including ones placed under a [section] header."""
    for key, value in section.items():
        if isinstance(value, str):
            os.environ.setdefault(key.upper(), value.strip())
        elif hasattr(value, "items"):
            _export_secrets(value)


try:
    _export_secrets(st.secrets)
except FileNotFoundError:   # no secrets file: running locally with .env
    pass

class _ErrorLog(logging.Handler):
    """Keeps uvicorn's error records, so a failed startup can be shown instead of a bare timeout."""

    def __init__(self) -> None:
        super().__init__(logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(self.format(record))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@st.cache_resource(show_spinner="Starting up: downloading the index and loading models (about a minute)...")
def start_api() -> str:
    """Start the API once per app process; returns its URL."""
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

    from app.api import app   # imported here so import errors surface with their real traceback

    # A free port, not a fixed one: the host may already use 8000, and a failed bind only logs.
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info")
    errors = _ErrorLog()   # attached after Config, which reconfigures uvicorn's loggers
    logging.getLogger("uvicorn.error").addHandler(errors)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 600   # model downloads on a cold, shared CPU can be slow
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{url}/health", timeout=2).ok:
                return url
        except requests.RequestException:
            pass
        if not thread.is_alive():
            break
        time.sleep(1)
    detail = "\n".join(errors.messages) or ("server thread exited" if not thread.is_alive() else "timed out after 10 minutes")
    st.error(f"The API failed to start (Python {sys.version.split()[0]}):")
    st.code(detail)
    st.stop()


os.environ["MEDDOC_API_URL"] = start_api()
runpy.run_path(str(Path(__file__).parent / "app" / "ui.py"), run_name="__main__")
