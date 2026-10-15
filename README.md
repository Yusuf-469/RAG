# MedDoc RAG

Citation-grounded question answering over public healthcare guidelines (WHO, CDC, NIH),
with a measurable retrieval-evaluation harness.

> Work in progress. Built milestone by milestone; see commit history.

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows (source .venv/bin/activate on Linux/macOS)
pip install -e ".[dev,ui]"
python scripts/download_docs.py      # CDC PDFs need a manual browser download (see printed steps)
python scripts/build_chunks.py && python scripts/build_index.py
copy .env.example .env               # then add your free GROQ_API_KEY
```

## Run

```bash
uvicorn app.api:app --port 8000      # API; interactive docs at http://127.0.0.1:8000/docs
streamlit run app/ui.py              # UI at http://127.0.0.1:8501
python scripts/ask.py "How much physical activity should older adults do?"   # CLI
pytest
```

**Disclaimer:** for educational purposes only. Not medical advice.
