# MedDoc RAG

Citation-grounded question answering over public healthcare guidelines (WHO, CDC, NIH),
with a measurable retrieval-evaluation harness.

> Work in progress. Built milestone by milestone; see commit history.

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows (source .venv/bin/activate on Linux/macOS)
pip install -e ".[dev]"
python scripts/download_docs.py
pytest
```

**Disclaimer:** for educational purposes only. Not medical advice.
