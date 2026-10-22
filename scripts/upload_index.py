"""Upload the served index to a private Hugging Face dataset, for the Streamlit Cloud deployment.

Usage:
    hf auth login                    # once, with a write token
    python scripts/upload_index.py --repo <hf-username>/meddoc-rag-index

Private because the chunks contain text from CC BY-NC-SA documents this repo does not
redistribute. Re-run after rebuilding the index; streamlit_app.py downloads it at startup.
"""

import argparse
import sys

from huggingface_hub import HfApi

from meddoc_rag.config import settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="dataset id, e.g. your-hf-username/meddoc-rag-index")
    args = parser.parse_args()

    index = settings.index_dir(settings.chunk_strategy)
    if not (index / "index.faiss").exists():
        print(f"missing {index}; run scripts/build_index.py first")
        return 1
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=index,
                      commit_message=f"{settings.chunk_strategy} index")
    print(f"Uploaded {index} to https://huggingface.co/datasets/{args.repo} (private)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
