#!/usr/bin/env python3

"""Upload the Tianwang evaluation directory to Hugging Face Hub."""

import argparse
from pathlib import Path
from typing import Optional


DEFAULT_REPO_ID = "MewtwoX23/tianwang_clip"
DEFAULT_SOURCE_DIR = Path("evaluation/tianwang_exp")


def upload_directory(
    source_dir: Path,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload Tianwang evaluation results",
) -> None:
    """Create the model repository if needed and upload source_dir to its root."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required; install it with `pip install huggingface_hub`"
        ) from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(source_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload evaluation/tianwang_exp to Hugging Face Hub."
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"directory to upload (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face model repository (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--token",
        help="Hugging Face access token; otherwise use HF_TOKEN or the local CLI login",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="create the repository as private if it does not exist",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload Tianwang evaluation results",
        help="commit message for the upload",
    )
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        parser.error(f"source directory does not exist: {args.source_dir}")

    try:
        upload_directory(
            source_dir=args.source_dir,
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
            commit_message=args.commit_message,
        )
    except RuntimeError as exc:
        parser.error(str(exc))

    print(f"Uploaded {args.source_dir} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
