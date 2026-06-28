#!/usr/bin/env python
"""Upload the completed full CPT model and model card to Hugging Face."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def read_hf_token(root: Path) -> str:
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token
    for env_path in [root / ".env", root.parent / ".env"]:
        if not env_path.exists():
            continue
        for line in env_path.read_text(errors="ignore").splitlines():
            value = line.strip()
            if value.startswith("export "):
                value = value[len("export ") :].strip()
            if value.startswith("HF_TOKEN") and "=" in value:
                token = value.split("=", 1)[1].strip().strip("\"'")
                if token:
                    return token
    raise RuntimeError("HF_TOKEN missing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL")
    parser.add_argument("--root-dir", default="/home/work/.projects/LLM-OS-Models/Terminal")
    parser.add_argument(
        "--model-dir",
        default=(
            "/home/work/.data/lfm2_ko_cpt/models/"
            "LFM2.5-8B-A1B-KO-CPT-FULL-20260628_lfm25_8b_ko_cpt_full_lfmstyle/final_full"
        ),
    )
    parser.add_argument(
        "--model-card",
        default=(
            "/home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_cpt/"
            "model_cards/LFM2.5-8B-A1B-KO-CPT-FULL.md"
        ),
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--skip-weights", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root_dir)
    model_dir = Path(args.model_dir)
    model_card = Path(args.model_card)
    token = read_hf_token(root)

    if not model_card.exists():
        raise FileNotFoundError(f"model card missing: {model_card}")
    if not args.skip_weights and not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"completed model missing or incomplete: {model_dir}")

    api = HfApi(token=token)
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True, token=token)

    api.upload_file(
        path_or_fileobj=str(model_card),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        commit_message="Update model card",
    )

    if not args.skip_weights:
        api.upload_folder(
            folder_path=str(model_dir),
            path_in_repo=".",
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.revision,
            commit_message=f"Upload full CPT model from {model_dir.name}",
            ignore_patterns=[
                "optimizer.pt",
                "scheduler.pt",
                "rng_state*.pth",
                "trainer_state.json",
                "training_args.bin",
                "*.tmp",
            ],
        )

    info = api.model_info(args.repo_id)
    print(f"uploaded https://huggingface.co/{args.repo_id}")
    print(f"sha {info.sha}")


if __name__ == "__main__":
    main()
