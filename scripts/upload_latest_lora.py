#!/usr/bin/env python
"""Upload the latest LoRA adapter checkpoint/final adapter to Hugging Face."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-id", default="LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-LoRA")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--revision", default=None)
    return parser.parse_args()


def checkpoint_key(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def latest_adapter_dir(output_dir: Path) -> Path:
    final_dir = output_dir / "final_lora"
    if (final_dir / "adapter_config.json").exists():
        return final_dir
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=checkpoint_key)
    for checkpoint in reversed(checkpoints):
        if (checkpoint / "adapter_config.json").exists():
            return checkpoint
    raise FileNotFoundError(f"No LoRA adapter found under {output_dir}")


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required.")

    source_dir = latest_adapter_dir(Path(args.output_dir))
    api = HfApi(token=token)
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True, token=token)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(source_dir),
        path_in_repo=".",
        revision=args.revision,
        commit_message=f"Upload LoRA adapter from {source_dir.name}",
        ignore_patterns=[
            "optimizer.pt",
            "scheduler.pt",
            "rng_state*.pth",
            "trainer_state.json",
            "training_args.bin",
        ],
    )
    print(f"uploaded {source_dir} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

