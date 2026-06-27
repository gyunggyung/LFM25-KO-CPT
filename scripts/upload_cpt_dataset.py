#!/usr/bin/env python
"""Upload prepared LFM2 Korean CPT dataset artifacts to Hugging Face."""

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
    parser.add_argument("--repo-id", default="LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-DATA")
    parser.add_argument("--root-dir", default="/home/work/.projects/LLM-OS-Models/Terminal")
    parser.add_argument("--data-root", default="/home/work/.data/lfm2_ko_cpt/datasets")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--skip-corpus", action="store_true")
    return parser.parse_args()


def upload(api: HfApi, repo_id: str, local: Path, remote: str, message: str) -> None:
    if not local.exists():
        print(f"missing {local}", flush=True)
        return
    print(f"uploading {local} -> {remote}", flush=True)
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=remote,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=message,
    )


def main() -> None:
    args = parse_args()
    root = Path(args.root_dir)
    data_root = Path(args.data_root)
    token = read_hf_token(root)
    api = HfApi(token=token)
    create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True, token=token)

    upload(
        api,
        args.repo_id,
        root / "lfm2_ko_cpt/dataset_cards/LFM2.5-8B-A1B-KO-CPT-DATA.md",
        "README.md",
        "Add dataset card",
    )
    upload(
        api,
        args.repo_id,
        root / "lfm2_ko_cpt/configs/ko_cpt_sources_full_20260627.json",
        "metadata/ko_cpt_sources_full_20260627.json",
        "Add source configuration",
    )
    upload(
        api,
        args.repo_id,
        data_root / "ko_cpt_mix_full_lfmstyle_20260627.stats.json",
        "metadata/ko_cpt_mix_full_lfmstyle_20260627.stats.json",
        "Add corpus statistics",
    )
    upload(
        api,
        args.repo_id,
        data_root / "ko_cpt_mix_full_lfmstyle_20260627.stats.json.full_report.json",
        "metadata/ko_cpt_mix_full_lfmstyle_20260627.stats.json.full_report.json",
        "Add preprocessing report",
    )
    if not args.skip_corpus:
        upload(
            api,
            args.repo_id,
            data_root / "ko_cpt_mix_full_lfmstyle_20260627.jsonl",
            "data/ko_cpt_mix_full_lfmstyle_20260627.jsonl",
            "Add prepared full LFM-style CPT corpus",
        )

    info = api.dataset_info(args.repo_id)
    print(f"uploaded https://huggingface.co/datasets/{args.repo_id}", flush=True)
    print(f"sha {info.sha}", flush=True)


if __name__ == "__main__":
    main()
