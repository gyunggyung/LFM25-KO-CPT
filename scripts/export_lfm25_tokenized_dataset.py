#!/usr/bin/env python
"""Export packed tokenized LFM2.5 Korean CPT data as Parquet shards."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-jsonl",
        default="/home/work/.data/lfm2_ko_cpt/datasets/ko_cpt_mix_full_lfmstyle_20260627.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/work/.data/lfm2_ko_cpt/datasets/tokenized_lfm25_8k_20260628",
    )
    parser.add_argument("--model-path", default="LiquidAI/LFM2.5-8B-A1B")
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--shard-blocks", type=int, default=4096)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_shard(out_dir: Path, shard_index: int, blocks: list[list[int]]) -> dict:
    shard_name = f"part-{shard_index:05d}.parquet"
    shard_path = out_dir / shard_name
    table = pa.table(
        {
            "input_ids": pa.array(blocks, type=pa.list_(pa.uint32())),
            "length": pa.array([len(block) for block in blocks], type=pa.uint16()),
        }
    )
    pq.write_table(table, shard_path, compression="zstd", use_dictionary=False)
    return {
        "file": shard_name,
        "blocks": len(blocks),
        "tokens": sum(len(block) for block in blocks),
        "bytes": shard_path.stat().st_size,
    }


def iter_jsonl_batches(path: Path, batch_size: int, limit_rows: int):
    batch: list[str] = []
    rows = 0
    chars = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if limit_rows and rows >= limit_rows:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text")
            if not isinstance(text, str) or not text:
                continue
            batch.append(text)
            rows += 1
            chars += len(text)
            if len(batch) >= batch_size:
                yield batch, rows, chars
                batch = []
        if batch:
            yield batch, rows, chars


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise RuntimeError(f"{manifest_path} already exists; pass --overwrite to rebuild")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if not isinstance(eos_id, int) or eos_id < 0:
            eos_id = None

    buffer: list[int] = []
    shard_blocks: list[list[int]] = []
    shards: list[dict] = []
    shard_index = 0
    rows_seen = 0
    chars_seen = 0
    tokens_seen = 0
    blocks_seen = 0
    started = time.time()

    progress = tqdm(desc="tokenizing", unit="row")
    for batch, rows_seen, chars_seen in iter_jsonl_batches(input_path, args.batch_size, args.limit_rows):
        encoded = tokenizer(batch, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        for ids in encoded:
            buffer.extend(ids)
            if eos_id is not None:
                buffer.append(eos_id)
            while len(buffer) >= args.seq_length:
                block = buffer[: args.seq_length]
                del buffer[: args.seq_length]
                shard_blocks.append(block)
                blocks_seen += 1
                tokens_seen += args.seq_length
                if len(shard_blocks) >= args.shard_blocks:
                    shards.append(write_shard(out_dir, shard_index, shard_blocks))
                    shard_index += 1
                    shard_blocks = []
                    elapsed = max(time.time() - started, 1.0)
                    write_json(
                        out_dir / "manifest.progress.json",
                        {
                            "status": "running",
                            "rows_seen": rows_seen,
                            "chars_seen": chars_seen,
                            "tokens_seen": tokens_seen,
                            "blocks_seen": blocks_seen,
                            "shards": shards,
                            "tokens_per_second": tokens_seen / elapsed,
                        },
                    )
        progress.update(len(batch))
    progress.close()

    if shard_blocks:
        shards.append(write_shard(out_dir, shard_index, shard_blocks))
    remainder_tokens = len(buffer)
    elapsed = max(time.time() - started, 1.0)
    manifest = {
        "status": "complete",
        "model_path": args.model_path,
        "source_jsonl": str(input_path),
        "seq_length": args.seq_length,
        "rows_seen": rows_seen,
        "chars_seen": chars_seen,
        "tokens_seen": tokens_seen,
        "blocks_seen": blocks_seen,
        "remainder_tokens_dropped": remainder_tokens,
        "shards": shards,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens_seen / elapsed,
        "format": {
            "file_type": "parquet",
            "columns": {
                "input_ids": "list<uint32>; exactly seq_length tokens for every row",
                "length": "uint16; always seq_length for full blocks",
            },
            "document_separator": "tokenizer eos token when available",
        },
    }
    write_json(manifest_path, manifest)
    write_json(out_dir / "README.json", manifest)
    progress_path = out_dir / "manifest.progress.json"
    if progress_path.exists():
        progress_path.unlink()
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
