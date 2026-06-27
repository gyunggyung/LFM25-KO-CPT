#!/usr/bin/env python
"""LoRA CPT for LiquidAI/LFM2.5 on Korean text-completion data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import unsloth  # noqa: F401 - must be imported before transformers.
import torch
from datasets import load_dataset
from huggingface_hub import login
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="LiquidAI/LFM2.5-8B-A1B")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--dataset-num-proc", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--dataloader-num-workers", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--target-modules", default="all-linear")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--hub-private", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def parse_target_modules(value: str) -> str | list[str]:
    value = value.strip()
    if value == "all-linear":
        return value
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if args.push_to_hub and local_rank == 0:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required for --push-to-hub.")
        login(token=token, add_to_git_credential=False)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16 if is_bfloat16_supported() else None,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=False,
        trust_remote_code=True,
        device_map={"": f"cuda:{local_rank}"} if torch.cuda.is_available() else None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=parse_target_modules(args.target_modules),
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    dataset = load_dataset("json", data_files={"train": args.dataset_path}, split="train")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if local_rank == 0:
        run_meta = {
            "model_path": args.model_path,
            "dataset_path": args.dataset_path,
            "output_dir": str(output_dir),
            "world_size": world_size,
            "max_seq_length": args.max_seq_length,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_sequences": world_size
            * args.per_device_train_batch_size
            * args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "max_steps": args.max_steps,
            "num_train_epochs": args.num_train_epochs,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "target_modules": args.target_modules,
            "load_in_4bit": args.load_in_4bit,
            "dataset_rows": len(dataset),
            "push_to_hub": args.push_to_hub,
            "hub_model_id": args.hub_model_id,
        }
        (output_dir / "run_meta.json").write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(run_meta, ensure_ascii=False, indent=2), flush=True)

    train_args = SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        packing=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        optim="adamw_8bit",
        report_to="none",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=True if world_size > 1 else None,
        dataset_num_proc=args.dataset_num_proc,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        hub_private_repo=args.hub_private if args.push_to_hub else None,
        hub_strategy="every_save" if args.push_to_hub else "checkpoint",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=True,
        args=train_args,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = output_dir / "final_lora"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    if local_rank == 0:
        print(f"saved_final_lora={final_dir}", flush=True)


if __name__ == "__main__":
    main()

