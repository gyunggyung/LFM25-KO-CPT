#!/usr/bin/env python
"""Small vLLM smoke test for LFM2.5 KO CPT checkpoints.

This is not a benchmark score. It catches obvious regressions in Korean
formatting, domain behavior, English instruction following, and LFM tool-call
syntax before the longer lm-eval/vLLM suite runs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


SMOKE_CASES: list[dict[str, Any]] = [
    {
        "id": "ko_legal_source",
        "messages": [
            {
                "role": "system",
                "content": "한국어로 답하되, 법률 정보는 출처와 한계를 구분해서 설명한다.",
            },
            {
                "role": "user",
                "content": "상가 임대차에서 계약갱신요구권과 권리금 회수기회 보호를 구분해서 짧게 설명해줘.",
            },
        ],
        "checks": ["계약갱신", "권리금", "임대차"],
    },
    {
        "id": "ko_finance",
        "messages": [
            {
                "role": "system",
                "content": "한국어 금융 설명을 하되 투자 조언으로 오해되지 않게 한다.",
            },
            {
                "role": "user",
                "content": "기준금리 인상이 은행 순이자마진과 채권 가격에 미치는 영향을 비교해줘.",
            },
        ],
        "checks": ["기준금리", "순이자마진", "채권"],
    },
    {
        "id": "ko_wiki_summary",
        "messages": [
            {"role": "system", "content": "한국어로 간결하고 구조적으로 답한다."},
            {
                "role": "user",
                "content": "조선 후기 실학의 핵심 특징을 세 가지로 요약해줘.",
            },
        ],
        "checks": ["실학", "조선", "요약"],
    },
    {
        "id": "en_instruction_regression",
        "messages": [
            {
                "role": "user",
                "content": "Answer in exactly three bullet points. Each bullet must have four words.",
            },
        ],
        "checks": ["-"],
    },
    {
        "id": "tool_call_korean",
        "messages": [
            {
                "role": "system",
                "content": (
                    "List of tools: "
                    '[{"name":"search_law","description":"Search Korean law",'
                    '"parameters":{"type":"object","properties":{"query":{"type":"string"}},'
                    '"required":["query"]}}]. '
                    "Use a tool call if current legal text is needed."
                ),
            },
            {
                "role": "user",
                "content": "상가건물 임대차보호법 권리금 조항을 확인해줘.",
            },
        ],
        "checks": ["<|tool_call_start|>", "search_law"],
    },
]


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        rendered = ["<|startoftext|>"]
        for message in messages:
            rendered.append(
                f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            )
        rendered.append("<|im_start|>assistant\n")
        return "".join(rendered)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [render_prompt(tokenizer, case["messages"]) for case in SMOKE_CASES]
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        repetition_penalty=1.05,
    )
    generations = llm.generate(prompts, sampling)

    with output.open("w", encoding="utf-8") as f:
        for case, generation in zip(SMOKE_CASES, generations):
            text = generation.outputs[0].text
            checks = {
                needle: (needle.lower() in text.lower()) for needle in case["checks"]
            }
            record = {
                "model_name": args.model_name,
                "model": args.model,
                "case_id": case["id"],
                "prompt": prompts[SMOKE_CASES.index(case)],
                "response": text,
                "response_preview": normalize(text)[:500],
                "checks": checks,
                "all_checks_passed": all(checks.values()),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {output}")


if __name__ == "__main__":
    main()
