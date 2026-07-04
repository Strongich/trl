# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# docstyle-ignore
"""
# GEOQA holdout accuracy for the TrOPD A/B checkpoints (vLLM, greedy)
#
# Rebuilds the exact 5% holdout split used by gold_vlm_tropd.py, merges each LoRA adapter into the
# base model, then evaluates every model with vLLM greedy decoding and reports exact-match accuracy
# on the final `<number>°` answer. Each model is evaluated in its OWN subprocess so the vLLM engine
# fully releases the GPU between models.

python trl/experimental/gold/gold_vlm_tropd_eval_geoqa.py \
    --base_model Qwen/Qwen3-VL-2B-Instruct \
    --adapter_dirs gold-vlm-Qwen3-VL-2B-Instruct-from-Qwen3-VL-8B-Instruct-baseline \
                   gold-vlm-Qwen3-VL-2B-Instruct-from-Qwen3-VL-8B-Instruct-tropd \
    --include_base
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys

from datasets import load_dataset


# ── Keep in sync with gold_vlm_tropd.py: the split below must match the training holdout ──
SYSTEM_PROMPT = """
Answer the question by briefly explaining the reasoning behind your answer.
Return the final answer as a single number followed immediately by the ° symbol.
"""


def normalize_solution(solution):
    solution = str(solution).replace("<answer>", "").replace("</answer>", "").strip()
    if solution and not solution.endswith("°"):
        solution = f"{solution}°"
    return solution


def filter_big_images(example):
    image = example["image"]
    return image.size[0] < 512 and image.size[1] < 512


def load_eval_dataset():
    """Rebuild the exact eval split of gold_vlm_tropd.py (same filter, same seed, same test_size)."""
    dataset = load_dataset("leonardPKU/GEOQA_R1V_Train_8K", split="train")
    dataset = dataset.filter(filter_big_images)
    return dataset.train_test_split(test_size=0.05, seed=42)["test"]


# ──────────────────────────────────────────────


def extract_number(text):
    """Extract the predicted answer: last number followed by °, else the last number in the text."""
    matches = re.findall(r"(-?\d+(?:\.\d+)?)\s*°", text)
    if not matches:
        matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(matches[-1]) if matches else None


def worker(args):
    """Evaluate a single model. Runs in its own subprocess so vLLM releases the GPU on exit."""
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    dataset = load_eval_dataset()
    processor = AutoProcessor.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 1},
    )

    llm_inputs = []
    targets = []
    for example in dataset:
        image = example["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": example["problem"]}]},
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        llm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
        targets.append(extract_number(normalize_solution(example["solution"])))

    outputs = llm.generate(llm_inputs, SamplingParams(temperature=0.0, max_tokens=1024))

    results = []
    correct = 0
    for output, target in zip(outputs, targets):
        completion = output.outputs[0].text
        prediction = extract_number(completion)
        is_correct = (
            prediction is not None
            and target is not None
            and math.isclose(prediction, target, rel_tol=1e-4, abs_tol=1e-2)
        )
        correct += is_correct
        results.append({"target": target, "prediction": prediction, "correct": is_correct, "completion": completion})

    summary = {
        "model_path": args.model_path,
        "num_examples": len(results),
        "num_correct": correct,
        "accuracy": correct / len(results),
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{args.model_path}] accuracy: {correct}/{len(results)} = {summary['accuracy']:.4f}")


def merge_adapter(base_model, adapter_dir):
    """Merge a LoRA adapter into the base model so vLLM can load it as a plain checkpoint."""
    merged_dir = os.path.abspath(adapter_dir.rstrip("/") + "-merged")
    if os.path.isdir(merged_dir):
        print(f"Reusing existing merged model: {merged_dir}")
        return merged_dir
    print(f"Merging {adapter_dir} into {base_model} -> {merged_dir}")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    base = AutoModelForImageTextToText.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(merged_dir)
    AutoProcessor.from_pretrained(adapter_dir).save_pretrained(merged_dir)
    del base, merged
    return merged_dir


def main(args):
    models = []  # (display name, path loadable by vLLM)
    if args.include_base:
        models.append((args.base_model.split("/")[-1] + " (no distill)", args.base_model))
    for adapter_dir in args.adapter_dirs:
        models.append((os.path.basename(adapter_dir.rstrip("/")), merge_adapter(args.base_model, adapter_dir)))

    os.makedirs(args.out_dir, exist_ok=True)
    summaries = []
    for name, path in models:
        out = os.path.join(args.out_dir, re.sub(r"[^\w.-]", "_", name) + ".json")
        subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", "--model_path", path, "--out", out],
            check=True,
        )
        with open(out) as f:
            summaries.append((name, json.load(f)))

    print("\n=== GEOQA holdout accuracy (greedy) ===")
    for name, summary in summaries:
        print(f"{name:60s} {summary['num_correct']:4d}/{summary['num_examples']:4d} = {summary['accuracy']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument(
        "--adapter_dirs",
        type=str,
        nargs="+",
        default=[
            "gold-vlm-Qwen3-VL-2B-Instruct-from-Qwen3-VL-8B-Instruct-baseline",
            "gold-vlm-Qwen3-VL-2B-Instruct-from-Qwen3-VL-8B-Instruct-tropd",
        ],
        help="Output dirs of gold_vlm_tropd.py runs (each contains the final LoRA adapter).",
    )
    parser.add_argument("--include_base", action="store_true", help="Also evaluate the un-distilled base model.")
    parser.add_argument("--out_dir", type=str, default="geoqa_eval_results")
    # Worker mode (internal): evaluate a single model in a fresh process.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model_path", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=str, help=argparse.SUPPRESS)
    cli_args = parser.parse_args()

    if cli_args.worker:
        worker(cli_args)
    else:
        main(cli_args)
