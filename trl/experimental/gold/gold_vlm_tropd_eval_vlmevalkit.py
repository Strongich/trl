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
# MathVista / MathVision benchmark for the TrOPD A/B checkpoints (VLMEvalKit + vLLM, 3 seeds)
#
# Merges each LoRA adapter into the base model, then for every (model, seed) pair runs VLMEvalKit's
# run.py in its OWN subprocess (fresh vLLM engine, GPU fully released between runs) via a config
# JSON that loads the merged checkpoint with `Qwen3VLChat(use_vllm=True)`.
#
# Seeds: VLMEvalKit's Qwen3VLChat hardcodes `seed=0` in its vLLM engine and its default decoding is
# near-greedy (temperature=0.01), so re-running it 3 times would give 3 identical numbers. This
# script therefore (a) patches `vllm.LLM` in the worker so each run gets its own engine seed, and
# (b) defaults to stochastic decoding (temperature=0.6, top_p=0.95) so the seeds actually vary.
# Note this measures *decoding* variance; training-seed variance requires retraining.
#
# Assumes a VLMEvalKit clone is set up on this machine (deps installed, `.env` with OPENAI_API_KEY
# if you want the LLM judge for answer extraction — otherwise VLMEvalKit falls back to heuristic
# matching; either is fine as long as it is identical for all models). Datasets download to
# $LMUData (default ~/LMUData).

python trl/experimental/gold/gold_vlm_tropd_eval_vlmevalkit.py \
    --vlmevalkit_dir ~/VLMEvalKit \
    --base_model Qwen/Qwen3-VL-2B-Instruct \
    --adapter_dirs gold-vlm-Qwen3-VL-2B-Instruct-from-Qwen3-VL-8B-Instruct-baseline \
                   gold-vlm-Qwen3-VL-2B-Instruct-from-Qwen3-VL-8B-Instruct-tropd \
    --data MathVista_MINI MathVision_MINI \
    --seeds 0 1 2
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys


# VLMEvalKit dataset name -> (dataset class in vlmeval/dataset, dataset kwarg)
DATA_CLASSES = {
    "MathVista_MINI": ("MathVista", "MathVista_MINI"),
    "MathVision_MINI": ("MathVision", "MathVision_MINI"),
    "MathVision": ("MathVision", "MathVision"),
}


def merge_adapter(base_model, adapter_dir):
    """Merge a LoRA adapter into the base model so vLLM can load it as a plain checkpoint."""
    merged_dir = adapter_dir.rstrip("/") + "-merged"
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


def worker(args):
    """Run VLMEvalKit's run.py for one config, with the vLLM engine seeded per run.

    Qwen3VLChat hardcodes LLM(seed=0), so we patch vllm.LLM *before* vlmeval is imported: any
    engine created during this run uses our seed instead.
    """
    import random

    import numpy as np
    import torch
    import vllm

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    seed = args.seed
    original_llm = vllm.LLM

    class SeededLLM(original_llm):
        def __init__(self, *a, **kw):
            kw["seed"] = seed
            super().__init__(*a, **kw)

    vllm.LLM = SeededLLM

    import runpy

    sys.argv = ["run.py", "--config", args.config, "--work-dir", args.work_dir, "--verbose"]
    runpy.run_path(os.path.join(args.vlmevalkit_dir, "run.py"), run_name="__main__")


def aggregate(work_dir, model_names, seeds):
    """Print per-seed scores and mean/std per (model, dataset) from VLMEvalKit's result csvs."""
    import pandas as pd

    print("\n=== VLMEvalKit results ===")
    for name in model_names:
        scores = {}  # dataset -> list of overall scores across seeds
        for seed in seeds:
            run_name = f"{name}-seed{seed}"
            for csv in sorted(glob.glob(os.path.join(work_dir, run_name, "*.csv"))):
                df = pd.read_csv(csv)
                dataset = os.path.basename(csv).replace(f"{run_name}_", "").rsplit(".", 1)[0]
                # Overall score: "Overall" row (MathVista/MathVision report task splits as rows)
                # or "Overall" column, depending on the dataset's csv layout.
                overall = None
                for col in df.columns:
                    if col.lower() in ("task&skill", "task", "split", "category"):
                        row = df[df[col].astype(str).str.lower() == "overall"]
                        if len(row):
                            num = row.select_dtypes("number")
                            if num.shape[1]:
                                overall = float(num.iloc[0, -1])
                        break
                if overall is None and "Overall" in df.columns:
                    overall = float(df["Overall"].iloc[0])
                if overall is not None:
                    scores.setdefault(dataset, []).append(overall)
                print(f"  {run_name}/{os.path.basename(csv)}")
        for dataset, values in scores.items():
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            print(f"{name:60s} {dataset:40s} {mean:6.2f} ± {std:.2f}  (n={len(values)}: {values})")


def main(args):
    vlmevalkit_dir = os.path.abspath(os.path.expanduser(args.vlmevalkit_dir))
    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    models = {}  # display name -> merged checkpoint path
    for adapter_dir in args.adapter_dirs:
        models[os.path.basename(adapter_dir.rstrip("/"))] = merge_adapter(args.base_model, adapter_dir)
    if args.include_base:
        models[args.base_model.split("/")[-1] + "-nodistill"] = args.base_model

    data_section = {}
    for dataset in args.data:
        cls, dataset_kwarg = DATA_CLASSES[dataset]
        data_section[dataset] = {"class": cls, "dataset": dataset_kwarg}

    for name, path in models.items():
        for seed in args.seeds:
            run_name = f"{name}-seed{seed}"
            config = {
                "model": {
                    run_name: {
                        "class": "Qwen3VLChat",
                        "model_path": path,
                        "use_vllm": True,
                        "max_new_tokens": 1024,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                    }
                },
                "data": data_section,
            }
            config_path = os.path.join(work_dir, f"{run_name}.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            print(f"\n=== Running {run_name} on {args.data} ===")
            # Fresh subprocess per (model, seed): vLLM releases the GPU between runs, and the
            # worker seeds the engine before vlmeval is imported. cwd = VLMEvalKit so its .env
            # (judge keys) is picked up.
            subprocess.run(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--worker",
                    "--seed",
                    str(seed),
                    "--config",
                    config_path,
                    "--work_dir",
                    work_dir,
                    "--vlmevalkit_dir",
                    vlmevalkit_dir,
                ],
                check=True,
                cwd=vlmevalkit_dir,
            )

    aggregate(work_dir, list(models.keys()), args.seeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlmevalkit_dir", type=str, default="VLMEvalKit", help="Path to the VLMEvalKit clone.")
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
    parser.add_argument("--data", type=str, nargs="+", default=["MathVista_MINI", "MathVision_MINI"], choices=list(DATA_CLASSES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature (>0 so seeds differ).")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--work_dir", type=str, default="vlmevalkit_results")
    # Worker mode (internal): run one (model, seed) config in a fresh process.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--config", type=str, help=argparse.SUPPRESS)
    cli_args = parser.parse_args()

    if cli_args.worker:
        worker(cli_args)
    else:
        main(cli_args)
