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

# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "peft",
#     "trackio",
# ]
# ///

# docstyle-ignore
"""
# TrOPD outlier-FKL masking A/B study for VLMs (Qwen3-VL-8B -> Qwen3-VL-2B)
#
# Runs two same-family, full on-policy (lmbda=1.0) reverse-KL (beta=1.0) distillation runs that differ
# ONLY in `use_outlier_fkl_loss`:
#   Run 1 (baseline): use_outlier_fkl_loss=False -> clean reverse-KL OPD.
#   Run 2 (TrOPD):    use_outlier_fkl_loss=True  -> trust-region reverse KL + teacher top-k forward KL on
#                     outlier tokens (https://huggingface.co/papers/2606.01249, Eq. 5-7).
# Same architecture and tokenizer, so the standard (matched-tokenizer) JSD/RKL path is used. vLLM is
# enabled for faster on-policy generation.
#
# With vLLM colocate, run each variant in its OWN process so the vLLM engine fully releases the GPU
# before the next variant loads its models (running both in one process OOMs on the second load):

accelerate launch trl/experimental/gold/gold_vlm_tropd.py \
    --student_model_name Qwen/Qwen3-VL-2B-Instruct \
    --teacher_model_name Qwen/Qwen3-VL-8B-Instruct \
    --variant baseline

accelerate launch trl/experimental/gold/gold_vlm_tropd.py \
    --student_model_name Qwen/Qwen3-VL-2B-Instruct \
    --teacher_model_name Qwen/Qwen3-VL-8B-Instruct \
    --variant tropd

# Longer runs (separate run names/output dirs, sparser eval/logging, checkpoints every 250 steps):
# add `--max_steps 1000` to both commands. The `train/outlier_token_frac` metric shows how often the
# TrOPD trust region (would have) triggered — if it stays ~0, both variants are equivalent by design.
"""

import argparse
import gc

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor

from trl.experimental.gold import GOLDConfig, GOLDTrainer


SYSTEM_PROMPT = """
Answer the question by briefly explaining the reasoning behind your answer.
Return the final answer as a single number followed immediately by the ° symbol.
"""


def normalize_solution(solution):
    solution = str(solution).replace("<answer>", "").replace("</answer>", "").strip()
    if solution and not solution.endswith("°"):
        solution = f"{solution}°"
    return solution


def make_conversation(example):
    """Convert GEOQA_R1V row into the chat format expected by TRL VLM trainers."""
    return {
        "prompt": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": example["problem"]},
                ],
            },
        ],
        "completion": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": normalize_solution(example["solution"])}],
            },
        ],
        "image": example["image"],
    }


def filter_big_images(example):
    image = example["image"]
    return image.size[0] < 512 and image.size[1] < 512


def convert_to_rgb(example):
    image = example["image"]
    if image.mode != "RGB":
        image = image.convert("RGB")
    example["image"] = image
    return example


def run(cli_args, train_dataset, eval_dataset, use_outlier_fkl_loss):
    """Run one distillation variant. Everything is identical across the two calls except
    `use_outlier_fkl_loss`, isolating the effect of TrOPD outlier masking."""
    # ──────────────────────────────────────────────
    # Models (reloaded per run so the two variants start from the same initialization)
    # ──────────────────────────────────────────────
    student_model = AutoModelForImageTextToText.from_pretrained(
        cli_args.student_model_name, torch_dtype=torch.bfloat16
    )
    teacher_model = AutoModelForImageTextToText.from_pretrained(
        cli_args.teacher_model_name, torch_dtype=torch.bfloat16
    )

    # Freeze everything except the language model head
    for name, param in student_model.named_parameters():
        if "language_model" not in name:
            param.requires_grad = False

    processor = AutoProcessor.from_pretrained(cli_args.student_model_name, padding_side="left")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=r"^.*language_model.*\.(q_proj|k_proj|v_proj)$",
    )

    variant = "tropd" if use_outlier_fkl_loss else "baseline"
    student_short = cli_args.student_model_name.split("/")[-1]
    teacher_short = cli_args.teacher_model_name.split("/")[-1]
    run_name = f"gold-vlm-{student_short}-from-{teacher_short}-{variant}"
    if cli_args.max_steps != 100:
        run_name += f"-{cli_args.max_steps}steps"  # keep the original 100-step run dirs/wandb names intact

    # Schedule knobs scale with run length: the 100-step A/B keeps its original cadence; longer runs
    # evaluate/print less often and keep intermediate checkpoints (to eval both variants at the same
    # step if one destabilizes late).
    long_run = cli_args.max_steps > 100

    args = GOLDConfig(
        output_dir=run_name,
        run_name=run_name,
        # GOLD-specific: full on-policy (lmbda=1.0), reverse-KL distillation (beta=1.0)
        lmbda=1.0,
        beta=1.0,
        temperature=0.6,
        max_completion_length=1024,
        max_grad_norm=1.0,
        teacher_model_name_or_path=cli_args.teacher_model_name,
        num_generations=1,
        use_uld_loss=False,
        # TrOPD outlier-FKL masking (the only setting that differs between the two runs)
        use_outlier_fkl_loss=use_outlier_fkl_loss,
        outlier_fkl_top_k=64,
        # vLLM
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.5,
        vllm_max_model_length=2048,
        max_length=3072,
        # Training schedule
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        max_steps=cli_args.max_steps,
        learning_rate=1e-4,
        warmup_steps=10,
        save_steps=250 if long_run else 500,  # 500 is the HF default, never reached in 100-step runs
        # Evaluation
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        eval_steps=100 if long_run else 25,
        # Precision
        bf16=True,
        # Logging
        logging_steps=10,
        log_completions=True,
        log_completions_steps=50 if long_run else 10,  # default 100 would only print once per 100 steps
        report_to="wandb",
    )

    trainer = GOLDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    # Close the wandb run so the next variant starts a fresh run instead of logging into this one
    # (both variants run in the same process, and the Trainer does not finish the run on its own).
    if trainer.accelerator.is_main_process:
        import wandb

        if wandb.run is not None:
            wandb.finish()

    # Free the GPU memory held by this variant (models, optimizer, vLLM) before the next run starts.
    del trainer, student_model, teacher_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--teacher_model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Training steps. Runs longer than 100 steps switch to a sparser eval/logging cadence and keep "
        "intermediate checkpoints every 250 steps.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["baseline", "tropd", "both"],
        default="both",
        help="Which variant(s) to run. With vLLM colocate, prefer running 'baseline' and 'tropd' as two "
        "separate launches so the vLLM engine releases the GPU between runs; 'both' runs them in one process "
        "and may OOM on the second model load.",
    )
    cli_args = parser.parse_args()

    # ──────────────────────────────────────────────
    # Dataset (built once, shared by both runs)
    # ──────────────────────────────────────────────
    dataset = load_dataset("leonardPKU/GEOQA_R1V_Train_8K", split="train")
    dataset = dataset.filter(filter_big_images)
    dataset = dataset.map(convert_to_rgb)
    dataset = dataset.map(make_conversation)

    # Hold out 5% for evaluation
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # ──────────────────────────────────────────────
    # A/B: baseline reverse-KL OPD vs. + TrOPD outlier masking
    # ──────────────────────────────────────────────
    variants = {"baseline": [False], "tropd": [True], "both": [False, True]}[cli_args.variant]
    for use_outlier_fkl_loss in variants:
        run(cli_args, train_dataset, eval_dataset, use_outlier_fkl_loss)
