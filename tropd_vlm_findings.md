# TrOPD outlier-FKL masking for VLM on-policy distillation — findings

**Method:** Trust Region On-Policy Distillation (TrOPD), [paper 2606.01249](https://huggingface.co/papers/2606.01249), Eq. 5–7 (on-policy part only; off-policy guidance Eq. 8–9 excluded).
**Proposal:** [huggingface/trl#5933](https://github.com/huggingface/trl/discussions/5933).
**Implementation:** `use_outlier_fkl_loss` / `outlier_fkl_top_k` in `GOLDConfig`, `GOLDTrainer.outlier_fkl_loss`, branch `tropd-opd-update` ([Strongich/trl](https://github.com/Strongich/trl)).

## TL;DR

In a stable VLM LoRA distillation regime, TrOPD's outlier masking is **neutral-to-slightly-negative**: it
actively rewrote the loss on ~12% of tokens yet never beat the plain reverse-KL baseline on any metric,
with the deficit *growing* with training length (−1.7 pts MathVista at 500 steps). The trust region fired
throughout (this was not a no-op), the baseline eliminated outlier tokens *faster* than the masking did,
and the TrOPD student drifted off-calibration on off-policy text. Recommendation: ship as an **opt-in
flag**, do not make it a default; the method's target regime (unstable training: high LR, long horizons,
full finetuning) remains untested here.

## Implementation summary

- Per-token trust gate `M ~ Bernoulli(min(π_T(x)/π_S(x), 1))` on the sampled token (Eq. 6 — the
  speculative-decoding acceptance probability). Trust tokens train with **full-vocab reverse KL**
  (identical to the baseline's per-token term, for the cleanest A/B; the paper's k1 estimator is a
  memory optimization we don't need since GOLD already materializes full logits). Outlier tokens train
  with **teacher top-k forward KL** (Eq. 7, `top_k=64`).
- Matched-tokenizer path only (`use_uld_loss=False`), incompatible with the Liger fused path (needs full
  logits). Config validation enforces both.
- Unit tests verify the limits: trust mask forced to 1 ⇒ exactly `generalized_jsd_loss(beta=1.0)`
  (reverse KL); forced to 0 ⇒ exactly the top-k FKL term.
- Diagnostic metric `outlier_token_frac` = `E[1 − min(π_T/π_S, 1)]` over completion tokens, logged in
  **both** variants (realized routing fraction for TrOPD, counterfactual for the baseline), so the two
  runs log a directly comparable signal.

## Experimental setup

| | |
|---|---|
| Student / teacher | Qwen3-VL-2B-Instruct ← Qwen3-VL-8B-Instruct (same tokenizer) |
| Data | GEOQA_R1V_Train_8K, images < 512px, 95/5 train/holdout split (seed 42) |
| Objective | Full on-policy (`lmbda=1.0`), reverse KL (`beta=1.0`), generation temperature 0.6, `max_completion_length=1024` |
| Adapter | LoRA r=16, α=32, dropout 0.05, q/k/v of the language model only |
| Optimization | Effective batch 16 (2 × grad-accum 8), LR 1e-4, 10 warmup steps, bf16, vLLM colocate |
| A/B | The **only** difference between variants is `use_outlier_fkl_loss` (`outlier_fkl_top_k=64`) |
| Runs | 100 steps (v1) and 500 steps (v2); one training seed each |

**Evaluation:**
- **GEOQA holdout** (in-domain): 374 items, vLLM greedy decoding, exact match on the final `<number>°`.
- **MathVista_MINI / MathVision_MINI** (out-of-domain): VLMEvalKit (`Qwen3VLChat`, vLLM backend),
  3 decoding seeds (temperature 0.6, top-p 0.95; the vLLM engine seed is patched per run since
  VLMEvalKit hardcodes `seed=0`), GPT-4o-mini answer-extraction judge. ± values are std over the
  3 decoding seeds — they capture decoding variance only, not training-seed or item-sampling variance.

## Results

### 100 steps (v1)

| Model | GEOQA holdout | MathVista_MINI | MathVision_MINI |
|---|---|---|---|
| Base 2B (no distillation) | 33.96% (127/374) | — | — |
| Baseline (reverse-KL OPD) | **36.63%** (137/374) | **51.33 ± 0.05** | **20.18 ± 0.41** |
| TrOPD | 34.49% (129/374) | 51.27 ± 0.79 | 18.97 ± 0.86 |

Paired McNemar on GEOQA (same 374 items; discordant counts, exact two-sided p):

| Comparison | Discordant | p |
|---|---|---|
| base vs baseline | 32 vs 42 | 0.295 |
| base vs tropd | 36 vs 38 | 0.908 |
| baseline vs tropd | 37 vs 29 | 0.389 |

Nothing is significant at 100 steps — including distillation itself. The intervention was below the
detection floor of a 374-item eval (per-run binomial noise ≈ 2.5 pts).

### 500 steps (v2)

| Model | GEOQA holdout | MathVista_MINI | MathVision_MINI |
|---|---|---|---|
| Baseline (reverse-KL OPD) | **37.43%** (140/374) | **52.20 ± 0.91** | **20.50 ± 0.16** |
| TrOPD | 36.63% (137/374) | 50.47 ± 0.25 | 19.52 ± 1.09 |

- **MathVista** is the first gap that clears decoding noise: Δ = 1.73 ≈ 3× the standard error implied by
  the seed spreads (≈ 0.55). The variants also moved in *opposite directions* from 100 → 500 steps
  (baseline +0.9, TrOPD −0.8).
- **GEOQA**: net difference is 3 items; even the most extreme pairing (3 vs 0 discordant) gives exact
  McNemar p = 0.25, so this comparison cannot be significant under any pairing.
  <!-- TODO: exact 500-step McNemar from the per-example JSONs (run experiment.py on the VM with the
  *-v2-500steps result files): baseline vs tropd — n01 __ vs n10 __, p = __ -->
- Every baseline-vs-TrOPD comparison across both scales has the same sign (baseline ≥ TrOPD, 6/6).

### Training curves (caveat)

TrOPD showed a *lower* train/eval loss than the baseline in wandb. This is **not evidence of a better
student**: the two runs optimize different objectives, and the top-k FKL term is mechanically smaller
than the full-vocab reverse-KL terms it replaces (it is computed over 64 tokens and specifically
replaces the largest-magnitude contributions). Loss curves across the variants are not commensurable;
only the task metrics above are.

### Diagnostics: `outlier_token_frac`

- **Train (on-policy tokens): ~0.12–0.13 in both runs, declining to ~0.11–0.12 over 500 steps.**
  One in eight student-generated tokens fell outside the trust region — an ~88% speculative-decoding
  acceptance rate for 2B-vs-8B, i.e. the method was *active throughout*, rewriting the loss on ~12% of
  tokens. The tie/deficit is therefore not "the trust region never fired."
- **The baseline reduces outliers faster.** Both curves start together (same init; the metric is
  counterfactual in the baseline run), but by steps 350–500 the baseline sits at ~0.11 vs TrOPD's ~0.12:
  training with full reverse KL *on the outlier tokens themselves* shrinks the outlier population faster
  than the top-k FKL substitute does. The protective substitution slows the very alignment it protects.
- **Eval (off-policy dataset tokens): TrOPD ~0.25–0.28 vs baseline ~0.11.** The teacher is identical, so
  this is purely a student property: the TrOPD student assigns more probability than the teacher to
  dataset tokens ~2.5× as often — a systematic calibration drift on off-policy text, which lines up with
  its out-of-domain benchmark deficit. Likely suspect: mass-covering FKL restricted to a 64-token slice
  of a ~150k vocabulary (`top_k=64` may be miscalibrated for VLM vocabularies vs. the paper's text-only
  setting), though the mechanism is not isolated here.

## Conclusions

1. **In a stable regime, TrOPD masking has no upside and a small, growing downside.** It replaced the
   loss on ~12% of tokens, never won a metric at either scale (0/6, one gap significant w.r.t. decoding
   noise), slowed outlier elimination, and drifted calibration on off-policy text — while the
   instability it guards against never materialized (temperature 0.6, LoRA, LR 1e-4, ≤ 500 steps).
2. **Ship as an opt-in flag; do not change any default.** The paper's text-only gains do not transfer
   for free to VLMs in ordinary setups, and the DAPO precedent for default-flips (broad benchmarks +
   independent replications) is nowhere near met.
3. **The `outlier_token_frac` diagnostic earns its place** — it converts a "tie" into an interpretable
   result and tells any future user whether their run is in a regime where the flag could matter
   (fraction rising ⇒ instability precursor ⇒ TrOPD's target regime).

## Limitations / future work

- One training seed per variant; 374-item in-domain eval (floor ≈ 2.5 pts); MINI benchmark splits.
- The method's claimed home turf is untested: high LR (3–5e-4+), generation temperature ≥ 1.0, full
  language-model finetuning instead of LoRA, longer horizons — regimes where the *baseline* should
  destabilize. A positive TrOPD result, if it exists for VLMs, lives there.
- `outlier_fkl_top_k` ablation (64 → 512/1024) to test the truncated-FKL-calibration hypothesis.
- Off-policy guidance (Eq. 8–9, cosine-annealed teacher prefixes) not implemented.
