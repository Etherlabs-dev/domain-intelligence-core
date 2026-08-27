# Project 03 — Progress, Decisions and Learnings

A running record of what was built, what broke, and what was learned. Appended
to as work proceeds. Newest section at the bottom.

---

## Why this file exists

Project 03 shipped a fine-tuned model twice before anyone checked whether the
training data could support the project's stated goal. It could not. This log
exists so that decisions and their reasons survive outside of chat history, and
so the same class of mistake is visible before it costs GPU hours.

---

## Stage 1 — Environment (2026-08-22 → 2026-08-23)

Four Kaggle runs failed before one trained. Each failure taught something that
is now encoded in the notebook.

| Failure | Cause | Fix |
|---|---|---|
| Run died after 2.9 hrs, OOM | `xformers<0.0.27` has no cp312 wheel; pip built it from source | Pin `xformers==0.0.34` |
| `UsageError: No API key configured` | `wandb.login(anonymous="allow")` is a no-op in current wandb and raises with no TTY | Key from a variable, wrapped so W&B can never kill a run |
| `Temporary failure in name resolution` | Notebook internet was off | Enable in UI; the API's `enable_internet` reports the last *committed* version, not the live draft |
| `Tesla P100 is sm_60` | Kaggle's API cannot set the accelerator, so it defaults to P100, which Unsloth cannot use | `assert major >= 7` in Step 2; set T4 x2 in the UI |

**Learnings**

- `kaggle kernels push` is not a file sync. It **creates and runs** a version,
  and resets the accelerator to P100. Launch training from the browser only.
- `%%capture` on an install cell hides the error that explains everything.
  Never capture a step that can fail.
- Kaggle's image ships `transformers 5.0.0`, which Unsloth excludes **by name**
  (`!=5.0.0`), and `datasets 5.0.0`, above its `<4.4.0` cap. A bare
  `pip install unsloth` has no resolvable solution. Pins are mandatory.
- `import unsloth` must precede transformers/trl/peft. It patches them at import
  time; if they load first the patches never apply.

---

## Stage 2 — First training runs (2026-08-23 → 2026-08-24)

| Run | Data | Result |
|---|---|---|
| Smoke | 2,000 rows | train 0.2506 / eval 0.2483 at step 50 |
| Session 1 | rows 0–20k | 1,188 steps, 2h54m, train 3.04 → 0.167 |
| Session 2 | rows 20k–40k | 1,250 steps, resumed from session 1, train 0.171 → 0.160 |

**Learnings**

- TRL defaults to **3 epochs**. Unset, the smoke test ran 357 steps instead of
  119. Always set `num_train_epochs` explicitly.
- Session 2's train loss *started* at 0.171 — where session 1 ended. That is the
  proof a resumed adapter actually carried over; a fresh adapter would start
  near 3.0. Loss continuity is better evidence than any log line.
- `lora_dropout=0.05` disables Unsloth's fast path (it patched 0 QKV / 0 O /
  0 MLP layers). Setting it to 0.0 restored full patching for a measured ~14%
  speedup — not the advertised 2x.
- Throughput is data-dependent: 8.59 s/step on rows 0–20k, 9.91 s/step on rows
  20k–40k. Longer sequences in the second slice.
- Only 1 of the 2 T4s is ever used. Multi-GPU needs `torchrun` against a script,
  which a notebook cell cannot do. **Still open.**
- **Diminishing returns arrived fast.** Session 1 moved eval loss −0.0220;
  session 2 moved it −0.0082 for identical compute. The plan called for 14
  sessions; the data said stop after 2.

---

## Stage 3 — The dataset defect (2026-08-24)

Before session 3, the training data was inspected properly for the first time.

```
Etherlabs/ios-risk-finetune-v1 — 276,772 rows
  unique instructions : 1
  unique outputs      : 2   (LEGITIMATE 99.1% / FRAUD 0.9%)
```

Every row carried the same instruction and one of two words as its answer. The
project brief calls for a model that "understands financial risk, fraud
patterns, AML typologies, and regulatory language." No dataset shaped like this
can teach that.

**Root cause**, traced to `ios-risk-data-foundry`:

1. `configs/pipeline_config.yaml` had `synthetic: enabled: false` and
   `sec_edgar: enabled: false`
2. `merger.merge_sources()` never listed `synthetic_scenario_pairs.jsonl` as a
   source, so the 10,000 reasoning pairs recorded in `dataset_manifest.json`
   could not reach the export even with the flags on

A third defect was found in the generator itself: it put `FraudType` and
`RiskLevel` **in the model's input**. Those are the answer. The model was being
taught to copy, and the eval prompts omit both fields — so it would have failed
at eval time regardless.

**Learnings**

- Loss going down says nothing about whether the model is learning the task.
  With 99.1% of answers identical, "always say LEGITIMATE" scores ~0.16 loss and
  99% accuracy while being worthless.
- A published artifact is not a verified artifact. `dataset_manifest.json`
  recorded 10,000 scenario pairs that were never in the export.
- **Inspect label distribution and instruction cardinality before training.**
  Two lines of Python would have caught this before ~7 GPU-hours.

---

## Stage 4 — v2 dataset and retraining (2026-08-24)

`Etherlabs/ios-risk-finetune-v2`, 20,000 rows:

| | v1 | v2 |
|---|---|---|
| Unique instructions | 1 | 2 |
| Unique outputs | 2 | 9,035 |
| Risk tiers | none | LOW 8,000 · HIGH 1,000 · CRITICAL 1,000 |
| Fraud share (tabular half) | 0.9% | 25% |

Trained fresh rather than continuing the v1 adapter: v1 had learned "always
answer LEGITIMATE", a worse starting point than untouched Llama.

Result: train loss 3.06 → 0.135, eval 0.1536 → 0.1372, no divergence.

The end-of-run smoke probes were correct on all three — right tier, right
typology, and it switched between prose and single-token output based on the
instruction. But it also emitted **`Probability of fraud: 89.4%`** and
**`Total score: 68`**, neither of which exists in the training data or the
input, and referred to "account history" that was not provided.

**Learnings**

- Fine-tuning teaches **behaviour and format**, not reasoning. The competent AML
  answer seen before any useful training was base Llama 3.1's own knowledge.
- The input ceiling binds harder than the model. Seven numeric features cannot
  support reasoning about merchant, device, geography or account history.
- Confabulated precision is the dangerous failure mode. "89.4%" reads as
  calibrated to anyone who does not know it was invented.
- Eval loss across different datasets is **not comparable**. v1's 0.1605 and
  v2's 0.1372 measure different held-out sets and say nothing about each other.

---

## Stage 5 — Evaluation and v3 groundwork (2026-08-25)

**Eval harness** (`eval/testset.py`, `eval/domain_eval.py`,
`notebooks/03_eval_results.ipynb`): 50 risk scenarios + 200 balanced
classification cases, generated on a different seed than training. Scores tier
accuracy and avg quality against the Project 03 targets (0.70 / 0.60) and
precision/recall/F1 against the Project 01 XGBoost baseline (P 0.9011,
R 0.8367, F1 0.8677). **Runs the base model over the identical set** — without
that control a score means nothing.

**v3 sources under construction**

- `foundry/sources/aml_typologies.py` — 6 typologies (structuring, layering,
  smurfing, trade-based, rapid movement, mule networks) plus hard negatives that
  look suspicious and are not
- `scripts/distill_reasoning.py` — rewrites templated explanations as varied
  analyst prose using an open-weights model, validating every rewrite against
  tier, typology, and invented figures
- Regulatory grounding via the eCFR API (31 CFR Chapter X — the Bank Secrecy
  Act) — **in progress**; the API returned 503/queue-full on first attempt

`sec_edgar.py` stays disabled: its "answers" were templated from the search
query rather than the filing, so every result for a query was identical. It
produced input that looked real attached to output that was fabricated.

**Licence finding:** the Llama 3.1 Community Licence requires `"Llama"` at the
**beginning** of any fine-tuned model's name, plus a "Built with Llama"
attribution and a NOTICE file. `Etherlabs/ios-risk-llama3-v1` does not comply.
Publish as `Llama-3.1-8B-IOS-Risk-v1`.

---

## Open items

| Item | Status |
|---|---|
| Run `03_eval_results.ipynb` — the first defensible number | **blocking publication** |
| Distillation on Together.ai (200-record sample first) | ready to run |
| Regulatory grounding via eCFR | in progress |
| Second T4 idle — needs `torchrun` against `train/` | open |
| Model card with the Llama naming fix and stated limitations | pending eval |
