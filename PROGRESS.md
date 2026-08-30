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

---

## Stage 6 — Quota loss and the regulatory source (2026-08-25 → 2026-08-27)

### A 12-hour hang from one careless cell

The first eval notebook cloned this repository inside Step 2 to fetch the
scoring code. The repository is **private**. `git clone` prompted:

```
Username for 'https://github.com':
```

A Kaggle batch session has no stdin, so git blocked on that prompt until Kaggle
killed the session at its 12-hour ceiling. It produced nothing and consumed a
large share of the weekly GPU quota — 32h25m of 30h were spent, locking the
account out for ~39 hours.

**Fix:** the notebook now makes no network call except the base model download.
`testset.json` and `domain_eval.py` live in the `ios-risk-eval-assets` Kaggle
dataset, attached as a normal input. Three guards were added:

- GPU capability check runs **first**, so an API-pushed P100 session dies in
  seconds rather than hanging on model load
- the adapter search **requires** a directory matching `v2` and raises with a
  listing rather than silently scoring the v1 adapter
- `kernel-metadata.json` declares both inputs so a push attaches them itself

**Learnings**

- Never push a notebook that can block on stdin. Anything interactive must fail
  fast in a batch session, and `GIT_TERMINAL_PROMPT=0` is the minimum guard.
- A notebook that depends on a private repo depends on credentials it does not
  have. Ship assets as data, not as a clone.
- Kaggle quota is visible in the account menu tooltip: used, reserved, and time
  to reset. It is a rolling window, not a fixed weekly date.

### Regulatory grounding — BSA via the eCFR API

`sec_edgar.py` was retired rather than fixed. Its answers were templated from
the **search query**, not the filing text, so every chunk retrieved for one
query carried an identical fabricated answer. Real input, fake output — the
worst possible direction, and the same defect class that made v1 unusable.

Replaced by `foundry/sources/bsa_regulations.py`, which builds pairs from
31 CFR Chapter X via the public eCFR API:

```
468 pairs · 117 CFR sections · 328 unique questions · 117 unique answers
Parts 1010, 1020, 1021, 1022, 1023, 1025
```

The direction is reversed: the **answer** is verbatim regulatory text with its
citation, and only the **question** is templated.

**Learnings**

- If exactly one half of an instruction pair must be synthetic, it must be the
  question. A templated question with a real answer teaches real content; a real
  question with a templated answer teaches fabrication.
- Each answer appears ~4 times under different questions. Good for citation
  recall, mild memorisation risk. Acceptable for a recall task.
- These pairs teach regulatory language and citation, not judgement. The model
  learns what a rule requires, not when it applies to an ambiguous fact pattern.
  That is what the typology cases are for.

### Distillation provider — licence check

An earlier claim in this project that NVIDIA Nemotron is "explicitly licensed
for generating synthetic training data" was **wrong** and was corrected after
reading the licence. The NVIDIA Open Model License says:

> "NVIDIA claims no ownership rights in outputs. You are responsible for outputs
> and their subsequent uses."

but is **silent** on training other models on those outputs. The clause that
does address it applies only to NVIDIA **Cosmos** models, a different family.
Silence is not prohibition, but it is inference rather than permission.

Apache 2.0 models (Qwen, Mistral, Mixtral) and MIT models (DeepSeek) have no
output clause at all, so there is nothing to interpret.

Nemotron was chosen anyway, via NVIDIA NIM's free tier (~1,000 credits, no
card, OpenAI-compatible at `https://integrate.api.nvidia.com/v1`). The licence's
attribution requirement is met by the repository `NOTICE` file.

**Learning:** verify a licence before recommending a model on it. "Widely
understood to permit X" is not a licence term.

### Credit arithmetic

1,000 free credits do not cover 6,000 records, and neither does the 5,000
extension. Partial distillation is acceptable: the model needs to learn that
explanations *vary*, and a substantial distilled minority mixed with templated
majority teaches that.

---

## Stage 7 — Evaluation results (2026-08-29)

`notebooks/03_eval_results.ipynb`, Kaggle version `eval-v2-vs-base`, 1229.9s on
T4 x2. Base model and tuned adapter over the **same** 250 held-out cases.

### Project 03 criteria: PASS

**Risk assessment — 50 held-out scenario prompts**

| metric | base | tuned | delta | target |
|---|---|---|---|---|
| tier accuracy | 0.2400 | **0.9800** | +0.7400 | >0.70 PASS |
| avg quality | 0.3133 | **0.9000** | +0.5867 | >0.60 PASS |
| reasoning rate | 0.6800 | 1.0000 | +0.3200 | — |
| action rate | 0.0200 | 0.7200 | +0.7000 | — |

**Classification — 200 balanced held-out transactions**

| metric | base | tuned | XGBoost (Project 01) |
|---|---|---|---|
| precision | 0.0000 | **0.9310** | 0.9011 |
| recall | 0.0000 | 0.8100 | 0.8367 |
| F1 | 0.0000 | **0.8663** | 0.8677 |
| unparseable | 200 / 200 | **0** | n/a |

The 8B fine-tune matches the purpose-built gradient-boosted baseline on F1
(0.8663 vs 0.8677) with higher precision.

**Read the base column correctly.** Base scored 0.0 because all 200 of its
responses were unparseable — untouched Llama writes prose instead of emitting
`FRAUD`/`LEGITIMATE`. It means "cannot perform the task in this format", not
"bad at detecting fraud". Do not publish it as the latter.

### The open defect: fabricated scores in 66% of outputs

33 of 50 risk assessments append an invented numeric score:

```
"AML risk score: 29."
"Fraud risk score: 0.435. Probability of fraud: 0.011."
"A model confidence score of 0.449"
"Transaction history shows no history of fraud"   <- no history was provided
```

The scales are mutually incoherent — 29 and 22 read as 0-100, 0.435 and 0.011
as probabilities — so none of them is a metric. The tier, typology and reasoning
are correct; the trailing number is noise.

**Root cause: no EOS token in the training format.** Confirmed twice over:

1. The v2 training set contains **zero** occurrences of "risk score",
   "probability", "confidence" or "transaction history" across all 20,000
   outputs. The model was never taught this.
2. `notebooks/02_training_run.ipynb` formats examples as
   `### Response:\n{output}` and never appends `tokenizer.eos_token`.

The model is never shown where an answer ends, so it never learns to stop. It
produces the correct assessment, reaches the end of what it was taught, and
continues with whatever seems plausible. The invented scores are autocomplete
filling silence, not a reasoning failure.

**Fix:** append `tokenizer.eos_token` to each formatted training example. One
line. This should be verified by re-running the eval and checking the
"responses containing an invented probability or score" count falls from 33/50.

---

## Current state (2026-08-29)

### Artifacts that exist

| Artifact | Where | State |
|---|---|---|
| `Etherlabs/ios-risk-finetune-v1` | HuggingFace, public | **Defective** — 1 instruction, 2 outputs. Do not train on it. |
| `Etherlabs/ios-risk-finetune-v2` | HuggingFace, private | 20,000 rows, 2 instructions, 9,035 unique outputs |
| v2 adapter | Kaggle `ios-risk-brain-v1-fine-tune`, latest version, `ios-risk-llama3-v2/` | Passes Project 03. Not yet on HuggingFace. |
| `ios-risk-eval-assets` | Kaggle dataset, private | `testset.json` + `domain_eval.py` |
| Eval result | Kaggle `ios-risk-project03-eval`, `eval-v2-vs-base` | PASS, table above |

### v3 inputs

| Source | Count | State |
|---|---|---|
| AML typology pairs | 6,000 | generated, `data/processed/aml_typology_pairs.jsonl` |
| AML distilled | ~3,150 of 6,000 | **in progress**, see below |
| BSA regulation pairs | 468 | generated, 117 CFR sections |

### The distillation job

Running on Ugo's Mac, launched with `nohup`, writing to
`data/processed/aml_sample_distilled.jsonl` and logging to
`/tmp/distill_full.log`. Model `nvidia/nemotron-3-super-120b-a12b` via NVIDIA
NIM. 98% keep rate, `failed=0`.

Throughput is 4-11 records/min — NVIDIA throttles hard, well below its
advertised 40 rpm, and the log shows thousands of handled `RateLimitError`
retries. Those are expected and are not failures.

**It is fully resumable.** Every completed record is a line in the output. Re-run
the identical command and it hashes what is on disk and continues. Nothing is
ever redone or lost. To resume:

```bash
cd ~/Documents/ios-risk-data-foundry
export DISTILL_API_KEY=<nvapi key>
./venv/bin/python -m scripts.distill_reasoning \
  --in data/processed/aml_typology_pairs.jsonl \
  --out data/processed/aml_sample_distilled.jsonl \
  --base-url https://integrate.api.nvidia.com/v1 \
  --model nvidia/nemotron-3-super-120b-a12b --rpm 35
```

---

## What remains

Ordered by value, highest first.

1. **Fix the EOS bug and retrain.** One line in the notebook's formatter, then
   one ~3h Kaggle session. Re-run the eval and confirm the fabricated-score
   count drops. This is the single highest-value change available: the model
   already passes, and this removes its only serious defect.

2. **Publish.** The adapter is not yet on HuggingFace. Two constraints:
   - The Llama 3.1 Community Licence requires `"Llama"` at the **start** of the
     model name. `Etherlabs/ios-risk-llama3-v1` does not comply. Publish as
     **`Llama-3.1-8B-IOS-Risk-v1`**, with a "Built with Llama" attribution and
     a NOTICE file.
   - The model card must state the templated-reasoning limitation and, until
     the EOS fix lands, the fabricated-score rate.

3. **Build v3** — distilled AML + BSA regulation + v2 data. Only worth doing
   after the EOS fix, since the fix may change what v3 needs to address.

4. **Rotate the exposed keys.** The HF token and W&B key were in the notebook
   and in chat; the NVIDIA key was pasted into chat. Ugo decided to defer this
   until generation finished. It has not been done.

5. **Idle second T4.** Unsloth uses one of two GPUs. Needs `torchrun` against
   the `train/` package rather than notebook cells. Open since stage 2.

### Traps a new agent should know

- `kaggle kernels push` **creates and runs** a version and resets the
  accelerator to P100. Never use it to sync code before a training run.
- Kaggle's API cannot set the accelerator. T4 x2 must be chosen in the UI.
- Never put anything in a Kaggle notebook that can block on stdin. A `git clone`
  of a private repo cost 12 hours of quota this way.
- Do not trust `dataset_manifest.json` or a published dataset without inspecting
  instruction cardinality and label distribution first. Two lines of Python
  would have caught the v1 defect before ~7 GPU-hours.
- Eval loss across different datasets is not comparable.

---

## Stage 8 — Independent audit and v3 release preparation (2026-08-29)

The imported Claude Code handoff was treated as untrusted. Project briefs,
both related repositories, local documentation, source code, generated data,
notebooks, logs, and evaluation fixtures were re-read before further work.

### Corrections to Stage 7

The v2 result above is historical evidence, **not a valid final Project 03
result**:

- 101 of its 200 classification evaluation examples also appeared in v2
  training data;
- the 50 risk prompts came from substantially the same generator and rules as
  training, so they were too easy and narrow;
- the original scoring accepted partial prose and did not systematically count
  unsupported facts;
- therefore the reported v2 `PASS` and F1 `0.8663` must not be used as the
  project’s public final performance.

The Stage 7 statement that missing EOS was the confirmed root cause of invented
scores was also too strong. Recent TRL versions may add EOS automatically. The
cause was never proven from the actual tokenized trainer input. v3 addresses the
risk defensively: append EOS explicitly and assert the final token ID before the
first optimizer step. The data and evaluation defects were fixed separately.

### Preserved source artifacts

- Original distillation checkpoint: 3,355 JSONL records, unchanged.
- Strict audit output: 2,462 accepted model-assisted AML rewrites.
- Rejections: 2 duplicate source keys, 4 invented-figure outputs, 827 legacy
  prompts missing support for their answer, and 60 template fallbacks.
- The long-running NVIDIA process was stopped only after completed lines were
  safely on disk; generation is not required to resume for v3.

### Foundry v3

The final local export is deterministic and release-gated:

| Property | Verified value |
|---|---:|
| Records | 20,606 |
| Unique instruction/input pairs | 20,606 |
| Unique outputs | 10,025 |
| Distilled AML | 2,462 |
| Template AML | 3,538 |
| Grounded fraud/benign scenarios | 5,000 |
| Official eCFR training pairs | 364 |
| Tabular training records | 9,242 |
| SHA-256 | `485f02df11b2e1dd4b1dbe0bb4dd9a68615735bbcf64cc7fbbb08933008ca075` |

Scenario inputs now include every fact their answer relies on—for example CTR
threshold, burst exposure, account history, device change, and utilization
change. The validator rejects malformed/empty rows, duplicate prompts,
manifest drift, and unsupported model-score/probability language.

The regulatory cache exposed another collision: many different “general” CFR
sections shared the same vague question. Questions now name their controlling
section. The cached official XML produced 468 unique question pairs across 117
sections; 26 complete sections (104 phrasings) are held out from training.

### Frozen v3 evaluation

`eval/testset.json` contains 276 cases:

- 50 independently authored counterfactual risk cases;
- 200 balanced transaction cases selected only from the tabular source-record
  holdout;
- 26 regulatory prompts, one for each entirely held-out CFR section.

Verified overlap with v3 training:

```text
exact prompt overlap: 0
tabular source-record overlap: 0
regulatory citation overlap: 0
```

Scoring now requires the tier at the start of a risk response, exact one-token
classification, expected pattern/evidence/action, regulatory citation, and an
unsupported-claim audit. Release targets remain tier accuracy `> 0.70`, average
quality `> 0.60`, and unsupported-claim rate `<= 0.05`.

### Code and notebook safeguards

- training dataset schema and diversity fail before model construction;
- one epoch, dropout zero, fixed dependency set, and compliant output name;
- explicit EOS append and actual-token EOS assertion;
- deterministic inference wrapper for Projects 04/05;
- notebooks are valid JSON and synchronized with the Python pipeline;
- eval notebook requires the exact v3 adapter directory and refuses any v1/v2
  or ambiguous adapter attachment;
- secret scrubber edits notebook JSON structurally instead of corrupting quoted
  cells;
- CI runs lint, formatting, unit tests, and notebook JSON validation.

### Current completion boundary

Complete: Foundry v3, strict evaluation assets, training/inference code,
notebook preflights, tests, documentation, and public dataset publication.

The public artifact was independently downloaded after publication on
2026-08-29. It contains 20,606 JSONL records, 20,606 unique instruction/input
pairs, 10,025 unique outputs, and all required instruction/input/output fields.
Its downloaded SHA-256 is
`485f02df11b2e1dd4b1dbe0bb4dd9a68615735bbcf64cc7fbbb08933008ca075`,
matching the committed manifest exactly:
`https://huggingface.co/datasets/Etherlabs/ios-risk-finetune-v3`.

Still external/pending:

1. run one fresh v3 Kaggle training version;
2. run the base-versus-tuned v3 evaluation;
3. publish `Etherlabs/Llama-3.1-8B-IOS-Risk-v1` only if the frozen gates pass;
4. document final measured results and limitations.

The publication gate is now verified. The next action is to copy the local
training notebook exactly into the Kaggle draft and run its preflight before
starting the paid-in-quota GPU training step.

---

## Stage 9 — Final v3 Kaggle training (completed 2026-08-30)

The verified local `notebooks/02_training_run.ipynb` was imported into the
existing Kaggle training draft. Before starting the optimizer, two additional
quota-protection defects were found and fixed:

1. Hugging Face login was unconditional even when `HF_TOKEN` was empty. Because
   the v3 dataset is public, the notebook now uses unauthenticated public access
   when no token is configured. W&B logging is also optional.
2. Preflight and `trainer.train()` were in the same cell, leaving no safe review
   point. They are now separate cells. Step 7 prepares the real trainer and
   stops after all assertions; only Step 8 starts the multi-hour optimizer run.

The live Kaggle preflight completed successfully with:

```text
dataset: Etherlabs/ios-risk-finetune-v3
records loaded: 20,606
quality gate: 4 instruction families, 10,025 unique outputs
split: 19,606 train / 1,000 validation
model source: fresh unsloth/Meta-Llama-3.1-8B-Instruct
previous adapter: none
LoRA trainable parameters: 41,943,040 (0.520%)
EOS token verified on actual trainer input: 128009
preflight result: PASSED
```

After that evidence was reviewed, Step 8 was started. Its trainer report showed
one epoch, 1,226 optimizer steps, total batch size 16, and 19,606 training
examples. Kaggle Version 7, `v3-final-persisted-training`, completed successfully
in 16,786.9 seconds (about 4 hours 40 minutes) on T4 x2. It persisted the final
adapter at `Llama-3.1-8B-IOS-Risk-v1`; the top-level
`adapter_model.safetensors` is about 167.83 MB. Version 7 is pinned so dependent
notebooks resolve this completed output rather than an old v1/v2 run.

Training completion proves that the optimizer and persistence pipeline worked;
it does not prove model quality. The frozen base-versus-tuned evaluation remains
the release gate.

---

## Stage 10 — Evaluation packaging incident and fail-fast repair (2026-08-30)

The first formal v3 evaluation attempt correctly mounted the frozen v3 assets,
the pinned Version 7 adapter, and a Tesla T4 (`sm_75`). It then failed after
about 1 minute 39 seconds, before any v3 inference, with:

```text
ModuleNotFoundError: No module named 'train'
```

The uploaded `domain_eval.py` imported `ALPACA_PROMPT` from the local repository
package `train.dataset`, but the Kaggle dataset contained only `domain_eval.py`
and `testset.json`. The browser log showing `33/50` responses with invented
scores belonged to an older successful v2 evaluation and is not a v3 result.

The correction is deliberately broader than deleting that import:

- evaluation prompt templates are now self-contained inside `domain_eval.py`;
- the 276-case schema, task counts, required fields, uniqueness rules, labels,
  citations, and frozen test-set SHA-256 are validated before model loading;
- a bundle manifest pins the evaluator and test-set hashes and a bundle version;
- the notebook accepts only one exact `ios-risk-eval-assets-v3` mount and one
  exact top-level `Llama-3.1-8B-IOS-Risk-v1` adapter, rejecting checkpoints,
  stale lookalikes, truncated weights, and an incompatible base model;
- dependency versions are checked against the successful training environment;
- each generated response is flushed immediately to JSONL, so a later failure
  does not discard all expensive inference progress;
- base-model memory is explicitly released before loading the tuned adapter;
- the verdict helper no longer compares a metric with `None`, which would have
  crashed only after the full base and tuned evaluation had finished;
- notebook synchronization no longer risks merging training back into the
  preflight cell, and every normal evaluation code cell is compile-checked.

Automated validation now includes 21 tests, static checks, notebook formatting
and JSON checks, an isolated Python import, a simulated Kaggle input tree with a
stale lookalike dataset, and byte-for-byte remote verification. Kaggle dataset
`ethercess/ios-risk-eval-assets-v3` now serves bundle `2026-08-30.2` with:

```text
domain_eval.py  SHA-256 045a2019d068173dc3a73631d0e3c78938c2caa9e2b5630967548e5685c1d1d4
testset.json    SHA-256 85edb481b4bbceeb0a1630830882e2f5c05cf5b1a664667678727526462a7fe8
```

The next external action is a manual refresh of the Kaggle dataset input and a
manual import/review of the corrected evaluation notebook. No evaluator run was
started automatically during this repair.
