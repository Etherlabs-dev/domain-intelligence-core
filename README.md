# IOS Risk Domain Intelligence Core

Project 03 of IntelligenceOS: adapt Llama 3.1 8B Instruct to financial-risk
classification, evidence-grounded fraud/AML assessment, and BSA regulatory
recall—then prove what changed against the untouched base model.

> **Current status:** the exact v3 dataset is publicly verified on Hugging Face,
> final v3 Kaggle training completed successfully, and the pinned adapter output
> is preserved. The first v3 evaluation attempt stopped before inference because
> its Kaggle asset imported a repository-only module. The corrected self-contained
> asset is published and remotely verified; base-versus-tuned results remain
> pending. No model-improvement or production-impact claim is made before them.

## Final experiment design

| Component | Verified configuration |
|---|---|
| Base | `unsloth/Meta-Llama-3.1-8B-Instruct` |
| Data | `Etherlabs/ios-risk-finetune-v3` — 20,606 unique pairs |
| Method | QLoRA 4-bit, LoRA rank/alpha 16, dropout 0, one epoch |
| Compute | One Kaggle T4 is used by the current Unsloth notebook |
| Output | `Etherlabs/Llama-3.1-8B-IOS-Risk-v1` |
| Evaluation | 276 leakage-controlled cases, run identically on base and tuned models |

The second visible T4 is not counted as training capacity: the notebook stack
uses one GPU. Changing to distributed training would be a separate experiment,
not a free speed switch.

## What changed after the failed v1/v2 experiments

- v1 contained one instruction and two labels with 99.1% legitimate examples.
- v2 added prose, but its evaluation overlapped the training generator and 101
  of 200 classification evaluation records leaked into training.
- The claim that missing EOS caused v2's invented scores was not proven. v3 now
  appends EOS explicitly and checks the actual tokenized trainer input before
  the first optimizer step.
- v3 supplies every scenario fact used by its expected explanation and rejects
  unsupported probabilities, scores, malformed rewrites, and duplicate prompts.
- Tabular evaluation is held out by source-record hash; regulatory evaluation
  holds out complete CFR sections; risk cases are independently authored.

## Repository map

```text
train/       reusable QLoRA pipeline and GPU preflight
inference/   deterministic predictor for Projects 04 and 05
eval/        fixed v3 test set and strict multi-task scoring
notebooks/   self-contained Kaggle training and evaluation notebooks
configs/     mirrored human-readable training configuration
tests/       dataset, EOS, inference, and evaluation tests
scripts/     notebook synchronization and secret-safe notebook scrubbing
```

## Local verification

Python 3.11 is recommended.

```bash
pip install -r requirements-test.txt
ruff check .
ruff format --check .
pytest -q
python -m json.tool notebooks/02_training_run.ipynb >/dev/null
python -m json.tool notebooks/03_eval_results.ipynb >/dev/null
```

## Release sequence

1. **Complete:** published and re-downloaded `ios-risk-finetune-v3`; verified
   20,606 rows and SHA-256 `485f02df11b2e1dd4b1dbe0bb4dd9a68615735bbcf64cc7fbbb08933008ca075`.
2. **Complete:** trained the verified v3 notebook for one epoch (1,226 steps) and
   preserved pinned Version 7 output at `Llama-3.1-8B-IOS-Risk-v1`.
3. **Complete:** published and re-downloaded self-contained evaluation bundle
   `2026-08-30.2`; its manifest and both payload hashes match locally.
4. Refresh `ios-risk-eval-assets-v3` in the Kaggle evaluation draft, import the
   corrected local evaluation notebook, and review the fail-fast preflight.
5. Run both the untouched base model and the exact pinned v3 adapter on the same
   276 frozen cases.
6. Publish the adapter only if the frozen Project 03 gates pass; otherwise keep
   the results and diagnose the failed dimension without moving the test set.

## Project 03 gates

- risk-tier accuracy `> 0.70`;
- average risk-response quality `> 0.60`;
- unsupported-claim rate `<= 0.05`;
- classification precision/recall/F1 reported, not hidden behind accuracy;
- regulatory citation recall reported;
- tuned results compared with the untouched base model on identical cases.

These are research release gates. They do not authorize automated account
restriction, SAR filing, or production financial decisions.

See [`PROGRESS.md`](PROGRESS.md) for the complete incident and decision history,
and [`LEARNING_APPROACH.md`](LEARNING_APPROACH.md) for the plain-language handoff.
