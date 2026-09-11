# Project 03 educational baseline

Built with Llama. This release demonstrates measured domain specialization of
Llama 3.1 8B with a QLoRA adapter. It is a teaching and evaluation artifact, not
a qualified financial-decision product. The original RC1 qualification failed;
the release preserves those results and the later matched GPT-4.1 comparison.

## Start with the evidence

Run from a checkout with Python 3.10 or newer:

```bash
python -I release/educational-v1/replay.py
```

This verifies release checksums and recomputes scores from saved predictions and
hash-bound review judgments. It uses no GPU, model weights, API key or network.
It is replay, not live inference. `comparison-report.md` gives results and limits;
`rc1-original-review.json` preserves the preceding qualification assessment.

## Load the model on a compatible CUDA machine

Use the exact tested dependency pins in `manifest.json` and a T4-or-newer CUDA
runtime. The reference run used torch 2.10.0, transformers 5.5.0, trl 0.24.0,
peft 0.19.1, unsloth 2026.8.22 and unsloth_zoo 2026.8.16. Package compatibility
must be checked before loading; this is not a CPU inference recipe.

Download the published adapter snapshot using its immutable revision from the
release receipt, verify its file hashes, then run:

```python
import sys
from pathlib import Path

adapter = Path("/path/to/downloaded/adapter")
sys.path.insert(0, str(adapter / "runtime.zip"))
from inference.native_predictor import NativeRiskPredictor

facts = {
    "WarrantyClaim": "manufacturer approved defective-item claim",
    "CreditVoucher": "matches approved claim",
    "ClaimRecipient": "verified original buyer",
}
with NativeRiskPredictor.load(adapter) as predictor:
    result = predictor.assess_risk(facts)
    print(result["generation"]["text"])
    print(result["validation"])
```

The example is illustrative and not a new benchmark. The loader verifies native
manifest/base identity and returns raw generation and contract checks. Contract
validity is not substantive correctness. No proposed action is executed.

## Reproduce the training design

`candidate_data.json` preserves the 1,900 training rows and evaluation splits;
`manifest.json` freezes their identities, runtime sources and recipe. Training
used 238 steps, batch size 2, accumulation 8, one worker, rank/alpha 16, learning
rate 1e-4 and a fixed final-step selection rule. The worker and full-session caps
were 21,000 and 21,600 seconds. Two T4s were visible; only one training worker ran.

`09_training_record.ipynb` is the exact saved execution notebook. It references
the original private Kaggle dataset, so it is a provenance artifact, not a
one-click public notebook. To execute independently, stage every required file
and source in the package manifest, preserve hashes and native tokenizer checks,
and provide your own compute and base-model access. The repository's candidate
builder documents the package construction. Do not silently claim bit-identical
GPU retraining across runtimes.

The included `runtime.zip` is the byte-frozen Python source runtime from the run;
it can be inspected with any ZIP reader. It is used to prevent subsequent source
edits from silently changing replay or inference semantics.

See `ATTRIBUTION.md`, `PROJECT04_HANDOFF.md` and `WRITING_KIT.md` for source
boundaries, next-project integration and the teaching walkthrough.
