"""Keep the checked-in Kaggle notebooks aligned with the verified Python pipeline."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def set_source(cell: dict, source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)
    if cell["source"] and not cell["source"][-1].endswith("\n"):
        cell["source"][-1] += "\n"


def update_training_notebook() -> None:
    path = ROOT / "notebooks/02_training_run.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))

    source = "".join(notebook["cells"][9]["source"])
    source = source.replace("import unsloth\n", "")
    source = source.replace(
        "from typing import List, Optional, Tuple, Dict",
        "from typing import Dict, List, Optional",
    )
    source = source.replace(
        "from datasets import Dataset, load_dataset",
        "from datasets import load_dataset",
    )
    source = source.replace("from trl import SFTTrainer\n", "")
    source = source.replace(
        "v2 is 20k total — take all of it", "v3 is 20,606 rows; train once"
    )
    source = source.replace(
        "session N starts at N-1 * 20000", "fresh run starts at row zero"
    )
    source = source.replace(
        "FIXED tail split, same every session", "fixed loss-monitoring split"
    )
    source = source.replace(
        'output_dir: str = "/kaggle/working/ios-risk-llama3-v1-smoke"',
        'output_dir: str = "/kaggle/working/Llama-3.1-8B-IOS-Risk-v1"',
    )
    source = source.replace(
        'wandb_run_name: Optional[str] = "llama3-8b-smoke-test"',
        'wandb_run_name: Optional[str] = "llama3-8b-ios-risk-v1"',
    )
    set_source(notebook["cells"][9], source)

    dataset_source = "".join(notebook["cells"][11]["source"])
    dataset_source = dataset_source.replace(
        "Multi-session aware. Kaggle caps a session at 9 hours, so the 276k dataset\n"
        "    is covered across many sessions. Two invariants make that work:\n\n"
        "      * The validation set is the LAST `val_size` rows, every session. Fixed,\n"
        "        never trained on, so eval loss is comparable run to run.\n"
        "      * The training slice is [sample_offset, sample_offset + max_samples),\n"
        "        so each session advances through data it has not seen.",
        "The final 1,000 shuffled rows monitor loss and are not used for optimization.\n"
        "    The separate Project 03 test set is held out by source record and CFR section.",
    )
    set_source(notebook["cells"][11], dataset_source)

    launch = """import dataclasses
import inspect

import torch
from trl import SFTConfig, SFTTrainer

# This is a fresh v3 run. Do not attach v1 or v2: both used datasets that later
# failed quality or evaluation-leakage checks.
config = TrainingConfig(
    max_samples=None,
    sample_offset=0,
    resume_from_adapter=None,
    output_dir="/kaggle/working/Llama-3.1-8B-IOS-Risk-v1",
    wandb_run_name="llama3-8b-ios-risk-v1",
    report_to=REPORT_TO,
)
print("[Preflight] fresh v3 run; no previous adapter will be loaded")

train_data, eval_data = load_and_prepare_dataset(config)
model, tokenizer = load_model_and_tokenizer(config)
assert tokenizer.eos_token and tokenizer.eos_token_id is not None, "Tokenizer has no EOS token"

def append_eos(example):
    text = example["text"]
    return {"text": text if text.endswith(tokenizer.eos_token) else text + tokenizer.eos_token}

train_data = train_data.map(append_eos, desc="Appending EOS to train")
eval_data = eval_data.map(append_eos, desc="Appending EOS to val")

fields = {field.name for field in dataclasses.fields(SFTConfig)}
sft_kwargs = dict(
    output_dir=config.output_dir,
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    num_train_epochs=config.num_train_epochs,
    learning_rate=config.learning_rate,
    lr_scheduler_type=config.lr_scheduler_type,
    warmup_steps=config.warmup_steps,
    weight_decay=config.weight_decay,
    optim=config.optim,
    logging_steps=config.logging_steps,
    eval_steps=config.eval_steps,
    save_strategy=config.save_strategy,
    save_steps=config.save_steps,
    save_total_limit=config.save_total_limit,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    report_to=config.report_to,
    run_name=config.wandb_run_name,
    seed=config.seed,
    dataset_text_field="text",
    dataset_num_proc=2,
    packing=False,
)
sft_kwargs["eval_strategy" if "eval_strategy" in fields else "evaluation_strategy"] = "steps"
sft_kwargs["max_length" if "max_length" in fields else "max_seq_length"] = config.max_seq_length
for key in [key for key in sft_kwargs if key not in fields]:
    sft_kwargs.pop(key)

training_args = SFTConfig(**sft_kwargs)
params = inspect.signature(SFTTrainer.__init__).parameters
tokenizer_key = "processing_class" if "processing_class" in params else "tokenizer"
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    args=training_args,
    **{tokenizer_key: tokenizer},
)

# This assertion checks the tokenized data that the trainer will actually use.
# If EOS is missing, the notebook stops here before spending training hours.
sample_ids = trainer.train_dataset[0].get("input_ids")
assert sample_ids, "Trainer did not expose tokenized input_ids"
assert sample_ids[-1] == tokenizer.eos_token_id, (
    f"EOS preflight failed: got {sample_ids[-1]}, expected {tokenizer.eos_token_id}"
)
print(f"[Preflight] EOS verified on actual trainer input: {tokenizer.eos_token_id}")

print("Starting training run on GPU...")
train_result = trainer.train()

os.makedirs(config.output_dir, exist_ok=True)
model.save_pretrained(config.output_dir)
tokenizer.save_pretrained(config.output_dir)
print(f"LoRA adapter successfully saved to {config.output_dir}")
"""
    set_source(notebook["cells"][15], launch)

    publish = "".join(notebook["cells"][19]["source"])
    publish = publish.replace(
        "Etherlabs/ios-risk-llama3-v2", "Etherlabs/Llama-3.1-8B-IOS-Risk-v1"
    )
    set_source(notebook["cells"][19], publish)
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


def update_eval_notebook() -> None:
    path = ROOT / "notebooks/03_eval_results.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    set_source(
        notebook["cells"][0],
        """# IOS Risk — Project 03 Evaluation

Compares the untouched Llama base model with the v3 adapter on the same 276
leakage-controlled cases: 50 counterfactual risk assessments, 200 held-out
transaction classifications, and 26 held-out regulatory sections.

**Required inputs**

- `ios-risk-eval-assets-v3`: `testset.json` and `domain_eval.py`
- the completed notebook output containing `Llama-3.1-8B-IOS-Risk-v1`

Use a T4 GPU. The notebook stops immediately if either exact input is absent.
""",
    )
    set_source(
        notebook["cells"][4],
        """import os
import sys

def find(predicate, label):
    hits = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if predicate(root, files):
            hits.append(root)
    if not hits:
        raise AssertionError(f"{label} not attached. Searched /kaggle/input")
    return hits

assets = find(lambda _root, files: "testset.json" in files and "domain_eval.py" in files,
              "v3 eval assets")[0]
sys.path.insert(0, assets)
print("eval assets :", assets)

adapters = [root for root in find(
    lambda _root, files: "adapter_config.json" in files, "trained v3 adapter"
) if "checkpoint" not in root]
expected = [path for path in adapters if os.path.basename(path.rstrip("/")) == "Llama-3.1-8B-IOS-Risk-v1"]
if len(expected) != 1:
    raise AssertionError(f"Expected exactly one v3 adapter; found {adapters}")
ADAPTER = expected[0]
print("adapter     :", ADAPTER)
""",
    )
    set_source(
        notebook["cells"][8],
        """# Importing FastLanguageModel first applies Unsloth's patches.
from unsloth import FastLanguageModel  # noqa: F401
from domain_eval import run

base_summary = run(
    model_id="unsloth/Meta-Llama-3.1-8B-Instruct",
    tag="base",
    testset_path=os.path.join(assets, "testset.json"),
    out_dir="/kaggle/working/eval_results",
)
""",
    )
    verdict = """import json

def row(label, base, tuned, target=None, lower_is_better=False):
    passed = tuned <= target if lower_is_better else tuned > target
    flag = "" if target is None else ("   PASS" if passed else "   FAIL")
    print(f"{label:<23} base {base:>7.4f} tuned {tuned:>7.4f} delta {tuned-base:>+7.4f}{flag}")

base_risk = base_summary["risk_assessment"]
tuned_risk = tuned_summary["risk_assessment"]
print("RISK ASSESSMENT (50 independently authored cases)")
row("tier accuracy", base_risk["tier_accuracy"], tuned_risk["tier_accuracy"], 0.70)
row("average quality", base_risk["avg_quality"], tuned_risk["avg_quality"], 0.60)
row("evidence rate", base_risk["evidence_rate"], tuned_risk["evidence_rate"])
row("action accuracy", base_risk["action_accuracy"], tuned_risk["action_accuracy"])
row("unsupported claims", base_risk["unsupported_claim_rate"],
    tuned_risk["unsupported_claim_rate"], 0.05, lower_is_better=True)

base_class = base_summary["classification"]
tuned_class = tuned_summary["classification"]
print("\\nCLASSIFICATION (200 held-out source records)")
for metric in ("precision", "recall", "f1"):
    row(metric, base_class[metric], tuned_class[metric])
print(f"unparseable — base {base_class['unparseable']}, tuned {tuned_class['unparseable']}")

base_reg = base_summary["regulatory_recall"]
tuned_reg = tuned_summary["regulatory_recall"]
print("\\nREGULATORY RECALL (26 entirely held-out CFR sections)")
row("citation accuracy", base_reg["citation_accuracy"], tuned_reg["citation_accuracy"])
print("\\nPROJECT 03:", "PASS" if tuned_summary["passes_project03"] else "FAIL")

with open("/kaggle/working/eval_results/comparison.json", "w") as handle:
    json.dump({"base": base_summary, "tuned": tuned_summary}, handle, indent=2)
"""
    set_source(notebook["cells"][12], verdict)
    set_source(
        notebook["cells"][14],
        """import json

with open("/kaggle/working/eval_results/eval_tuned.json") as handle:
    rows = json.load(handle)["risk_rows"]

for result in rows[:5]:
    mark = "OK" if result["quality_score"] == 1.0 else "REVIEW"
    print(f"[{mark}] expected {result['expected_tier']} got {result['predicted_tier']}")
    print("input:   ", result["input"])
    print("response:", result["response"][:400].replace("\\n", " "))
    print("unsupported:", result["unsupported_claims"])
    print()

suspect = [result for result in rows if result["unsupported_claims"]]
print(f"responses with unsupported claims: {len(suspect)}/{len(rows)}")
""",
    )
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    update_training_notebook()
    update_eval_notebook()
    print("Synchronized both Kaggle notebooks with the verified v3 pipeline.")
