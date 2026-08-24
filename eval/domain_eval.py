"""
IOS Risk — Domain Evaluation Suite
==================================
Scores a model against the held-out set from eval/testset.py.

Project 03's definition of done:
    tier accuracy  > 0.70
    avg quality    > 0.60   (tier + reasoning + recommended action)

Two tasks are scored separately because they measure different things:

  RISK ASSESSMENT — does the model name the right risk tier, explain why,
      and recommend an action? This is what the fine-tune is FOR. A base
      model with no fine-tuning will usually ramble without committing to
      a tier, which is exactly the gap this measures.

  CLASSIFICATION — precision / recall / F1 on FRAUD, comparable to the
      XGBoost baseline in Project 01 (AP 0.8756, P 0.9011, R 0.8367,
      F1 0.8677). Expect the LLM to lose here. That is a finding, not a
      failure: XGBoost cannot produce a tier or an explanation at all.

Run it twice — once on the base model, once on the adapter — and diff.

    python -m eval.domain_eval --model unsloth/Meta-Llama-3.1-8B-Instruct --tag base
    python -m eval.domain_eval --model /kaggle/working/ios-risk-llama3-v2 --tag tuned
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ALPACA_PROMPT = """Below is an instruction that describes a financial risk analysis task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

# Words that indicate the model explained itself rather than just labelling.
REASONING_MARKERS = (
    "DETECTED", "PATTERN", "INDICATOR", "SUGGESTS", "CONSISTENT",
    "VELOCITY", "ANOMAL", "BEHAVIOUR", "BEHAVIOR", "CHARACTERISTIC",
)
# Words that indicate a recommended action, not just a diagnosis.
ACTION_MARKERS = (
    "BLOCK", "FREEZE", "REVIEW", "FILE", "ESCALATE", "REQUIRED",
    "RECOMMEND", "REFER", "INVESTIGAT", "MONITOR", "NO ACTION",
)


def extract_tier(response: str) -> str | None:
    """
    Pull the risk tier out of a free-text response.

    CRITICAL is checked before HIGH because 'CRITICAL RISK' responses often
    also contain the word 'high'. Order matters here.
    """
    upper = response.upper()
    for tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        if re.search(rf"\b{tier}\b(\s+RISK)?", upper):
            return tier
    return None


def extract_label(response: str) -> str | None:
    """First of FRAUD / LEGITIMATE to appear wins — models often echo both."""
    upper = response.upper()
    hits = [(upper.find(w), w) for w in ("FRAUD", "LEGITIMATE") if w in upper]
    return min(hits)[1] if hits else None


def score_risk_assessment(response: str, expected_tier: str) -> dict[str, Any]:
    upper = response.upper()
    predicted = extract_tier(response)
    correct_tier = predicted == expected_tier
    has_reasoning = any(m in upper for m in REASONING_MARKERS)
    has_action = any(m in upper for m in ACTION_MARKERS)
    return {
        "predicted_tier": predicted,
        "correct_tier": correct_tier,
        "has_reasoning": has_reasoning,
        "has_action": has_action,
        "quality_score": sum([correct_tier, has_reasoning, has_action]) / 3,
    }


def classification_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Precision / recall / F1 treating FRAUD as the positive class."""
    tp = sum(1 for r in rows if r["expected_label"] == "FRAUD" and r["predicted_label"] == "FRAUD")
    fp = sum(1 for r in rows if r["expected_label"] != "FRAUD" and r["predicted_label"] == "FRAUD")
    fn = sum(1 for r in rows if r["expected_label"] == "FRAUD" and r["predicted_label"] != "FRAUD")
    tn = sum(1 for r in rows if r["expected_label"] != "FRAUD" and r["predicted_label"] != "FRAUD")
    unparsed = sum(1 for r in rows if r["predicted_label"] is None)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": len(rows),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "unparseable_responses": unparsed,
        "accuracy": round((tp + tn) / len(rows), 4) if rows else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def load_model(model_id: str, max_seq_length: int = 1024):
    from unsloth import FastLanguageModel

    print(f"[Eval] Loading {model_id} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id, max_seq_length=max_seq_length, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=False,  # greedy: eval must be reproducible
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def run(model_id: str, tag: str, testset_path: str, out_dir: str) -> dict[str, Any]:
    testset = json.loads(Path(testset_path).read_text())
    model, tokenizer = load_model(model_id)

    risk_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []

    for i, case in enumerate(testset["cases"], 1):
        prompt = ALPACA_PROMPT.format(case["instruction"], case["input"])
        is_risk = case["task"] == "risk_assessment"
        response = generate(model, tokenizer, prompt, max_new_tokens=200 if is_risk else 8)

        if is_risk:
            risk_rows.append({**case, "response": response,
                              **score_risk_assessment(response, case["expected_tier"])})
        else:
            class_rows.append({**case, "response": response,
                               "predicted_label": extract_label(response)})

        if i % 25 == 0:
            print(f"  {i}/{len(testset['cases'])} cases")

    n = len(risk_rows)
    summary = {
        "model": model_id,
        "tag": tag,
        "risk_assessment": {
            "n": n,
            "tier_accuracy": round(sum(r["correct_tier"] for r in risk_rows) / n, 4) if n else 0.0,
            "reasoning_rate": round(sum(r["has_reasoning"] for r in risk_rows) / n, 4) if n else 0.0,
            "action_rate": round(sum(r["has_action"] for r in risk_rows) / n, 4) if n else 0.0,
            "avg_quality": round(sum(r["quality_score"] for r in risk_rows) / n, 4) if n else 0.0,
        },
        "classification": classification_metrics(class_rows),
        "project03_targets": {"tier_accuracy": 0.70, "avg_quality": 0.60},
        "xgboost_baseline_project01": {
            "precision": 0.9011, "recall": 0.8367, "f1": 0.8677, "average_precision": 0.8756,
        },
    }
    ra = summary["risk_assessment"]
    summary["passes_project03"] = (
        ra["tier_accuracy"] > 0.70 and ra["avg_quality"] > 0.60
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"eval_{tag}.json").write_text(
        json.dumps({"summary": summary, "risk_rows": risk_rows, "class_rows": class_rows}, indent=2)
    )
    print("\n" + json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="IOS Risk domain evaluation")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="label for the output file, e.g. base / tuned")
    ap.add_argument("--testset", default="eval/testset.json")
    ap.add_argument("--out-dir", default="eval/results")
    args = ap.parse_args()
    run(args.model, args.tag, args.testset, args.out_dir)
