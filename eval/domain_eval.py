"""Strict multi-task evaluation for IOS Risk Project 03."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from train.dataset import ALPACA_PROMPT

PATTERN_ALIASES = {
    "card_testing": ("CARD TEST", "CARD VALIDATION"),
    "account_takeover": ("ACCOUNT TAKEOVER", "CREDENTIAL COMPROMISE"),
    "bust_out": ("BUST-OUT", "BUST OUT"),
    "structuring": ("STRUCTURING", "CTR EVASION"),
    "layering": ("LAYERING", "PASS-THROUGH"),
    "legitimate": ("LEGITIMATE", "NO TYPOLOGY", "NO SUSPICIOUS"),
}
ACTION_ALIASES = {
    "restrict": ("BLOCK", "FREEZE", "RESTRICT", "HOLD", "STEP-UP"),
    "escalate": ("ESCALATE", "INVESTIGAT", "ENHANCED DUE DILIGENCE", "SAR"),
    "no_action": ("NO ACTION", "STANDARD MONITOR", "CONTINUE MONITOR"),
}
SCORE_CLAIM = re.compile(
    r"\b(?:probability|confidence|risk score|fraud score|model score|score of|likelihood)\b",
    re.IGNORECASE,
)


def extract_tier(response: str) -> str | None:
    match = re.match(r"\s*(CRITICAL|HIGH|MEDIUM|LOW)\s+RISK\b", response, re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_label(response: str) -> str | None:
    match = re.fullmatch(r"\s*(FRAUD|LEGITIMATE)\s*[.!]?\s*", response, re.IGNORECASE)
    return match.group(1).upper() if match else None


def unsupported_claims(response: str, input_text: str) -> list[str]:
    claims: list[str] = []
    if SCORE_CLAIM.search(response):
        claims.append("invented_score_or_probability")
    checks = {
        "device": "device",
        "account history": "history",
        "transaction history": "history",
        "geolocation": "location",
        "merchant category": "merchant",
    }
    lower_response, lower_input = response.lower(), input_text.lower()
    for phrase, required in checks.items():
        if phrase in lower_response and required not in lower_input:
            claims.append(f"unsupported_{required}")
    return sorted(set(claims))


def score_risk(response: str, case: dict[str, Any]) -> dict[str, Any]:
    upper = response.upper()
    evidence_hits = [
        token for token in case["expected_evidence"] if token.upper() in upper
    ]
    result = {
        "predicted_tier": extract_tier(response),
        "correct_tier": extract_tier(response) == case["expected_tier"],
        "correct_pattern": any(
            alias in upper for alias in PATTERN_ALIASES[case["expected_pattern"]]
        ),
        "evidence_supported": len(evidence_hits)
        >= min(2, len(case["expected_evidence"])),
        "correct_action": any(
            alias in upper for alias in ACTION_ALIASES[case["expected_action"]]
        ),
        "unsupported_claims": unsupported_claims(response, case["input"]),
    }
    result["quality_score"] = (
        sum(
            bool(result[key])
            for key in (
                "correct_tier",
                "correct_pattern",
                "evidence_supported",
                "correct_action",
            )
        )
        / 4
    )
    return result


def classification_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(
        r["expected_label"] == "FRAUD" and r["predicted_label"] == "FRAUD" for r in rows
    )
    fp = sum(
        r["expected_label"] != "FRAUD" and r["predicted_label"] == "FRAUD" for r in rows
    )
    fn = sum(
        r["expected_label"] == "FRAUD" and r["predicted_label"] != "FRAUD" for r in rows
    )
    tn = sum(
        r["expected_label"] != "FRAUD" and r["predicted_label"] == "LEGITIMATE"
        for r in rows
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "unparseable": sum(r["predicted_label"] is None for r in rows),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def load_model(model_id: str, max_seq_length: int = 1024):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id, max_seq_length=max_seq_length, load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def run(model_id: str, tag: str, testset_path: str, out_dir: str) -> dict[str, Any]:
    testset = json.loads(Path(testset_path).read_text())
    model, tokenizer = load_model(model_id)
    risk_rows: list[dict] = []
    class_rows: list[dict] = []
    regulation_rows: list[dict] = []
    for index, case in enumerate(testset["cases"], 1):
        prompt = ALPACA_PROMPT.format(case["instruction"], case["input"], "")
        response = generate(
            model, tokenizer, prompt, 8 if case["task"] == "classification" else 240
        )
        if case["task"] == "risk_assessment":
            risk_rows.append(
                {**case, "response": response, **score_risk(response, case)}
            )
        elif case["task"] == "classification":
            class_rows.append(
                {
                    **case,
                    "response": response,
                    "predicted_label": extract_label(response),
                }
            )
        else:
            regulation_rows.append(
                {
                    **case,
                    "response": response,
                    "citation_correct": case["expected_citation"] in response,
                }
            )
        if index % 25 == 0:
            print(f"  {index}/{len(testset['cases'])}")

    risk_n = len(risk_rows)
    unsupported_n = sum(bool(row["unsupported_claims"]) for row in risk_rows)
    risk_summary = {
        "n": risk_n,
        "tier_accuracy": round(
            sum(row["correct_tier"] for row in risk_rows) / risk_n, 4
        ),
        "pattern_accuracy": round(
            sum(row["correct_pattern"] for row in risk_rows) / risk_n, 4
        ),
        "evidence_rate": round(
            sum(row["evidence_supported"] for row in risk_rows) / risk_n, 4
        ),
        "action_accuracy": round(
            sum(row["correct_action"] for row in risk_rows) / risk_n, 4
        ),
        "avg_quality": round(
            sum(row["quality_score"] for row in risk_rows) / risk_n, 4
        ),
        "unsupported_claim_rate": round(unsupported_n / risk_n, 4),
    }
    regulation_n = len(regulation_rows)
    summary = {
        "schema_version": 2,
        "model": model_id,
        "tag": tag,
        "testset_sources": testset["sources"],
        "risk_assessment": risk_summary,
        "classification": classification_metrics(class_rows),
        "regulatory_recall": {
            "n": regulation_n,
            "citation_accuracy": round(
                sum(row["citation_correct"] for row in regulation_rows) / regulation_n,
                4,
            )
            if regulation_n
            else None,
        },
        "targets": {
            "tier_accuracy": 0.70,
            "avg_quality": 0.60,
            "max_unsupported_claim_rate": 0.05,
        },
    }
    summary["passes_project03"] = (
        risk_summary["tier_accuracy"] > 0.70
        and risk_summary["avg_quality"] > 0.60
        and risk_summary["unsupported_claim_rate"] <= 0.05
    )
    payload = {
        "summary": summary,
        "risk_rows": risk_rows,
        "class_rows": class_rows,
        "regulatory_rows": regulation_rows,
    }
    output = Path(out_dir) / f"eval_{tag}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--testset", default="eval/testset.json")
    parser.add_argument("--out-dir", default="eval/results")
    args = parser.parse_args()
    run(args.model, args.tag, args.testset, args.out_dir)
