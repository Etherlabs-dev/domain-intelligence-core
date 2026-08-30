"""Strict multi-task evaluation for IOS Risk Project 03."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

EVAL_BUNDLE_VERSION = "2026-08-30.2"
EXPECTED_TESTSET_SHA256 = (
    "85edb481b4bbceeb0a1630830882e2f5c05cf5b1a664667678727526462a7fe8"
)
EXPECTED_COUNTS = {
    "risk_assessment": 50,
    "classification": 200,
    "regulatory_recall": 26,
}

# Keep the Kaggle evaluation asset self-contained. Importing this from
# train.dataset caused the first v3 evaluation to fail because Kaggle mounts only
# domain_eval.py and testset.json, not the repository's train package.
ALPACA_PROMPT = """Below is an instruction that describes a financial risk analysis task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

ALPACA_NO_INPUT_PROMPT = """Below is an instruction that describes a financial risk analysis task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

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


class TestsetValidationError(ValueError):
    """Raised before model loading when the frozen evaluation asset is invalid."""


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_testset(payload: Any) -> dict[str, Any]:
    """Validate the frozen 276-case contract before consuming GPU time."""
    if not isinstance(payload, dict):
        raise TestsetValidationError("Test set must be a JSON object")
    if payload.get("schema_version") != 2:
        raise TestsetValidationError(
            f"Expected test-set schema_version 2; got {payload.get('schema_version')!r}"
        )
    if payload.get("counts") != EXPECTED_COUNTS:
        raise TestsetValidationError(
            f"Expected frozen counts {EXPECTED_COUNTS}; got {payload.get('counts')!r}"
        )
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise TestsetValidationError("Test set 'cases' must be a list")
    expected_total = sum(EXPECTED_COUNTS.values())
    if len(cases) != expected_total:
        raise TestsetValidationError(
            f"Expected {expected_total} cases; got {len(cases)}"
        )

    actual_counts: Counter[str] = Counter()
    seen_cases: set[tuple[str, str, str]] = set()
    source_record_ids: set[str] = set()
    regulatory_citations: set[str] = set()
    valid_tiers = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    valid_labels = {"FRAUD", "LEGITIMATE"}

    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise TestsetValidationError(f"Case {index} must be an object")
        task = case.get("task")
        if task not in EXPECTED_COUNTS:
            raise TestsetValidationError(f"Case {index} has unknown task {task!r}")
        actual_counts[task] += 1
        if not _nonempty_text(case.get("instruction")):
            raise TestsetValidationError(f"Case {index} has an empty instruction")
        if not _nonempty_text(case.get("input")):
            raise TestsetValidationError(f"Case {index} has an empty input")
        identity = (task, case["instruction"].strip(), case["input"].strip())
        if identity in seen_cases:
            raise TestsetValidationError(f"Case {index} duplicates an earlier prompt")
        seen_cases.add(identity)

        if task == "risk_assessment":
            if case.get("expected_tier") not in valid_tiers:
                raise TestsetValidationError(
                    f"Risk case {index} has invalid tier {case.get('expected_tier')!r}"
                )
            if case.get("expected_pattern") not in PATTERN_ALIASES:
                raise TestsetValidationError(
                    f"Risk case {index} has invalid pattern {case.get('expected_pattern')!r}"
                )
            if case.get("expected_action") not in ACTION_ALIASES:
                raise TestsetValidationError(
                    f"Risk case {index} has invalid action {case.get('expected_action')!r}"
                )
            evidence = case.get("expected_evidence")
            if (
                not isinstance(evidence, list)
                or len(evidence) < 2
                or any(not _nonempty_text(item) for item in evidence)
                or len(set(evidence)) != len(evidence)
            ):
                raise TestsetValidationError(
                    f"Risk case {index} needs at least two unique evidence strings"
                )
        elif task == "classification":
            if case.get("expected_label") not in valid_labels:
                raise TestsetValidationError(
                    f"Classification case {index} has invalid label "
                    f"{case.get('expected_label')!r}"
                )
            source_id = case.get("source_record_id")
            if not _nonempty_text(source_id) or source_id in source_record_ids:
                raise TestsetValidationError(
                    f"Classification case {index} has a missing or duplicate source_record_id"
                )
            source_record_ids.add(source_id)
        else:
            citation = case.get("expected_citation")
            if not _nonempty_text(citation) or citation in regulatory_citations:
                raise TestsetValidationError(
                    f"Regulatory case {index} has a missing or duplicate citation"
                )
            regulatory_citations.add(citation)

    if dict(actual_counts) != EXPECTED_COUNTS:
        raise TestsetValidationError(
            f"Case task counts do not match frozen counts: {dict(actual_counts)}"
        )
    return payload


def validate_testset_file(
    path: str | Path, expected_sha256: str | None = EXPECTED_TESTSET_SHA256
) -> dict[str, Any]:
    """Read, authenticate, and validate the frozen evaluation test set."""
    testset_path = Path(path)
    raw = testset_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise TestsetValidationError(
            f"Test-set SHA-256 mismatch: got {digest}, expected {expected_sha256}"
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TestsetValidationError(f"Invalid test-set JSON: {exc}") from exc
    validate_testset(payload)
    print(
        f"[Preflight] Evaluation bundle {EVAL_BUNDLE_VERSION}: "
        f"{len(payload['cases'])} cases, SHA-256 {digest}"
    )
    return payload


def format_prompt(instruction: str, input_text: str) -> str:
    """Use exactly the same prompt structure as training without repo imports."""
    if input_text.strip():
        return ALPACA_PROMPT.format(instruction, input_text, "")
    return ALPACA_NO_INPUT_PROMPT.format(instruction, "")


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
    inputs = tokenizer(
        [prompt], return_tensors="pt", truncation=True, max_length=1024
    ).to("cuda")
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


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def run(model_id: str, tag: str, testset_path: str, out_dir: str) -> dict[str, Any]:
    # Validate before model loading so a bad or stale asset fails in seconds.
    testset = validate_testset_file(testset_path)
    output_dir = Path(out_dir)
    response_log = output_dir / f"eval_{tag}.responses.jsonl"
    response_log.parent.mkdir(parents=True, exist_ok=True)
    response_log.write_text("")

    risk_rows: list[dict] = []
    class_rows: list[dict] = []
    regulation_rows: list[dict] = []
    model = tokenizer = None
    try:
        model, tokenizer = load_model(model_id)
        with response_log.open("a", buffering=1) as stream:
            for index, case in enumerate(testset["cases"], 1):
                prompt = format_prompt(case["instruction"], case["input"])
                response = generate(
                    model,
                    tokenizer,
                    prompt,
                    8 if case["task"] == "classification" else 240,
                )
                if case["task"] == "risk_assessment":
                    result = {
                        **case,
                        "response": response,
                        **score_risk(response, case),
                    }
                    risk_rows.append(result)
                elif case["task"] == "classification":
                    result = {
                        **case,
                        "response": response,
                        "predicted_label": extract_label(response),
                    }
                    class_rows.append(result)
                else:
                    result = {
                        **case,
                        "response": response,
                        "citation_correct": case["expected_citation"] in response,
                    }
                    regulation_rows.append(result)

                # Preserve each expensive model response immediately. If a later
                # case fails, Kaggle still persists this diagnostic JSONL file.
                stream.write(
                    json.dumps({"index": index, "task": case["task"], "result": result})
                    + "\n"
                )
                stream.flush()
                if index % 25 == 0 or index == len(testset["cases"]):
                    print(f"  {index}/{len(testset['cases'])}")
    finally:
        # Loading the tuned adapter after the base model otherwise risks retaining
        # several GB in CUDA's allocator across the two sequential runs.
        del model, tokenizer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

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
    output = output_dir / f"eval_{tag}.json"
    _write_json_atomic(output, payload)
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
