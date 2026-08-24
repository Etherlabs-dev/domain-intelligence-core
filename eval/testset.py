"""
IOS Risk — Held-Out Evaluation Set
==================================
Builds the fixed evaluation set used by eval/domain_eval.py.

The set has two halves, because the model is trained on two tasks:

  1. RISK ASSESSMENT — scenario prompts expecting a risk tier
     (LOW / HIGH / CRITICAL) plus reasoning and a recommended action.
     This is what Project 03's success criteria measure.

  2. CLASSIFICATION — tabular prompts expecting the single token
     FRAUD or LEGITIMATE. This is the head-to-head against the
     XGBoost baseline from Project 01's eval harness.

HELD OUT MEANS HELD OUT
  Scenario prompts are generated with EVAL_SEED, which differs from the
  training seed, so no prompt here was produced during dataset build.
  Tabular prompts are drawn from a disjoint slice of the v1 dataset.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

# Different from the build seed (42). Same generator, different draws.
EVAL_SEED = 20260824

N_SCENARIO_FRAUD = 24  # 6 per fraud type
N_SCENARIO_LEGIT = 26
N_TABULAR = 200  # balanced 50/50 so recall is measurable

TIER_OF_FRAUD_TYPE = {
    "card_testing": "HIGH",
    "account_takeover": "CRITICAL",
    "money_mule": "HIGH",
    "bust_out": "CRITICAL",
}

RISK_INSTRUCTION = (
    "You are IOS Risk, an AI system for financial risk assessment. "
    "Analyse the following transaction and assess its fraud risk."
)
CLASSIFY_INSTRUCTION = (
    "Classify this financial transaction as FRAUD or LEGITIMATE based on the features provided."
)


def _scenario_cases(foundry_path: Path) -> list[dict[str, Any]]:
    """Generate held-out scenario prompts using the foundry's own factory."""
    import sys

    sys.path.insert(0, str(foundry_path))
    from foundry.sources.synthetic_generator import (  # noqa: E402
        FraudScenarioFactory,
        format_feature_line,
    )

    factory = FraudScenarioFactory(seed=EVAL_SEED)
    generators = {
        "card_testing": factory.generate_card_testing,
        "account_takeover": factory.generate_account_takeover,
        "money_mule": factory.generate_money_mule,
        "bust_out": factory.generate_bust_out,
    }

    cases: list[dict[str, Any]] = []
    per_type = N_SCENARIO_FRAUD // len(generators)
    for fraud_type, gen in generators.items():
        for _ in range(per_type):
            rec = gen()
            cases.append({
                "task": "risk_assessment",
                "instruction": RISK_INSTRUCTION,
                "input": format_feature_line(
                    amount=rec["Amount"],
                    hour=int(rec["Time"]) // 3600 % 24,
                    txn_count=rec["txn_count_1h"],
                ),
                "expected_tier": TIER_OF_FRAUD_TYPE[fraud_type],
                "expected_pattern": fraud_type,
            })

    rng = random.Random(EVAL_SEED)
    for _ in range(N_SCENARIO_LEGIT):
        amount = round(rng.lognormvariate(3.5, 1.2), 2)
        hour = rng.randint(0, 22)
        txn_count = rng.randint(1, 4)
        cases.append({
            "task": "risk_assessment",
            "instruction": RISK_INSTRUCTION,
            "input": format_feature_line(amount=amount, hour=hour, txn_count=txn_count),
            "expected_tier": "LOW",
            "expected_pattern": "legitimate",
        })

    rng.shuffle(cases)
    return cases


def _tabular_cases() -> list[dict[str, Any]]:
    """
    Draw balanced tabular prompts from the tail of v1.

    Training used a random resample of the whole of v1, so perfect disjointness
    cannot be guaranteed without the exact build RNG. The tail is used because
    it is the least likely region to have been sampled heavily, and the result
    is reported as an approximate held-out estimate rather than a clean one.
    """
    from datasets import load_dataset

    ds = load_dataset("Etherlabs/ios-risk-finetune-v1", split="train")
    total = len(ds)
    tail = ds.select(range(total - 60_000, total))

    fraud, legit = [], []
    for row in tail:
        (fraud if row["output"] == "FRAUD" else legit).append(row)

    rng = random.Random(EVAL_SEED)
    half = N_TABULAR // 2
    picked = rng.sample(fraud, min(half, len(fraud))) + rng.sample(legit, half)
    rng.shuffle(picked)

    return [{
        "task": "classification",
        "instruction": CLASSIFY_INSTRUCTION,
        "input": r["input"],
        "expected_label": r["output"],
    } for r in picked]


def build(foundry_path: str, out_path: str = "eval/testset.json") -> dict[str, Any]:
    cases = _scenario_cases(Path(foundry_path)) + _tabular_cases()
    payload = {
        "eval_seed": EVAL_SEED,
        "n_risk_assessment": sum(1 for c in cases if c["task"] == "risk_assessment"),
        "n_classification": sum(1 for c in cases if c["task"] == "classification"),
        "cases": cases,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(cases)} cases to {out_path}")
    print(f"  risk_assessment: {payload['n_risk_assessment']}")
    print(f"  classification : {payload['n_classification']}")
    return payload


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--foundry", required=True, help="path to ios-risk-data-foundry checkout")
    ap.add_argument("--out", default="eval/testset.json")
    args = ap.parse_args()
    build(args.foundry, args.out)
