"""Build the fixed, leakage-controlled Project 03 v3 evaluation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

RISK_INSTRUCTION = (
    "You are IOS Risk, a financial risk intelligence system. Analyse only the supplied "
    "facts, assign a risk tier, identify the pattern, and recommend an action."
)


def _case(
    text: str, tier: str, pattern: str, evidence: list[str], action: str
) -> dict[str, Any]:
    return {
        "task": "risk_assessment",
        "instruction": RISK_INSTRUCTION,
        "input": text,
        "expected_tier": tier,
        "expected_pattern": pattern,
        "expected_evidence": evidence,
        "expected_action": action,
    }


def authored_risk_cases() -> list[dict[str, Any]]:
    """Fifty counterfactual cases authored independently of the training generators."""
    cases: list[dict[str, Any]] = []
    for i in range(5):
        cases.extend(
            [
                _case(
                    f"Amount: ${0.41 + i * 0.17:.2f} | Hour: {i + 1} | TxnCount1h: {17 + i} | "
                    "MicroTxn: 1 | OffHours: 1 | LargeTxn: 0 | RoundAmt: 0",
                    "HIGH",
                    "card_testing",
                    ["micro", "velocity"],
                    "restrict",
                ),
                _case(
                    f"Amount: ${0.49 + i * 0.10:.2f} | Hour: 14 | TxnCount1h: 1 | MicroTxn: 1 | "
                    "OffHours: 0 | Merchant: verified transit authority | Recurring: 1",
                    "LOW",
                    "legitimate",
                    ["recurring", "verified"],
                    "no_action",
                ),
                _case(
                    f"Amount: ${820 + i * 113:.2f} | Hour: 2 | TxnCount1h: 2 | NewDevice: 1 | "
                    "NewCountry: 1 | FailedAuth24h: 4 | CustomerVerified: 0",
                    "CRITICAL",
                    "account_takeover",
                    ["new device", "failed"],
                    "restrict",
                ),
                _case(
                    f"Amount: ${910 + i * 97:.2f} | Hour: 2 | TxnCount1h: 1 | NewDevice: 0 | "
                    "NewCountry: 0 | CustomerVerified: 1 | Merchant: hotel | TravelNotice: 1",
                    "LOW",
                    "legitimate",
                    ["verified", "travel"],
                    "no_action",
                ),
                _case(
                    f"Amount: ${730 + i * 55:.2f} | TxnCount1h: {14 + i} | AccountHistoryMonths: 24 | "
                    "PriorPaymentPattern: on_time | CreditUtilizationBeforePct: 9 | "
                    "CreditUtilizationAfterPct: 99 | UtilizationWindowHours: 18",
                    "CRITICAL",
                    "bust_out",
                    ["utilization", "payment"],
                    "restrict",
                ),
                _case(
                    f"Amount: ${730 + i * 55:.2f} | TxnCount1h: 1 | AccountHistoryMonths: 24 | "
                    "CreditUtilizationBeforePct: 9 | CreditUtilizationAfterPct: 14 | CustomerVerified: 1",
                    "LOW",
                    "legitimate",
                    ["verified", "utilization"],
                    "no_action",
                ),
                _case(
                    f"CashDeposits: {4 + i} | DepositWindowDays: 4 | SmallestDeposit: $8,450 | "
                    f"LargestDeposit: $9,650 | TotalCashDeposited: ${38200 + i * 900:,} | "
                    "DistinctBranches: 4 | DocumentedSource: none",
                    "HIGH",
                    "structuring",
                    ["below", "branch"],
                    "escalate",
                ),
                _case(
                    f"CashDeposits: {4 + i} | DepositWindowDays: 30 | TotalCashDeposited: "
                    f"${38200 + i * 900:,} | BusinessType: registered grocery | DistinctBranches: 1 | "
                    "DepositPatternVsPrior12m: consistent | DocumentedSource: sales receipts",
                    "LOW",
                    "legitimate",
                    ["documented", "consistent"],
                    "no_action",
                ),
                _case(
                    f"InboundAmount: ${81000 + i * 6000:,} | OutboundTransfers: {4 + i} | "
                    "DistinctBeneficiaryAccounts: 4 | TimeToFullyDisburseHours: 9 | "
                    "ResidualBalance: $120 | DocumentedPurpose: none",
                    "CRITICAL",
                    "layering",
                    ["rapid", "beneficiar"],
                    "escalate",
                ),
                _case(
                    f"InboundAmount: ${81000 + i * 6000:,} | OutboundTransfers: 1 | "
                    "Beneficiary: verified mortgage lender | TimeToFullyDisburseHours: 9 | "
                    "Source: documented title escrow | DocumentedPurpose: property closing",
                    "LOW",
                    "legitimate",
                    ["documented", "verified"],
                    "no_action",
                ),
            ]
        )
    return cases


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    tabular_holdout: str,
    regulatory_holdout: str,
    out_path: str = "eval/testset.json",
    seed: int = 20260829,
) -> dict[str, Any]:
    tabular_path, regulatory_path = Path(tabular_holdout), Path(regulatory_holdout)
    tabular = _read_jsonl(tabular_path)
    fraud = [row for row in tabular if row["output"] == "FRAUD"]
    legitimate = [row for row in tabular if row["output"] == "LEGITIMATE"]
    rng = random.Random(seed)
    selected = rng.sample(fraud, 100) + rng.sample(legitimate, 100)
    rng.shuffle(selected)
    classification = [
        {
            "task": "classification",
            "instruction": row["instruction"],
            "input": row["input"],
            "expected_label": row["output"],
            "source_record_id": row["source_record_id"],
        }
        for row in selected
    ]

    one_per_citation: dict[str, dict] = {}
    for row in _read_jsonl(regulatory_path):
        one_per_citation.setdefault(row["citation"], row)
    regulatory = [
        {
            "task": "regulatory_recall",
            "instruction": row["instruction"],
            "input": row["input"],
            "expected_citation": row["citation"],
        }
        for row in one_per_citation.values()
    ]

    risk = authored_risk_cases()
    payload = {
        "schema_version": 2,
        "seed": seed,
        "sources": {
            "tabular_holdout": {
                "file": tabular_path.name,
                "sha256": _sha256(tabular_path),
            },
            "regulatory_holdout": {
                "file": regulatory_path.name,
                "sha256": _sha256(regulatory_path),
            },
            "risk_cases": "independently authored counterfactual fixtures",
        },
        "counts": {
            "risk_assessment": len(risk),
            "classification": len(classification),
            "regulatory_recall": len(regulatory),
        },
        "cases": risk + classification + regulatory,
    }
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(payload['cases'])} leakage-controlled cases to {output}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabular-holdout", required=True)
    parser.add_argument("--regulatory-holdout", required=True)
    parser.add_argument("--out", default="eval/testset.json")
    args = parser.parse_args()
    build(args.tabular_holdout, args.regulatory_holdout, args.out)
