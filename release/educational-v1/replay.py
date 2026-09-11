"""Recompute the published benchmark without downloading weights or using an API."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "SHA256SUMS.json").read_text())
for name, expected in manifest.items():
    actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Release file changed: {name}")
sys.path.insert(0, str(ROOT / "runtime.zip"))
from train.release_candidate.scoring import aggregate  # noqa: E402

data = json.loads((ROOT / "candidate_data.json").read_text())
cases = [
    row
    for row in data["records"] + data["familiar"]
    if row["split"] not in ("train", "development")
]
identities = {row["id"] for row in cases}
protocols = {
    row["id"]: "regulatory_familiar"
    if row["split"] == "familiar_section_paraphrase"
    else "regulatory_transfer"
    for row in cases
    if row["task"] == "regulatory_identification"
}
reviews = json.loads((ROOT / "comparison-judgments.json").read_text())
for model in ("tuned", "gpt"):
    if model == "gpt":
        generations = json.loads((ROOT / "gpt-generations.json").read_text())
    else:
        records = [
            json.loads(s) for s in (ROOT / "tuned-raw.jsonl").read_text().splitlines()
        ]
        generations = {
            r["id"]: r["generation"] for r in records if r["id"] in identities
        }
    score = aggregate(
        cases,
        generations,
        reviews[model]["semantic"],
        protocols,
        reviews[model]["failures"],
    )
    assert score["review_complete"] and score["failure_reviews_complete"]
    print(
        model,
        json.dumps(
            {
                "classification_f1": score["classification"]["f1"],
                "risk_full_answer": score["risk"]["full_answer_accuracy"],
                "regulatory_familiar_joint": score["regulatory_familiar"][
                    "joint_accuracy"
                ],
                "regulatory_transfer_joint": score["regulatory_transfer"][
                    "joint_accuracy"
                ],
            },
            indent=2,
        ),
    )
