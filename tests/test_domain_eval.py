import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from eval.domain_eval import (
    EXPECTED_COUNTS,
    TestsetValidationError as EvalTestsetValidationError,
    extract_label,
    extract_tier,
    format_prompt,
    score_risk,
    unsupported_claims,
    validate_testset,
    validate_testset_file,
)
from scripts.build_eval_assets import build as build_eval_assets

ROOT = Path(__file__).resolve().parents[1]


def test_classification_requires_exact_label():
    assert extract_label("FRAUD") == "FRAUD"
    assert extract_label("The transaction is FRAUD") is None


def test_tier_must_open_the_response():
    assert extract_tier("HIGH RISK — CARD TESTING DETECTED") == "HIGH"
    assert extract_tier("The result is HIGH RISK") is None


def test_unsupported_score_and_missing_device_are_flagged():
    response = "HIGH RISK. Probability of fraud: 89%. A new device was used."
    claims = unsupported_claims(response, "Amount: $1.00")
    assert "invented_score_or_probability" in claims
    assert "unsupported_device" in claims


def test_negative_action_text_does_not_get_generic_review_credit():
    case = {
        "expected_tier": "LOW",
        "expected_pattern": "legitimate",
        "expected_evidence": ["documented", "verified"],
        "expected_action": "no_action",
        "input": "Source: documented | Beneficiary: verified",
    }
    result = score_risk(
        "LOW RISK — LEGITIMATE. Source is documented and beneficiary verified. "
        "No action; continue monitoring.",
        case,
    )
    assert result["correct_action"]
    assert result["quality_score"] == 1.0


def test_frozen_testset_passes_full_preflight():
    payload = validate_testset_file(ROOT / "eval" / "testset.json")
    assert payload["counts"] == EXPECTED_COUNTS
    assert len(payload["cases"]) == 276


def test_testset_preflight_rejects_unknown_tasks_before_model_loading():
    payload = json.loads((ROOT / "eval" / "testset.json").read_text())
    broken = copy.deepcopy(payload)
    broken["cases"][0]["task"] = "mystery_task"
    with pytest.raises(EvalTestsetValidationError, match="unknown task"):
        validate_testset(broken)


def test_testset_preflight_rejects_duplicate_prompts():
    payload = json.loads((ROOT / "eval" / "testset.json").read_text())
    broken = copy.deepcopy(payload)
    broken["cases"][1]["instruction"] = broken["cases"][0]["instruction"]
    broken["cases"][1]["input"] = broken["cases"][0]["input"]
    with pytest.raises(
        EvalTestsetValidationError, match="duplicates an earlier prompt"
    ):
        validate_testset(broken)


def test_testset_preflight_rejects_hash_drift(tmp_path):
    changed = tmp_path / "testset.json"
    changed.write_text((ROOT / "eval" / "testset.json").read_text() + "\n")
    with pytest.raises(EvalTestsetValidationError, match="SHA-256 mismatch"):
        validate_testset_file(changed)


def test_eval_asset_imports_in_isolated_python_without_repository(tmp_path):
    asset = tmp_path / "domain_eval.py"
    asset.write_bytes((ROOT / "eval" / "domain_eval.py").read_bytes())
    probe = (
        "import importlib.util; "
        f"s=importlib.util.spec_from_file_location('domain_eval', {str(asset)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "print(m.EVAL_BUNDLE_VERSION)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_kaggle_asset_builder_outputs_verified_self_contained_bundle(tmp_path):
    bundle = build_eval_assets(tmp_path / "bundle")
    assert {path.name for path in bundle.iterdir()} == {
        "domain_eval.py",
        "testset.json",
        "bundle_manifest.json",
        "dataset-metadata.json",
    }
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    assert manifest["case_count"] == 276
    assert set(manifest["files"]) == {"domain_eval.py", "testset.json"}
    metadata = json.loads((bundle / "dataset-metadata.json").read_text())
    assert metadata["id"] == "ethercess/ios-risk-eval-assets-v3"


def test_prompt_formatter_is_self_contained_and_matches_training_shape():
    prompt = format_prompt("Assess the transaction.", "Amount: $12")
    assert "### Instruction:\nAssess the transaction." in prompt
    assert "### Input:\nAmount: $12" in prompt
    assert prompt.endswith("### Response:\n")


def test_eval_notebook_verdict_handles_rows_without_targets(capsys, tmp_path):
    notebook = json.loads((ROOT / "notebooks" / "03_eval_results.ipynb").read_text())
    verdict = "".join(notebook["cells"][12]["source"])
    verdict = verdict.replace(
        '"/kaggle/working/eval_results/comparison.json"',
        repr(str(tmp_path / "comparison.json")),
    )
    risk = {
        "tier_accuracy": 0.8,
        "avg_quality": 0.7,
        "evidence_rate": 0.6,
        "action_accuracy": 0.5,
        "unsupported_claim_rate": 0.0,
    }
    classification = {
        "precision": 0.8,
        "recall": 0.7,
        "f1": 0.75,
        "unparseable": 0,
    }
    namespace = {
        "base_summary": {
            "risk_assessment": risk,
            "classification": classification,
            "regulatory_recall": {"citation_accuracy": 0.2},
        },
        "tuned_summary": {
            "risk_assessment": risk,
            "classification": classification,
            "regulatory_recall": {"citation_accuracy": 0.3},
            "passes_project03": True,
        },
    }
    exec(verdict, namespace)
    assert "PROJECT 03: PASS" in capsys.readouterr().out


def test_every_eval_notebook_code_cell_compiles():
    notebook = json.loads((ROOT / "notebooks" / "03_eval_results.ipynb").read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if any(line.lstrip().startswith("!") for line in source.splitlines()):
                continue
            compile(source, f"eval-cell-{index}", "exec")


def test_eval_notebook_preflight_accepts_only_exact_verified_mounts(tmp_path, capsys):
    input_root = tmp_path / "input"
    asset_mount = input_root / "datasets" / "ethercess" / "ios-risk-eval-assets-v3"
    build_eval_assets(asset_mount)

    # A stale lookalike must not be selected merely because its filenames match.
    stale_mount = input_root / "datasets" / "ethercess" / "ios-risk-eval-assets"
    stale_mount.mkdir(parents=True)
    for filename in ("domain_eval.py", "testset.json", "bundle_manifest.json"):
        (stale_mount / filename).write_text("stale")

    adapter_mount = (
        input_root
        / "notebooks"
        / "ethercess"
        / "ios-risk-brain-v1-fine-tune"
        / "Llama-3.1-8B-IOS-Risk-v1"
    )
    adapter_mount.mkdir(parents=True)
    (adapter_mount / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": (
                    "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit"
                )
            }
        )
    )
    weights = adapter_mount / "adapter_model.safetensors"
    with weights.open("wb") as handle:
        handle.seek(10_000_000)
        handle.write(b"\0")

    notebook = json.loads((ROOT / "notebooks" / "03_eval_results.ipynb").read_text())
    preflight = "".join(notebook["cells"][4]["source"])
    preflight = preflight.replace('"/kaggle/input"', repr(str(input_root)))
    namespace = {}
    exec(preflight, namespace)

    assert namespace["assets"] == str(asset_mount)
    assert namespace["ADAPTER"] == str(adapter_mount)
    assert namespace["testset"]["counts"] == EXPECTED_COUNTS
    assert "All isolated inputs validated" in capsys.readouterr().out
