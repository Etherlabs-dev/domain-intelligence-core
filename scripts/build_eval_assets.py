"""Build the exact self-contained Kaggle Project 03 evaluation asset bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from eval.domain_eval import EVAL_BUNDLE_VERSION, validate_testset_file

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ("domain_eval.py", "testset.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output_dir: str | Path) -> Path:
    """Create a fresh upload directory and verify every bundled artifact."""
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to mix eval assets with a non-empty directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    for name in SOURCE_FILES:
        shutil.copy2(ROOT / "eval" / name, output / name)

    # Re-run the same frozen contract against the copied file, not the source.
    testset = validate_testset_file(output / "testset.json")
    manifest = {
        "bundle_version": EVAL_BUNDLE_VERSION,
        "case_count": len(testset["cases"]),
        "files": {name: _sha256(output / name) for name in SOURCE_FILES},
    }
    (output / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "IOS Risk Eval Assets V3",
                "id": "ethercess/ios-risk-eval-assets-v3",
                "licenses": [{"name": "unknown"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Built verified Kaggle eval bundle {EVAL_BUNDLE_VERSION} at {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build(args.out)
