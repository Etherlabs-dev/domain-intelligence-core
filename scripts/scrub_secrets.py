#!/usr/bin/env python3
"""
Rewrites hardcoded credentials into environment lookups, in place.

Run this before the first public push:  python3 scripts/scrub_secrets.py

Notebooks keep tokens inline while iterating on Kaggle (convenient, and the
Kaggle notebook is private). This turns them into env reads so the same file
is safe to publish, without touching any other code.
"""

import re
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# assignment name -> regex matching the literal secret value
SECRETS = {
    "HF_TOKEN": r"hf_[A-Za-z0-9]{34,}",
    "WANDB_API_KEY": r"wandb_v1_[A-Za-z0-9_-]{30,}",
}

TARGETS = ["notebooks/*.ipynb", "train/*.py", "eval/*.py", "inference/*.py"]


def scrub(text: str) -> tuple[str, int]:
    n = 0
    for var, value_re in SECRETS.items():
        # HF_TOKEN = "hf_xxx"   ->   HF_TOKEN = os.environ.get("HF_TOKEN", "")
        # .ipynb stores source as JSON strings, so the quotes around a literal
        # arrive escaped: HF_TOKEN = \"hf_...\". Match bare and escaped both,
        # or the scrubber silently reports success on a notebook it did not touch.
        q = r'(?:\\?["\'])'
        pattern = rf"({re.escape(var)}\s*=\s*){q}{value_re}{q}"
        text, k = re.subn(pattern, rf'\1os.environ.get("{var}", "")', text)
        n += k
        # any surviving bare literal (e.g. inline in a call)
        text, k = re.subn(
            rf'["\']{value_re}["\']', f'os.environ.get("{var}", "")', text
        )
        n += k
    return text, n


def scrub_notebook(text: str) -> tuple[str, int]:
    """Scrub code cells structurally so replacements cannot corrupt JSON."""
    notebook = json.loads(text)
    total = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        cleaned, count = scrub(source)
        if count:
            cell["source"] = cleaned.splitlines(keepends=True)
            total += count
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", total


def main() -> int:
    total = 0
    for pattern in TARGETS:
        for path in ROOT.glob(pattern):
            original = path.read_text(encoding="utf-8")
            cleaned, n = (
                scrub_notebook(original) if path.suffix == ".ipynb" else scrub(original)
            )
            if n:
                path.write_text(cleaned, encoding="utf-8")
                print(f"  {path.relative_to(ROOT)}: {n} secret(s) rewritten")
                total += n

    if total:
        print(
            f"\n{total} secret(s) scrubbed. Set them in your shell before running locally:"
        )
        for var in SECRETS:
            print(f"  export {var}=...")
        print("On Kaggle, use Add-ons -> Secrets, or paste back in temporarily.")
    else:
        print("No hardcoded secrets found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
