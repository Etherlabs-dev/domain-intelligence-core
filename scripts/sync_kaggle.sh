#!/usr/bin/env bash
# Keeps notebooks/02_training_run.ipynb and the Kaggle notebook identical.
#
#   ./scripts/sync_kaggle.sh pull    Kaggle  -> local   (you edited in the browser)
#   ./scripts/sync_kaggle.sh push    local   -> Kaggle  (you edited in the IDE)
#   ./scripts/sync_kaggle.sh diff    show what differs, change nothing
#
# Target kernel is whatever notebooks/kernel-metadata.json points at.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NB="$ROOT/notebooks/02_training_run.ipynb"
KAGGLE="$ROOT/venv/bin/kaggle"
SLUG=$(python3 -c "import json;print(json.load(open('$ROOT/notebooks/kernel-metadata.json'))['id'])")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

# Compare code cells only — execution counts and outputs always differ.
code_only() {
  python3 -c "
import json,sys
nb=json.load(open(sys.argv[1]))
for c in nb['cells']:
    if c['cell_type']=='code':
        print('### CELL')
        print(''.join(c['source']).rstrip())
" "$1"
}

case "${1:-diff}" in
  pull)
    echo "Pulling $SLUG -> $NB"
    "$KAGGLE" kernels pull "$SLUG" -p "$TMP" >/dev/null 2>&1 || true
    src=$(find "$TMP" -name '*.ipynb' | head -1)
    [ -n "$src" ] || { echo "No notebook returned for $SLUG"; exit 1; }
    cp "$src" "$NB"
    echo "Local notebook updated from Kaggle."
    ;;
  push)
    echo "Pushing $NB -> $SLUG"
    echo
    echo "  WARNING: 'kaggle kernels push' is NOT a file sync."
    echo "  It creates a new version AND IMMEDIATELY RUNS IT — same as"
    echo "  Save & Run All. On this notebook that starts a ~3 hour GPU"
    echo "  training run and consumes weekly quota."
    echo
    echo "  Check SESSION in Step 7 before continuing. To upload code"
    echo "  without training, edit the cells in the browser instead."
    echo
    read -r -p "Type RUN to push and start training: " ans
    [ "$ans" = "RUN" ] || { echo "Aborted (nothing pushed)."; exit 1; }
    "$KAGGLE" kernels push -p "$ROOT/notebooks"
    echo
    echo "A run has started. Cancel at:"
    echo "  https://www.kaggle.com/code/$SLUG"
    ;;
  diff)
    "$KAGGLE" kernels pull "$SLUG" -p "$TMP" >/dev/null 2>&1 || true
    src=$(find "$TMP" -name '*.ipynb' | head -1)
    [ -n "$src" ] || { echo "No notebook returned for $SLUG"; exit 1; }
    code_only "$NB"  > "$TMP/local.txt"
    code_only "$src" > "$TMP/remote.txt"
    if diff -q "$TMP/local.txt" "$TMP/remote.txt" >/dev/null; then
      echo "In sync — code cells are identical."
    else
      echo "OUT OF SYNC (- local, + Kaggle):"
      diff -u --label local --label kaggle "$TMP/local.txt" "$TMP/remote.txt" || true
    fi
    ;;
  *) echo "usage: $0 {pull|push|diff}"; exit 1 ;;
esac
