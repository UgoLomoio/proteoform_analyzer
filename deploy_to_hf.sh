#!/usr/bin/env bash
# ============================================================================
# deploy_to_hf.sh -- push the Proteoform Analyzer to a Hugging Face ZeroGPU Space
# ----------------------------------------------------------------------------
# This script is INTERACTIVE and SAFE BY DEFAULT: it never hardcodes an account
# or token and will not push anywhere until you pass your own Space id. It
# prepares the working tree (removes nested .git dirs, __pycache__, configures
# Git LFS for the large weights) and then creates + pushes the Space.
#
# PREREQUISITES (install once):
#   pip install "huggingface_hub[cli]"
#   git lfs install
#   huggingface-cli login          # paste a WRITE token from
#                                   # https://huggingface.co/settings/tokens
#
# USAGE:
#   ./deploy_to_hf.sh <your-username>/<space-name>
#   # e.g.  ./deploy_to_hf.sh ugolomoio/proteoform-analyzer
#
# OPTIONS (environment variables):
#   HF_HARDWARE   ZeroGPU tier to request (default: zero-a10g). The Space still
#                 builds on CPU first; set hardware in the Space settings UI or
#                 with `huggingface-cli` if this account is entitled to ZeroGPU.
#   DRY_RUN=1     Prepare the tree and print the commands but do NOT push.
# ============================================================================
set -euo pipefail

REPO_ID="${1:-}"
HF_HARDWARE="${HF_HARDWARE:-zero-a10g}"
DRY_RUN="${DRY_RUN:-0}"

# Directory this script lives in == the Space repo root (contains app.py).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

err() { printf '\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[36m==>\033[0m %s\n' "$*"; }

[ -n "$REPO_ID" ] || err "Missing Space id. Usage: ./deploy_to_hf.sh <user>/<space-name>"
[ -f "$ROOT/app.py" ] || err "app.py not found in $ROOT -- run this from the repo root."
[ -d "$ROOT/proteoform_analyzer" ] || err "proteoform_analyzer/ package folder not found."

# --- tool checks -----------------------------------------------------------
command -v git >/dev/null || err "git not found."
command -v git-lfs >/dev/null 2>&1 || git lfs version >/dev/null 2>&1 || \
    err "git-lfs not found. Install it and run: git lfs install"
command -v huggingface-cli >/dev/null 2>&1 || \
    err "huggingface-cli not found. Run: pip install 'huggingface_hub[cli]'"

# --- 1) clean the tree ------------------------------------------------------
info "Removing nested .git directories (e.g. _vendored/RFantibody/.git)…"
find "$ROOT/proteoform_analyzer" -type d -name ".git" -prune -exec rm -rf {} + 2>/dev/null || true

info "Removing __pycache__ / *.pyc…"
find "$ROOT" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "$ROOT" -type f -name "*.pyc" -delete 2>/dev/null || true

# --- 2) git + LFS setup -----------------------------------------------------
info "Initialising git + LFS…"
git lfs install --local >/dev/null 2>&1 || git lfs install >/dev/null 2>&1 || true
# Start from a CLEAN git history every time. If a previous run committed the
# large binaries WITHOUT LFS (the usual cause of "your push was rejected
# because it contains binary files"), that bad history would persist and LFS
# would NOT retroactively rewrite it. Removing .git guarantees the very first
# `git add` below applies the LFS filters from .gitattributes to every file.
if [ -d "$ROOT/.git" ]; then
    info "Removing existing .git to guarantee a clean LFS-tracked history…"
    rm -rf "$ROOT/.git"
fi
git init -q
# .gitattributes already declares the LFS patterns; make sure it is staged.
[ -f "$ROOT/.gitattributes" ] || err ".gitattributes missing -- expected LFS config."
# Stage .gitattributes FIRST so its LFS rules are in effect before any binary
# is added. This is what makes `git add -A` route binaries through LFS.
git add .gitattributes

info "Verifying every BINARY file is matched by an LFS pattern…"
# NOTE: HF's Xet backend rejects ANY untracked binary file regardless of size
# (a 2-3 MB ELF binary or .pkl.gz is enough to fail the push), so we must check
# binary CONTENT, not just files >10 MB. A file is treated as binary if it has
# a NUL byte in its first 8 KB (git's own heuristic).
UNTRACKED_BIN="$(python3 - "$ROOT" <<'PY'
import os, subprocess, sys
root = sys.argv[1]
def is_binary(p):
    try:
        with open(p, "rb") as f:
            return b"\x00" in f.read(8192)
    except Exception:
        return False
cands = []
for dp, dn, fns in os.walk(root):
    dn[:] = [d for d in dn if d != ".git"]
    for fn in fns:
        p = os.path.join(dp, fn)
        if os.path.islink(p):
            continue
        sz = os.path.getsize(p)
        # binary of any size, OR any file >10 MB (text or not)
        if is_binary(p) or sz > 10 * 1024 * 1024:
            cands.append(os.path.relpath(p, root))
bad = []
for i in range(0, len(cands), 200):
    batch = cands[i:i+200]
    out = subprocess.run(["git", "-C", root, "check-attr", "filter"] + batch,
                         capture_output=True, text=True).stdout
    for line in out.strip().splitlines():
        parts = line.rsplit(": ", 2)
        if len(parts) == 3 and parts[2] != "lfs":
            bad.append(parts[0])
for b in bad:
    print(b)
PY
)"
if [ -n "$UNTRACKED_BIN" ]; then
    printf '\033[31mERROR:\033[0m these binary/large files are NOT covered by an LFS pattern in .gitattributes:\n'
    printf '%s\n' "$UNTRACKED_BIN" | sed 's/^/  /'
    err "Add matching patterns to .gitattributes before pushing, or the HF push WILL be rejected."
fi
info "All binary/large files are LFS-tracked."

# --- 3) create the Space (idempotent) --------------------------------------
info "Ensuring the Space '$REPO_ID' exists (sdk=gradio)…"
CREATE_CMD=(huggingface-cli repo create "$REPO_ID" --repo-type space --space_sdk gradio -y)
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: ${CREATE_CMD[*]}"
else
    "${CREATE_CMD[@]}" || info "Space may already exist -- continuing."
fi

# --- 4) commit + push -------------------------------------------------------
REMOTE_URL="https://huggingface.co/spaces/$REPO_ID"
info "Configuring remote 'space' -> $REMOTE_URL"
git remote remove space 2>/dev/null || true
git remote add space "$REMOTE_URL"

git add -A
git commit -q -m "Deploy Proteoform Analyzer (ZeroGPU) $(date -u +%Y-%m-%dT%H:%M:%SZ)" || \
    info "Nothing new to commit."

if [ "$DRY_RUN" = "1" ]; then
    info "DRY_RUN=1 -> not pushing. Tree is prepared and committed locally."
    echo "To push manually:  git push --force space HEAD:main"
    exit 0
fi

info "Pushing to $REMOTE_URL (this uploads the LFS weights; can take a while)…"
git push --force space HEAD:main

info "Done. Open: $REMOTE_URL"
info "If ZeroGPU is not auto-selected, set Hardware = '$HF_HARDWARE' in the Space Settings,"
info "and add any secrets (e.g. BOLTZ_API_KEY) under Settings > Variables and secrets."
