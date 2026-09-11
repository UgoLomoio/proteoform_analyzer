# Deploying the Proteoform Analyzer to a Hugging Face ZeroGPU Space

This guide walks you through publishing the app as a **Gradio Space on Hugging
Face** using **ZeroGPU** (on-demand GPU allocation). It covers the two edited
entry files (`app.py`, root `requirements.txt`), the LFS setup for the large
vendored weights, the automated `deploy_to_hf.sh` script, and — importantly — an
**honest matrix of what actually runs on ZeroGPU** vs. what needs extra setup.

---

## 0. TL;DR

```bash
# one-time setup
pip install "huggingface_hub[cli]"
git lfs install
huggingface-cli login          # paste a WRITE token

# from the repo root (the folder that contains app.py)
./deploy_to_hf.sh <your-username>/proteoform-analyzer
```

Then open the Space, and under **Settings → Hardware** confirm it is set to a
**ZeroGPU** tier (e.g. `zero-a10g`). Add a `BOLTZ_API_KEY` secret if you want
Boltz-2 folding/docking.

---

## 1. Repository layout the Space expects

Hugging Face Spaces automatically runs **`app.py` at the repository root**. The
application itself is the `proteoform_analyzer/` Python package. The repo must
therefore look like this (this is exactly the tree in this zip):

```
<repo root>/
├── app.py                     # HF entry point (ZeroGPU-safe)  ← edited/added
├── requirements.txt           # LEAN Space deps                 ← edited/added
├── packages.txt               # apt packages (default-jre, fpocket)  ← added
├── README.md                  # Space card with YAML config header   ← added
├── .gitattributes             # Git LFS rules for the weights         ← added
├── deploy_to_hf.sh            # this deploy script                    ← added
├── DEPLOY_HF_ZEROGPU.md       # this guide                            ← added
├── CHANGES_zerogpu.md         # summary of all code changes           ← added
└── proteoform_analyzer/       # the full application package (unchanged assets)
    ├── __init__.py, gui.py, cli.py
    ├── core/  (steps/, config, pipeline, viz, …)
    └── _vendored/  (RFantibody, ThermoMPNN, p2rank, diffsbdd, TMalign, …)
```

`app.py` puts the repo root on `sys.path` and imports the code as
`proteoform_analyzer.gui`, so the package's relative imports resolve.

---

## 2. One-time local setup

1. **Install the Hugging Face CLI and Git LFS**

   ```bash
   pip install "huggingface_hub[cli]"
   git lfs install
   ```

2. **Log in with a WRITE token**

   Create a token at <https://huggingface.co/settings/tokens> (role: *Write*),
   then:

   ```bash
   huggingface-cli login
   ```

---

## 3. Deploy (automated)

From the repo root:

```bash
./deploy_to_hf.sh <your-username>/<space-name>
# example:
./deploy_to_hf.sh ugolomoio/proteoform-analyzer
```

The script:

1. removes the nested `_vendored/RFantibody/.git` repo and all `__pycache__`
   (a nested `.git` will otherwise break the push);
2. **re-initialises git from a clean state** (removes any existing `.git`) so
   the large binaries are committed *through* Git LFS on the very first `git
   add` — a prior non-LFS commit would otherwise persist and the push would be
   rejected;
3. initialises **Git LFS** and verifies that **every binary file (any size) and
   every file > 10 MB** is matched by a pattern in `.gitattributes`, aborting
   with the offending list if one is not. Note: Hugging Face's Xet backend
   rejects *any* untracked binary regardless of size — a 2–4 MB executable or
   `.pkl.gz` is enough to fail the push — so a size-only check is not
   sufficient;
4. creates the Space (`--repo-type space --space_sdk gradio`) if it does not
   exist;
5. commits and **force-pushes** to the Space's `main` branch (this uploads the
   ~1.8 GB of LFS weights — expect it to take a while on a slow connection).

**Dry run first** (prepares + commits locally, prints the push command, but does
NOT upload):

```bash
DRY_RUN=1 ./deploy_to_hf.sh <your-username>/<space-name>
```

---

## 4. Deploy (manual, if you prefer)

```bash
# from the repo root
find proteoform_analyzer -type d -name .git -exec rm -rf {} +   # drop nested git
find . -type d -name __pycache__ -exec rm -rf {} +

# Start from a CLEAN history: if you pushed before WITHOUT LFS, that old commit
# must not survive, or the binaries stay un-LFS'd and the push is rejected again.
rm -rf .git

git lfs install
git init
git add .gitattributes          # stage the LFS rules FIRST, before any binary
git commit -m "Add LFS tracking rules"
git add -A                      # now every matched binary is routed through LFS
git commit -m "Deploy Proteoform Analyzer (ZeroGPU)"

# Sanity check: this MUST list your weights as 'lfs' (not empty / not 'text').
git lfs ls-files | head

huggingface-cli repo create <user>/<space> --repo-type space --space_sdk gradio -y
git remote add space https://huggingface.co/spaces/<user>/<space>
git push --force space HEAD:main
```

---

## 5. Configure the Space

In the Space UI → **Settings**:

- **Hardware**: choose a **ZeroGPU** tier (e.g. `zero-a10g`). The Space always
  builds and starts on CPU; ZeroGPU attaches a GPU only while a `@spaces.GPU`
  function runs.
- **Variables and secrets** (optional):
  - `BOLTZ_API_KEY` — enables Boltz-2 folding & docking via the hosted Boltz
    API. Also add `boltz-api` to `requirements.txt`. Without it, structures fall
    back to WT-backbone side-chain grafting (TM-score == 1.0 by construction;
    the app shows a banner explaining this).

---

## 6. What actually runs on ZeroGPU (be honest with yourself)

ZeroGPU accelerates only work that runs a torch model **in the Space process**
inside a `@spaces.GPU`-wrapped call. In this app that is the ESM2 embedding and
the ESM2 zero-shot ddG scorer (`app.py` wraps them). Everything else is CPU, an
external API, or an external binary.

| Feature | Runs on the default Space? | GPU-accelerated? | Notes |
|---|---|---|---|
| Sequence / mutation / proteoform enumeration | ✅ | — | CPU |
| **ESM2 embeddings + UMAP** | ✅ | ✅ ZeroGPU | wrapped `@spaces.GPU` |
| **ESM2 zero-shot ddG** | ✅ | ✅ ZeroGPU | wrapped `@spaces.GPU` |
| Protein-contact-network (PCN) | ✅ | — | CPU |
| TM-score, impact score, tables, plots | ✅ | — | CPU |
| 3D viewer (incl. docked / antibody complexes) | ✅ | — | renders result files |
| **ThermoMPNN ddG** | ⚠️ | — | needs a ThermoMPNN checkout configured; else falls back to ESM2 ddG |
| **Boltz-2 folding / docking** | ⚠️ | via API | set `BOLTZ_API_KEY` + add `boltz-api`; else WT-graft fallback (fold) / clean skip (dock) |
| **AutoDock Vina docking** | ❌ | — | needs the `vina` binary + `meeko`; not installed by default |
| **Molecular Dynamics** | ❌ | — | needs `openmm`/`mdtraj`; not installed by default |
| **DiffSBDD / GNN ligand design** | ❌ | — | needs `torch-scatter`/`dgl`/`openbabel`; not installed by default |
| **RFAntibody antibody design** | ❌ | — | needs a local RFAntibody checkout + GPU subprocess |

Legend: ✅ works out of the box · ⚠️ works with the noted configuration ·
❌ intentionally omitted from the default Space build.

To enable an ❌/⚠️ feature, add its Python packages to `requirements.txt`, its
native tools to `packages.txt` (or a custom `Dockerfile`), and set any required
secrets. The lean default keeps the Space building quickly and reliably.

---

## 7. Notes on the large weights (~1.8 GB)

The `_vendored/` tree ships RFantibody (~1.6 GB of `.pt`), ThermoMPNN
(`thermoMPNN_default.pt`, and 20–36 MB training CSVs), P2Rank models (`.zst`),
and the DiffSBDD checkpoint (`.ckpt`). `.gitattributes` tracks all of these via
Git LFS. If you do **not** intend to run the antibody / DiffSBDD / P2Rank steps
on the Space, you can shrink the upload dramatically by deleting those
subfolders before pushing:

```bash
rm -rf proteoform_analyzer/_vendored/RFantibody/weights   # ~1.6 GB
rm -rf proteoform_analyzer/_vendored/diffsbdd/checkpoints  # ~77 MB
```

The app degrades cleanly if a weight is absent (the corresponding step skips
with a message).

---

## 8. Troubleshooting

- **Push rejected: "your push was rejected because it contains binary files"
  (HF Xet)** — this is the most common failure. HF rejects *any* binary that is
  not tracked by Git LFS, **regardless of size** (files as small as 2–4 MB, e.g.
  an extension-less executable like `tmalign_exe/TMalign_cpp` or a `.pkl.gz`,
  trigger it). Two things must both be true:
  1. **A pattern matches the file.** Every binary type shipped here is already
     covered by `.gitattributes` (`*.pt *.pth *.ckpt *.zst *.jar *.bin *.model
     *.npy *.npz *.h5 *.pkl *.gz *.pdf *.png`, the big ThermoMPNN CSVs, and the
     extension-less `**/tmalign_exe/TMalign_cpp`). If you add a *new* binary of a
     type not listed, add a matching line first. Verify with:
     `git check-attr filter -- <path>`  → must print `filter: lfs`.
  2. **The file was committed AFTER the pattern existed.** LFS filters apply at
     `git add` time, so a binary committed before `.gitattributes` was in effect
     stays a plain blob and keeps getting rejected. `git commit --amend` does
     **not** fix this. Either re-init from a clean history (what
     `deploy_to_hf.sh` now does automatically — just re-run it), or rewrite the
     existing history in place:
     ```bash
     git lfs migrate import --include="*.gz,*.pdf,**/tmalign_exe/TMalign_cpp" --everything
     # (or simply: rm -rf .git && re-run ./deploy_to_hf.sh)
     ```
  Confirm the weights are actually in LFS before pushing: `git lfs ls-files`
  should list them.
- **Build fails on a heavy wheel** (`torch-scatter`, `dgl`, `openbabel`) — those
  are not in the lean `requirements.txt`; if you added one back, either pin a
  wheel compatible with the Space's Python/torch or move it to a `Dockerfile`.
- **`spaces` import error locally** — expected off-platform; `app.py` installs a
  no-op shim so it runs anyway. On the Space the real `spaces` package is
  provided by the ZeroGPU runtime and via `requirements.txt`.
- **App builds but no GPU speedup** — confirm Hardware is a ZeroGPU tier and
  that you are exercising the ESM2 UMAP / ddG steps (the only in-process GPU
  work).
```
