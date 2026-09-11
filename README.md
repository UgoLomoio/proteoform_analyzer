---
title: Proteoform Analyzer
emoji: 🧬
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.39.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
suggested_hardware: zero-a10g
short_description: An AI-powered tool to analyze proteoform effects.
---

# Proteoform Analyzer

An AI-assisted tool to analyze the effects of single-point mutations,
post-translational modifications (PTMs), and their pairwise combinations
(proteoforms) on proteins of arbitrary oligomeric state (monomers, dimers,
tetramers, hexamers, …).

## Running on this Space (ZeroGPU)

This Space uses **ZeroGPU**: a GPU is attached only for the duration of the
in-process model calls (ESM2 sequence embedding and the ESM2 zero-shot ddG
scorer), which are wrapped with `@spaces.GPU` in `app.py`.

**What runs here out of the box (CPU/GPU, no extra setup):**

- Sequence retrieval, mutation & proteoform enumeration
- ESM2 embeddings + UMAP variant map (GPU-accelerated)
- ESM2 zero-shot ddG stability scores (GPU-accelerated)
- Protein-contact-network (PCN) centralities & communities
- TM-score comparison, impact scoring, and all tables / interactive plots
- 3D structure viewer, including docked receptor+ligand and antibody complexes
  when those result files are present

**What needs extra configuration (degrades cleanly with a message otherwise):**

- **FoldX ΔΔG** — not bundled (academic licence); see the dedicated
  [FoldX installation](#foldx-installation-required-for-foldx-ddg) section below.
- **Boltz-2 folding & docking** — set a Space secret `BOLTZ_API_KEY` (and add
  `boltz-api` to `requirements.txt`) to use the hosted Boltz API. Without a key,
  mutant/proteoform structures fall back to side-chain grafting onto the WT
  backbone (TM-score == 1.0 by construction; the app shows a prominent banner
  explaining this).
- **AutoDock Vina docking**, **Molecular Dynamics**, **DiffSBDD / GNN ligand
  design**, and **RFAntibody antibody design** need additional native binaries
  and/or heavy Python wheels that are not installed on the default Space build.

See **DEPLOY_HF_ZEROGPU.md** in this repo for the full deployment guide and the
"what runs where" matrix, and **CHANGES_zerogpu.md** for the list of changes
made to enable Spaces deployment.

## Local use

```bash
pip install -r requirements.txt          # lean Space deps, or:
pip install -r proteoform_analyzer/requirements.txt   # full local deps
python -m proteoform_analyzer.gui         # Gradio GUI
python -m proteoform_analyzer.cli --help  # command-line interface
```

## FoldX installation (required for FoldX ΔΔG)

FoldX predicts the change in folding free energy ($\Delta\Delta G$) of missense
variants and of PTM mimetic substitutions (phosphorylation → Glu, acetylation
→ Gln). **FoldX is *not* bundled with ProteoformAnalyzer** — including in the
Docker image — because its licence forbids redistribution. You must obtain your
own copy; the app degrades cleanly (FoldX rows are simply absent from the ΔΔG
table) if it is not configured.

1. **Download** FoldX from the official site:
   <https://foldxsuite.cemm.at/>. Academic users register for the free
   academic licence, then download the Linux binary package (e.g.
   `FoldX5_Linux64.zip`).

2. **Install** the binary and its rotamer library:

   ```bash
   unzip FoldX5_Linux64.zip
   # the binary is 64-bit Linux executable `foldx_20251209` (name varies by release)
   mkdir -p "$HOME/foldx"
   cp foldx_20251209 "$HOME/foldx/foldx"
   chmod +x "$HOME/foldx/foldx"
   # rotabase.txt must sit NEXT TO the binary (same directory);
   # it is included in the downloaded FoldX package
   cp rotabase.txt "$HOME/foldx/rotabase.txt"
   ```

   If `rotabase.txt` is missing, FoldX fails on the first `RepairPDB` call.

3. **Configure** — ProteoformAnalyzer resolves the FoldX binary in this order:

   1. the `foldx_binary` field of the analysis configuration,
   2. the `FOLDX_BINARY` environment variable,
   3. a `foldx` executable on the system `PATH`.

   Pick whichever suits your setup:

   ```bash
   # option A — environment variable (works for GUI and CLI)
   export FOLDX_BINARY="$HOME/foldx/foldx"

   # option B — GUI: paste the path in the "FoldX binary path" field
   #           on the Setup tab

   # option C — CLI flag
   python -m proteoform_analyzer.cli run ... --foldx-binary "$HOME/foldx/foldx"
   ```

   Related CLI options: `--foldx-rotabase` (explicit path to `rotabase.txt`)
   and `--foldx-n-runs` (number of BuildModel runs per structure).

4. **Verify** the installation:

   ```bash
   "$HOME/foldx/foldx" --version   # prints the FoldX version banner
   ```

   then run a small analysis and check that `ddg_summary.csv` contains rows
   with `method = foldx_mimetic` (PTM proteoforms) or FoldX variant rows
   alongside the ThermoMPNN predictions.

**Docker users:** mount your FoldX directory read-only and point
`FOLDX_BINARY` at it — no image rebuild required:

```bash
docker run --gpus all -p 7860:7860 \
    -v "$HOME/foldx:/opt/foldx:ro" \
    -e FOLDX_BINARY=/opt/foldx/foldx \
    proteoform-analyzer
```

## Docker installation

The repository ships an all-in-one GPU-enabled **Dockerfile** (CUDA 11.8 base)
that contains the app, all native tools (Open Babel, AutoDock Vina, Java for
P2Rank), the RFAntibody environment (its own Python 3.10 venv, created at build
time by `uv sync`), and — by default — the RFAntibody model weights (~750 MB).

**Prerequisites**

- Docker 24+ (or any recent Docker Engine / Docker Desktop)
- For GPU acceleration (strongly recommended for RFAntibody / ESM2): an
  NVIDIA driver plus the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  (`docker run --gpus all ...` must work)
- ~20 GB free disk space (the final image is ~15 GB with weights)

**Build**

```bash
# full image, weights baked in (recommended)
docker build -t proteoform-analyzer .

# or, to skip the ~750 MB weights download at build time
# (they are then downloaded automatically on first antibody-design run):
docker build -t proteoform-analyzer --build-arg DOWNLOAD_WEIGHTS=false .
```

**Run**

```bash
# GPU (recommended) — GUI on http://localhost:7860
docker run --gpus all -p 7860:7860 proteoform-analyzer

# CPU-only (works, but RFAntibody/ESM2 are slow)
docker run -p 7860:7860 proteoform-analyzer

# persist results on the host (pipeline writes to /app/results)
docker run --gpus all -p 7860:7860 \
    -v "$(pwd)/results:/app/results" \
    proteoform-analyzer

# CLI instead of the GUI (override the entry command)
docker run --gpus all proteoform-analyzer \
    python3.11 -m proteoform_analyzer.cli run --fast --protein hemoglobin
```

**Notes**

- The container listens on port **7860** (override with `-e PORT=...`).
- **FoldX is not included in the image** (licence-restricted); mount it at
  runtime as shown in the
  [FoldX installation](#foldx-installation-required-for-foldx-ddg) section
  (`-v "$HOME/foldx:/opt/foldx:ro" -e FOLDX_BINARY=/opt/foldx/foldx`).
- The Boltz-2 hosted API can be enabled with
  `-e BOLTZ_API_KEY=<your-key>`; without it, structure folding falls back to
  side-chain grafting (the GUI shows a banner when this happens).
- The image runs the same `app.py` as the Hugging Face Space; the ZeroGPU
  shims auto-disable themselves off-platform.
