# syntax=docker/dockerfile:1
# ============================================================================
# Proteoform Analyzer — all-in-one GPU image
#
# Builds a single image containing:
#   * the Gradio web app + CLI (python 3.11, torch 2.4.0 cu121)
#   * all native tools (Open Babel, AutoDock Vina, Java for P2Rank)
#   * the vendored RFAntibody environment (python 3.10 venv via `uv sync`)
#   * the RFAntibody model weights (~750 MB; skip with --build-arg
#     DOWNLOAD_WEIGHTS=false)
#
# NOTE — FoldX is NOT bundled: its academic licence forbids redistribution.
# Mount your own FoldX binary + rotabase.txt at runtime, e.g.:
#   docker run --gpus all -p 7860:7860 \
#       -v "$HOME/foldx:/opt/foldx:ro" \
#       -e FOLDX_BINARY=/opt/foldx/foldx \
#       proteoform-analyzer
# The app resolves the binary from config.foldx_binary -> $FOLDX_BINARY
# -> `foldx` on PATH, and expects rotabase.txt next to the binary.
#
# Build:   docker build -t proteoform-analyzer .
# Run:     docker run --gpus all -p 7860:7860 proteoform-analyzer
# See the "Docker installation" section of README.md for full instructions.
# ============================================================================

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# --- System packages ---------------------------------------------------------
# python3.11 runs the app (matches the HF Space); python3.10 is the interpreter
# uv picks for the RFAntibody venv (requires-python >=3.10,<3.13). default-jre
# runs P2Rank; openbabel/autodock-vina back the docking step.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3.11-distutils \
        python3.10 python3.10-venv python3.10-dev \
        python3-pip \
        default-jre openbabel autodock-vina \
        build-essential git wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- uv (manages the RFAntibody venv) ----------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# --- Application source ------------------------------------------------------
COPY . /app

# --- Python dependencies (app + CLI) -----------------------------------------
RUN python3.11 -m pip install --upgrade pip \
    && python3.11 -m pip install -r pre-requirements.txt \
    && python3.11 -m pip install --no-cache-dir -r requirements.txt

# --- RFAntibody environment (its own python-3.10 .venv via uv sync) ----------
# The .venv shipped in the repo is machine-specific and excluded by
# .dockerignore; `uv sync` recreates it deterministically from uv.lock.
RUN cd /app/proteoform_analyzer/_vendored/RFantibody && uv sync

# --- RFAntibody model weights (~750 MB) --------------------------------------
# Baked into the image by default so first use is instant and offline-friendly.
# Build with --build-arg DOWNLOAD_WEIGHTS=false to skip (the app then downloads
# them on first use, into a mounted volume if you want them to persist).
ARG DOWNLOAD_WEIGHTS=true
RUN if [ "$DOWNLOAD_WEIGHTS" = "true" ]; then \
        cd /app/proteoform_analyzer/_vendored/RFantibody/weights \
        && for f in RFdiffusion_Ab.pt ProteinMPNN_v48_noise_0.2.pt RF2_ab.pt; do \
               [ -f "$f" ] || wget -q "https://files.ipd.uw.edu/pub/RFantibody/$f"; \
           done ; \
    fi

# --- Runtime ------------------------------------------------------------------
ENV PROTEOFORM_ZEROGPU=0 \
    PORT=7860
# FoldX is intentionally not installed here (academic licence). Provide it at
# runtime:  -v "$HOME/foldx:/opt/foldx:ro" -e FOLDX_BINARY=/opt/foldx/foldx
EXPOSE 7860

# Results are written under /app/results by default — mount a volume to keep
# them:  docker run --gpus all -p 7860:7860 -v $(pwd)/results:/app/results ...
CMD ["python3.11", "app.py"]
