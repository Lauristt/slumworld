# slumworld

> A research-oriented machine learning workspace for **slum / informal settlement detection** using satellite imagery.  
> Designed for **clean experimentation**, **HPC scalability**, and **reproducible inference pipelines**.

---

## Repository Overview

This repo follows a **modular ML research layout**, separating *core logic*, *experiment execution*, and *cluster orchestration*.

```text
slumworld/
├── slumworldML/          # Main Python package
│   ├── src/              # Core ML / data / model logic
│   ├── runners/          # Training & inference entrypoints
│   ├── slurm/            # SLURM / HPC job scripts
│   └── __init__.py
├── main.py               # Lightweight entry script
├── pyproject.toml        # Dependency & project configuration
└── uv.lock               # Locked environment (uv)
```

---

## Quick Start

### 1. Environment Setup

Recommended (using **uv**):

```bash
uv sync
```

Fallback (pip + venv):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

---

### 2. Run

```bash
python main.py
```

or run specific experiments via scripts in `slumworldML/runners/`.

---

## `slumworldML/` Package Structure

### `src/` — Core Implementation Layer

This directory contains **all reusable and testable logic**, including:

- Data loading & preprocessing
- Model architectures (e.g. segmentation models)
- Training / inference utilities
- Evaluation & post-processing logic

**Design principle:**  
Code in `src/` should be *pure*, importable, and environment-agnostic.

> If it defines *how something works*, it belongs in `src/`.

---

### `runners/` — Experiment & Inference Entrypoints

This folder hosts **thin execution scripts** that assemble components from `src/`.

Typical responsibilities:
- Parse experiment parameters
- Launch training or inference jobs
- Handle I/O paths and logging

Characteristics:
- Minimal logic
- Easy to modify per experiment
- Safe to run locally or on compute nodes

> If a script *does something*, it belongs in `runners/`.

---

### `slurm/` — HPC / Cluster Orchestration

This directory contains **SLURM job submission scripts** for large-scale runs.

Includes:
- Resource specifications (GPU / CPU / memory / walltime)
- Environment activation
- Calls into `runners/` scripts

Typical workflow:
1. Adjust parameters in a SLURM script
2. Submit with `sbatch`
3. SLURM executes the corresponding runner

This keeps **infrastructure concerns fully isolated** from ML logic.

---

## Architectural Philosophy

The project enforces a clear **three-layer separation**:

| Layer | Responsibility |
|------|----------------|
| `src/` | What the model *is* |
| `runners/` | What the experiment *does* |
| `slurm/` | Where & how it *runs* |

This structure scales cleanly from:
- Local prototyping  
- Multi-GPU servers  
- Shared academic HPC clusters  

---

## Development Notes

- Keep business logic inside `src/`
- Avoid hard-coding paths in `runners/`
- Treat `slurm/` as infrastructure-only
- Prefer explicit configs over implicit globals

---

## License

No license file is currently included.  
Consider adding one (MIT / Apache-2.0) if you plan to share or reuse this code publicly.

---

*Maintained by @Lauristt*
