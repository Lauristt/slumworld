# slumworld

> A lightweight Python workspace for experimenting with **slum mapping / informal-settlement detection** workflows (satellite imagery → model inference → artifacts).
>
> Repo: `Lauristt/slumworld` citeturn1view0

---

## What’s in this repo

At the moment, the repository is intentionally small and code-focused:

- **`slumworldML/`** — the core Python package (project logic lives here). citeturn1view0  
- **`main.py`** — an entry script / runnable entrypoint. citeturn1view0  
- **`pyproject.toml`** + **`uv.lock`** — modern dependency management via `pyproject` + `uv`. citeturn1view0  

```text
slumworld/
├─ slumworldML/
├─ main.py
├─ pyproject.toml
└─ uv.lock
```

---

## Quickstart

### 1) Create an environment & install dependencies

If you use **uv**:

```bash
uv sync
```

If you prefer **pip** (fallback):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

> Tip: `pyproject.toml` is the source of truth for dependencies and tooling configuration. citeturn1view0turn10search2

### 2) Run

```bash
python main.py
```

If `main.py` exposes CLI arguments, you can usually discover them with:

```bash
python main.py -h
```

---

## Typical workflow (high-level)

This project is structured to support an iterative ML workflow:

1. **Prepare inputs**: imagery tiles / metadata / labels
2. **Run experiments**: training or inference routines in `slumworldML/`
3. **Export outputs**: masks, overlays, metrics, logs

---

## Development

### Code style

- Keep `slumworldML/` as the *only* package directory (avoid drifting utilities into the repo root).
- Prefer small modules with clear I/O contracts (paths in → artifacts out).

### Updating dependencies

If you’re using `uv`:

```bash
uv add <package>
uv lock
```

---

## Notes

- This README is generated from the public repo structure visible on GitHub; if you add docs, example configs, or a CLI spec, it’s easy to expand this into a fuller “user guide”. citeturn1view0

---

## License

No license file is currently present in the repo root. Consider adding one (e.g., MIT/Apache-2.0) if you intend others to reuse the code. citeturn1view0
