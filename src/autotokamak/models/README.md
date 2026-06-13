# `autotokamak.models`

Trained-model wrappers — load a checkpoint, return a `predict(config) -> psi` callable.

Populated in **Week 4–6**. Planned modules:

| Module | Purpose |
|---|---|
| `loader.py` | Load a checkpoint from `models/checkpoints/`, return the right model class |
| `wrapper.py` | A common `SurrogateModel` interface: `.predict(cfg) -> np.ndarray` |

Checkpoint files (weights) live at repo root in `models/checkpoints/`, not inside this package.
