# Surrogate model candidates (priority order)

Status: living checklist for manual / agent experiments beyond the PoC zoo.
Companion to [`search_space.md`](search_space.md) and [`project_agenda.md`](project_agenda.md).
Manual walkthrough: [`../notebooks/manual_surrogate.ipynb`](../notebooks/manual_surrogate.ipynb).

**Assumption for all entries below:** keep the existing pipeline

```text
X (N, 5)  →  StandardScaler  →  model  →  PCA coeffs (N, n_pca)  →  inverse_transform  →  ψ (N, nz, nr)
```

Score in **physical ψ space** with `psi_rmse`, and always report **ratio vs mean baseline**
(`baseline_mean_predictor_rmse`). Ratio ≪ 1 is the real win condition.

**Before chasing models:** fix data first. If `isoflux_used` is ~0 or `N ≲ 50`,
stronger models usually still collapse to the mean. Prefer `N ≥ 200–500` and a
healthy isoflux success rate, then walk this list top-down.

---

## Currently in the repo (PoC zoo)

Defined in `autotokamak.surrogate.zoo` — keep as baselines, not as the ceiling:

| Name | Notes |
|---|---|
| `poly_ridge` | Fast, strong default on low-rank ψ |
| `kernel_ridge` | Cheap RBF / Laplacian KRR |
| `gp` | Isotropic RBF + WhiteKernel; small-N friendly |
| `mlp` | sklearn `MLPRegressor`, ≤2×256; often weak |

PINN / DeepONet / FNO remain **out of PoC scope** per `project_agenda.md` until
the classical / boosting stack clearly beats baseline.

---

## Priority to test next

Ordered by expected gain ÷ train/run cost on a laptop (PCA-coeff regression).

### P1 — Try first (seconds–~1 min)

| # | Model | How to fit | Why |
|---|---|---|---|
| 1 | **HistGradientBoosting** (sklearn) or **LightGBM / XGBoost** | One booster per PCA coeff (or `MultiOutputRegressor`) | Best tabular learners for 5→`n_pca`; usually beats poly/MLP with little tuning |
| 2 | **ExtraTrees / RandomForest** | Multi-output on PCA coeffs | Robust, almost no tuning; hard to do worse than mean at small–medium N |
| 3 | **PLS (Partial Least Squares)** | `PLSRegression` on PCA coeffs or directly on flattened ψ (NaN-filled) | Built for multi-output; near-instant baseline that often beats poly ridge |

### P2 — Try next (seconds–few minutes)

| # | Model | How to fit | Why |
|---|---|---|---|
| 4 | **GP with Matérn + ARD** | Anisotropic length scales (one per input dim) | Fixes isotropic-RBF pathology when `Ip` dominates `r0`/`a` |
| 5 | **Independent GP / ridge per PCA mode** | Separate regressor per coefficient | Lets each spatial mode have its own smoothness; cheap at `n_pca ≈ 8–20` |
| 6 | **Kernel ridge with Laplacian + tuned γ** | Same zoo, tighter search / input scaling check | Cheap ablation if boosting helps but you want a sklearn-only stack |

### P3 — Only after P1–P2 beat baseline and N is larger

| # | Model | When | Cost |
|---|---|---|---|
| 7 | **Small PyTorch MLP / ResNet on PCA coeffs** | N ≳ 300; sklearn MLP already plateaued | minutes |
| 8 | **POD + DeepONet (branch/trunk)** | Want operator-style generalization across shapes | GPU-ish; needs more data |
| 9 | **FNO / U-Net on ψ grid** | Grid-structured residual errors dominate | heavier train; only if PCA bottleneck is proven |
| 10 | **PINN / physics-informed loss** | Hard physics constraints matter more than pure fit | slow; different eval story |

---

## Suggested test protocol

1. Freeze a clean `dataset.h5` (`N`, isoflux rate, grid recorded in the notebook / report).
2. Fit PCA on train only; pick `n_pca` from the cumulative-variance curve (~95–99%).
3. Run **current zoo** + **P1 models** with the same splits (`kfold`, same seed).
4. Rank by CV `psi_rmse` and **ratio vs baseline**; promote winners to held-out test once.
5. Only then widen to P2 / P3 — do not jump to FNO/DeepONet while ratio ≈ 1.

### Minimal success bar

- CV and test **ratio &lt; 0.7** vs mean baseline on a clean dataset → worth extending the agent zoo.
- Ratio ≈ 1.0 across P1 → stop changing models; inspect Phase-1 labels / isoflux / N.

---

## Implementation notes (when adding to the package)

- Prefer new factories in `src/autotokamak/surrogate/zoo.py` (or a sibling module) so the
  notebook, Optuna loop, and agent share one entry point.
- Keep **input `StandardScaler`** in the pipeline — raw `Ip ~ 1e5` vs `r0 ~ 0.4` breaks kernels/GPs.
- Multi-output boosting: start with **one model per PCA coeff**; shared multi-output trees are a later ablation.
- Persist winners in the same `winner.pkl` payload shape (`estimator`, `pca`, `model_name`, …)
  so `predict_with_winner` / eval plots keep working.
