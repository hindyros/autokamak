# Physics primer

Coder-friendly reference for the physics `autotokamak` computes. Written to help someone without a plasma-physics background spot obvious errors in a config or diagnostic.

## The equation

TokaMaker solves the **Grad–Shafranov equation** for axisymmetric MHD equilibrium in the 2D poloidal (R, Z) plane:

```
Δ* ψ(R, Z) = − μ₀ R² p'(ψ) − F(ψ) F'(ψ)
```

- `ψ(R, Z)` — poloidal flux function. Level sets are the magnetic flux surfaces.
- `Δ*` — the "GS operator", closely related to the ordinary Laplacian but with an extra 1/R factor coming from the R-derivative term of a cylindrically symmetric ∇².
- `p(ψ)` — pressure profile; `p'(ψ) = dp/dψ`.
- `F(ψ) = R B_φ` — poloidal current function (`B_φ` is the toroidal field).
- `μ₀` — vacuum permeability.

Boundary condition: `ψ` prescribed on the LCFS (fixed-boundary) or determined by an external coil set (free-boundary — not supported yet in this repo).

## Coordinates

- **R** — major radius, distance from the machine axis (in metres).
- **Z** — height above the machine midplane (in metres).
- **φ** — toroidal angle. Not solved for; axisymmetry means every ψ contour is a torus.

## What the LCFS is

**Last Closed Flux Surface** — the outermost ψ contour on which magnetic-field lines still close on themselves. Everything inside is "confined plasma"; everything outside is "scrape-off layer" or vacuum.

In fixed-boundary mode (this repo's default), the user provides the LCFS shape and the solver finds ψ inside it. `autotokamak.core.geometry.build_lcfs` builds an analytic D-shape from five parameters:

- **R0** — major radius of the LCFS centre.
- **Z0** — vertical centre.
- **a** — minor radius (half-width).
- **κ** (kappa) — elongation. 1.0 = circle; 1.6 = ITER-like; >2 gets numerically hard.
- **δ** (delta) — triangularity. 0.0 = symmetric; positive = "D-shape" (inward on the inboard side); negative "reversed-D" is possible but rare.

## The isoflux constraint

Once you have an LCFS shape, you can either (a) let TokaMaker guess ψ freely everywhere and hope the boundary matches, or (b) explicitly constrain ψ to be constant along the LCFS points (`gs.set_isoflux(lcfs)`). Option (b) is what "isoflux" means in this codebase.

Isoflux is more accurate but **fragile**: extreme shapes or badly-seeded ψ cause OFT to throw at construction time. `autotokamak.core.solver.solve_equilibrium` catches that, drops the constraint, and re-solves. The `get_last_solve_info()` call reports whether isoflux was used (`isoflux_used: True`) or the unconstrained fallback ran (`isoflux_used: False`).

**When fallback fires on a dataset sample, that sample's geometry inputs no longer describe the saved ψ.** No downstream surrogate hyperparameter can recover from this — you have to regenerate cleanly.

## Diagnostics that matter

`autotokamak.core.diagnostics.extract_scalars` returns:

- **R_axis, Z_axis** — location of the magnetic axis (the ψ extremum inside the LCFS). Physical range: `R_axis ≈ R0`, `|Z_axis| < 0.1 * a`. Off-axis means the equilibrium didn't converge or the shape is asymmetric.
- **q_0, q_95, q_edge** — safety factor at the axis, at 95% flux, at the edge. Physical ranges: `q_0 ∈ [0.8, 1.5]` (kink stability), `q_95 ∈ [2, 8]` (headline stability metric), `q_edge` finite and > q_95. `q_95 = NaN` means the profile extractor failed — see `debugging.md`.
- **p_axis, p_edge** — pressure at the axis and at the edge. `p_edge` should be ≪ `p_axis` (pressure is highest in the core).

## Shipped feasible box

From `dataset_config.yaml` (the current Phase-1 sweep bounds):

| Knob | Low | High | Units |
|---|---|---|---|
| R0 | 0.35 | 0.55 | m |
| a  | 0.10 | 0.20 | m |
| κ  | 1.0  | 1.6  | — |
| δ  | 0.0  | 0.4  | — |
| Ip | 80,000 | 200,000 | A |

Fixed: `z0=0`, `F0=0.10752 T·m`, `npts=80`, `mesh_dx=0.015`, `solver.order=1`, `Ip_ratio=1.0`.

Aspect ratio `R0/a ∈ [1.75, 5.5]`. The low end (1.75) is spherical-tokamak-ish; the high end is conventional-tokamak-ish. Both are valid.

Isoflux success rate inside this box is high but not 100% — the corners (high-κ, high-δ, small-a) fail more. Run `scripts/probe_feasible.py` to measure current rates before scaling `n_samples`.

## The output grid

For dataset generation, ψ is interpolated from the FEM mesh onto a rectangular (R, Z) grid so the downstream ML pipeline can treat it as an image tensor.

Shipped grid: `R ∈ [0.15, 0.80]` with 64 points, `Z ∈ [-0.40, 0.40]` with 96 points. That's `96 × 64 = 6144` output pixels per sample.

Pixels outside the LCFS carry `NaN` (no plasma there). The surrogate treats them as "don't care"; a per-sample RMSE metric masks them out.

## When a config looks physically wrong

- `kappa > 2`: numerically hard; usually means the config author made a typo.
- `delta > 0.6`: shape starts self-intersecting; `npts` must be very high to mesh it.
- `Ip < 10,000 A`: unusually low for a tokamak; check units — `Ip` is in **amperes**, not kA.
- `F0 = 0`: means no toroidal field. Physically valid (RFP-like) but not what TokaMaker is tuned for; expect solver to complain.
- Aspect ratio `R0/a < 1.2`: extremely spherical; below TokaMaker's tested range.

## Further reading (external)

Not linked from the Skill because it's portable, but for a real physics reference: Freidberg's *Ideal MHD* Ch. 6 (Grad–Shafranov derivation) and Wesson's *Tokamaks* Ch. 3–4 (equilibrium and stability, the source for the q-profile numbers above).
