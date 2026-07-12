# `autotokamak.core` — public API reference

Everything under `src/autotokamak/core/` is stable enough to import from Skill scripts (though Skill scripts prefer subprocess dispatch to avoid the OFT_env singleton).

## `autotokamak.core.schema` — Pydantic v2 config models

```python
class EquilibriumConfig(BaseModel):
    equation: EquationConfig
    boundary: BoundaryConfig
    mesh: MeshConfig
    solver: SolverConfig
    targets: TargetsConfig
    init_psi: InitPsiConfig = InitPsiConfig()
    outputs: OutputsConfig = OutputsConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EquilibriumConfig": ...
```

Component models and their fields:

| Model | Fields |
|---|---|
| `EquationConfig` | `name: Literal["gs"]` |
| `BoundaryConfig` | `type: Literal["isoflux"]`, `npts: int (0..1000]`, `r0, z0, a: float`, `kappa: float [0.5..3.0]`, `delta: float [-1.0..1.0]` |
| `MeshConfig` | `method: "gs_domain"`, `regions: List[MeshRegion]` (min 1) |
| `MeshRegion` | `name: str = "plasma"`, `type: "plasma"`, `dx: float > 0`, `tag: str \| None` |
| `SolverConfig` | `order: int [1..3]`, `F0: float`, `full_domain: bool = False`, `maxits: int \| None`, `free_boundary: bool = False` |
| `TargetsConfig` | any of `Ip, Ip_ratio, pax, estore, R0, V0` (at least one required) |
| `InitPsiConfig` | `method: Literal["tokamaker_default", "isoflux"] = "tokamaker_default"` |
| `OutputsConfig` | `out_dir: str = "outputs"`, `mesh_png: str = "mesh.png"`, `psi_png: str = "psi.png"` (`extra="allow"`) |

Also exported: `SweepConfig` (base + list of `CaseOverride`), `InvertConfig` (base + target ψ + optimize block), each with `from_yaml`.

**Gotchas:**
- `boundary.kappa` clamps to [0.5, 3.0] — an "obviously D-shaped" κ of 4.0 will fail validation.
- `TargetsConfig` uses a `@model_validator` — must set at least one of `Ip / Ip_ratio / pax / estore / R0 / V0` or the config rejects.
- `OutputsConfig(extra="allow")` — legacy prompts write unknown keys under `outputs:`; don't strip them.

## `autotokamak.core.geometry`

```python
def build_lcfs(*, r0, z0, a, kappa, delta, npts=80) -> np.ndarray:
    """Analytic D-shape via OFT's create_isoflux. Returns (npts, 2) [R, Z]."""

def build_mesh(lcfs, *, mesh_dx, region_name="plasma", region_tag="plasma"
              ) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate inside LCFS.
    Returns (gs_mesh, mesh_pts, mesh_lc, mesh_reg):
      - gs_mesh: gs_Domain handle (needed for plot_mesh)
      - mesh_pts: (N_nodes, 2)
      - mesh_lc:  (N_tri, 3) element connectivity (int)
      - mesh_reg: (N_tri,)   per-element region tag
    """

def build_mesh_from_config(cfg: dict
                          ) -> tuple[np.ndarray, Any, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper — reads cfg['boundary'] and cfg['mesh']['regions'][0]['dx']
    and returns (lcfs, gs_mesh, mesh_pts, mesh_lc, mesh_reg)."""
```

## `autotokamak.core.solver`

```python
def get_oft_env() -> OFT_env:
    """Process-wide singleton. Reads OFT_NTHREADS env var (default 2).
    Calling twice returns the same object; do NOT call OFT_env(...) directly."""

def make_solver(*, mesh_pts, mesh_lc, mesh_reg, cfg, env=None
                ) -> tuple[OFT_env, TokaMaker]:
    """Build TokaMaker, load mesh, apply settings + targets. Not yet solved.
    Pass env= for retries so the singleton isn't re-created."""

def solve_equilibrium(*, mesh_pts, mesh_lc, mesh_reg, lcfs, cfg) -> TokaMaker:
    """End-to-end: create → seed psi → set isoflux → solve.
    On isoflux failure, rebuilds TokaMaker on same env and solves unconstrained.
    Check get_last_solve_info() afterward to see which path was taken."""

def get_last_solve_info() -> dict:
    """{'isoflux_used': bool|None, 'fallback_reason': str|None}
    isoflux_used=False means the unconstrained fallback ran — geometry inputs
    do NOT faithfully describe the resulting psi."""
```

## `autotokamak.core.diagnostics`

```python
def extract_scalars(gs: TokaMaker) -> dict:
    """Best-effort scalar diagnostics. Tolerant to OFT API drift.
    Keys (when available):
      R_axis, Z_axis, axis_source     — magnetic axis
      q_0, q_edge, q_95               — safety-factor profile headline values
      p_axis, p_edge                  — pressure headline values
      stats                           — full gs.get_stats() blob
    Missing keys mean OFT couldn't produce them, not that they're zero."""
```

## `autotokamak.core.io` — atomic file I/O

```python
def atomic_write_text(path: Path, text: str, *, encoding="utf-8") -> None:
    """write to temp file in same dir → fsync → os.replace"""

def atomic_savez(path: Path, **arrays) -> None:
    """Atomic .npz write. A crashed sweep never leaves a truncated file."""

def unified_output_dir(out_base: Path | str, run_id: str | None = None) -> Path:
    """Standard layout: <out_base>/<run_id>/  (creates it)."""

def utc_run_id() -> str:
    """Sortable timestamp: '20260709T151200Z'."""

def assert_nonempty_file(path: Path, *, min_bytes=16) -> None: ...
def mkdir_p(p: Path) -> None: ...
```

## `autotokamak.core.logging` — OFT-aware terminal logger

```python
def elapsed() -> str:                      # "01:23.45" since module load
def log(msg: str, log_fp=None) -> None:    # stamped console line + optional file log
def section(title: str, log_fp=None) -> None:
def kv(label: str, value, log_fp=None, *, width=14) -> None:
```

Critical: OFT's compiled Fortran/C code writes via libc stdio, which buffers separately from Python's `sys.stdout`. These helpers call `libc.fflush(NULL)` on every log so OFT banner output stays in order. Do not replace with `print()`.
