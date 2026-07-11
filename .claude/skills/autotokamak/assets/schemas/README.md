# JSON Schemas

Auto-generated exports of the Pydantic v2 config models under `autotokamak.core.schema`. Useful for editor validation and for Claude to reason about invalid configs without loading `src/autotokamak/core/schema.py` into context.

## Regenerating

From a checkout with `autotokamak` installed:

```bash
python -c '
import json, pathlib
from autotokamak.core.schema import EquilibriumConfig, SweepConfig, InvertConfig
out = pathlib.Path(".")
out.joinpath("equilibrium.schema.json").write_text(json.dumps(EquilibriumConfig.model_json_schema(), indent=2))
out.joinpath("sweep.schema.json").write_text(json.dumps(SweepConfig.model_json_schema(), indent=2))
out.joinpath("invert.schema.json").write_text(json.dumps(InvertConfig.model_json_schema(), indent=2))
'
```

Additionally, the dataset sweep uses `autotokamak.data.schema.SweepConfig` (distinct from `autotokamak.core.schema.SweepConfig`):

```bash
python -c '
import json, pathlib
from autotokamak.data.schema import SweepConfig as DataSweepConfig
pathlib.Path("dataset_sweep.schema.json").write_text(json.dumps(DataSweepConfig.model_json_schema(), indent=2))
'
```

Note the naming collision — both modules export a `SweepConfig`. `autotokamak.core.schema.SweepConfig` is the discretization-sweep shape (base_config + cases[]); `autotokamak.data.schema.SweepConfig` is the LHS dataset-sweep shape (sampling + parameters + fixed).

## Staleness

These schemas are frozen at Skill authoring time. If `schema.py` changes (a new field, a range tweak), the JSON schemas will drift until regenerated. Truth lives in the Python source — treat the JSON as advisory.
