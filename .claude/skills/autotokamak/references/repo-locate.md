# Repo-locate protocol

The Skill is portable — it lives at `~/.claude/skills/autotokamak/`, not inside the repo. Every action script must resolve `$AUTOTOKAMAK_ROOT` before touching any repo path.

## Resolution order

```python
def locate_autotokamak_root() -> Path | None:
    # 1. explicit env var wins
    env = os.environ.get("AUTOTOKAMAK_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if _looks_like_autotokamak(p):
            return p

    # 2. walk up from cwd
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        if _looks_like_autotokamak(parent):
            return parent

    # 3. not found — caller enters read-only advisory mode
    return None


def _looks_like_autotokamak(p: Path) -> bool:
    pyproject = p / "pyproject.toml"
    if not pyproject.is_file():
        return False
    # match either  name = "autotokamak"  or  name = "autokamak"
    text = pyproject.read_text(errors="replace")
    return 'name = "autotokamak"' in text or 'name = "autokamak"' in text
```

## Read-only advisory mode

When `locate_autotokamak_root()` returns `None`, wrappers:

1. Print to stdout: `autotokamak: no checkout detected — read-only advisory mode`.
2. **Do not** attempt to run a solve, edit repo files, or spawn subprocesses that assume the repo exists.
3. Still allow reference-material questions to be answered from the Skill's `references/` directory.
4. Exit 0 (never non-zero) — this is not an error condition, it's a supported operational mode.

## What "success" prints

Every wrapper's first line of stdout should be:

```
autotokamak: root=<absolute-path>  cwd=<absolute-path>  python=<version>
```

Downstream JSON summary blocks come after.
