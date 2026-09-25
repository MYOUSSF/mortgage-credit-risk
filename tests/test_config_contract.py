"""
Contract test for src/config.py.

config.py is the single source of truth for cross-script constants, and every
numbered script binds what it needs at IMPORT time (`FIG_DIR = config.FIG_DIR`,
followed immediately by `FIG_DIR.mkdir(...)`). That makes a missing constant a
hard crash before `main()` is ever reached — and, because the binding happens
at module scope, it crashes all eleven plotting scripts identically while
leaving the ones that ran earlier in the pipeline looking fine.

That failure mode has happened: a refactor of the PATHS block to environment
variables dropped FIG_DIR and CHUNK_DIR, and scripts 02 and 03 died on
`AttributeError: module 'config' has no attribute 'FIG_DIR'` with the GPU
banner already printed, which makes it look like a runtime problem rather than
an import-time one.

This test scans the source for every `config.<NAME>` reference and asserts it
resolves, so the next such refactor fails in CI instead of on Kaggle.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# Referenced inside a docstring or comment rather than as an attribute.
_NOT_ATTRIBUTES = {"py"}


def _referenced_names(path: Path) -> set[str]:
    return {
        name for name in re.findall(r"\bconfig\.([A-Za-z_][A-Za-z0-9_]*)", path.read_text())
        if name not in _NOT_ATTRIBUTES
    }


def _script_paths():
    return sorted(p for p in SRC.glob("*.py") if p.name != "config.py")


@pytest.mark.parametrize("script", _script_paths(), ids=lambda p: p.name)
def test_every_config_reference_resolves(script):
    import config

    missing = sorted(n for n in _referenced_names(script) if not hasattr(config, n))
    assert not missing, (
        f"{script.name} references config.{{{', '.join(missing)}}}, which "
        f"config.py does not define. Every numbered script binds these at "
        f"import time, so this is an immediate crash, not a latent bug."
    )


def test_path_constants_are_present_and_are_paths():
    """
    The names the pipeline's directory scaffolding depends on. Pinned
    explicitly — the parametrised scan above only catches a name once some
    script references it, whereas these are structural.
    """
    import config

    for name in ("RAW_DIR", "MACRO_DIR", "PROC_DIR", "CHUNK_DIR", "FIG_DIR", "DATA_DIR"):
        assert hasattr(config, name), f"config.{name} is missing"
        assert isinstance(getattr(config, name), Path), f"config.{name} is not a Path"


def test_chunk_dir_sits_under_proc_dir():
    """01 deletes CHUNK_DIR after combining chunks, so it must not be
    pointed somewhere that deletion would be destructive."""
    import config

    assert config.CHUNK_DIR.parent == config.PROC_DIR


def test_path_constants_honour_environment_overrides(monkeypatch, tmp_path):
    """
    The PATHS block is environment-driven so the same code runs on Kaggle and
    locally. Reloading with MCR_DATA_DIR set must move every derived path.
    """
    import importlib
    import config

    monkeypatch.setenv("MCR_DATA_DIR", str(tmp_path))
    reloaded = importlib.reload(config)
    try:
        assert reloaded.DATA_DIR == tmp_path
        for name in ("RAW_DIR", "MACRO_DIR", "PROC_DIR", "CHUNK_DIR", "FIG_DIR"):
            assert tmp_path in getattr(reloaded, name).parents, (
                f"config.{name} ignored MCR_DATA_DIR"
            )
    finally:
        monkeypatch.delenv("MCR_DATA_DIR", raising=False)
        importlib.reload(config)
