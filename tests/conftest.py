"""
Shared fixtures for the pipeline test suite.

The numbered pipeline scripts (01_data_preprocessing.py, etc.) are not
packages — they're standalone scripts meant to be run with `python 0N_*.py`.
To unit-test the pure functions inside them without running the full
pipeline, we import each script as a module via importlib.

Two things make this non-trivial:
  1. Filenames start with a digit, so `import 01_data_preprocessing` is not
     valid syntax — we load by file path instead.
  2. Several scripts configure `logging.FileHandler("*.log", ...)` and
     `Path("data/...").mkdir(...)` at MODULE level (not inside main()), so
     merely importing them writes files relative to the current working
     directory. We import with cwd pointed at tests/.scratch/ so test runs
     don't drop log files or empty data/ dirs into the repo root.

Every pipeline script does `import config` (the shared config.py at the
repo root), so ROOT must be on sys.path before any of them are imported —
`python -m pytest` from the repo root adds cwd automatically, but plain
`pytest` does not, so we add it explicitly here for robustness.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path(__file__).resolve().parent / ".scratch"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(filename: str):
    module_name = filename[:-3]  # strip ".py"; sys.modules keys need not be valid identifiers
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    SCRATCH.mkdir(exist_ok=True)
    cwd = os.getcwd()
    os.chdir(SCRATCH)
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
    return module


@pytest.fixture(scope="session")
def preprocessing():
    return _load_module("01_data_preprocessing.py")


@pytest.fixture(scope="session")
def pd_logistic():
    return _load_module("02_pd_logistic_regression.py")


@pytest.fixture(scope="session")
def lgd_models():
    return _load_module("04_lgd_models.py")


@pytest.fixture(scope="session")
def survival_analysis():
    return _load_module("06_survival_analysis.py")


@pytest.fixture(scope="session")
def macro_scenario():
    return _load_module("07_macro_scenario_analysis.py")


@pytest.fixture(scope="session")
def monitoring():
    return _load_module("09_monitoring.py")


@pytest.fixture(scope="session")
def calibration():
    return _load_module("08_calibration.py")


@pytest.fixture(scope="session")
def basel_irb_capital():
    return _load_module("10_basel_irb_capital.py")
