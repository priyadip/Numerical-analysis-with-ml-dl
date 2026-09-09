"""Shared pytest configuration.

Puts `src/` on the import path so `import nalib` works without installing the package,
and provides fixtures that several test modules share.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

#: Every test that uses randomness uses this seed, so failures are reproducible.
SEED = 42


@pytest.fixture
def rng():
    """A seeded generator. Same numbers on every run, on every machine."""
    return np.random.default_rng(SEED)


@pytest.fixture
def unit_roundoff():
    return float(np.finfo(float).eps) / 2
