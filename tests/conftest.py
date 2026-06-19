"""Shared pytest fixtures for SpectraLM tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def ethanol_sample() -> dict[str, Any]:
    """Return a small paired-NMR ethanol sample."""
    return {
        "id": "ethanol",
        "smiles": "CCO",
        "canonical_smiles": "CCO",
        "molecular_formula": "C2H6O",
        "murcko_scaffold": "acyclic:CCO",
        "13C_NMR": {
            "frequency": "101 MHz",
            "solvent": "CDCl3",
            "peaks": [{"shift": 58.1}, {"shift": 18.2}],
        },
        "1H_NMR": {
            "frequency": "400 MHz",
            "solvent": "CDCl3",
            "peaks": [
                {
                    "shift": 3.65,
                    "multiplicity": "q",
                    "J": [7.0],
                    "integration": 2.0,
                },
                {
                    "shift": 1.18,
                    "multiplicity": "t",
                    "J": [7.0],
                    "integration": 3.0,
                },
                {
                    "shift": 2.0,
                    "multiplicity": "brs",
                    "J": [],
                    "integration": 1.0,
                },
            ],
        },
    }
