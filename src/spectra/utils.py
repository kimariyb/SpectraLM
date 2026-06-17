"""Line-shape functions and plotting utilities for NMR spectra."""

from __future__ import annotations

import os
import tempfile
import numpy as np

from pathlib import Path
from math import comb
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spectralm-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "spectralm-cache"))
    
from src.data.utils import normalize_multiplicity, parse_couplings


MULTIPLET_SPLITS: dict[str, list[int]] = {
    "s": [],
    "d": [1],
    "t": [2],
    "q": [3],
    "p": [4],
    "hept": [6],
    "dd": [1, 1],
    "ddd": [1, 1, 1],
    "dt": [1, 2],
    "td": [2, 1],
    "dq": [1, 3],
    "ddt": [1, 1, 2],
    "tt": [2, 2],
    "qd": [3, 1],
    "dddd": [1, 1, 1, 1],
}

BROAD_PREFIX = "br"
BROAD_LINE_WIDTH_MULTIPLIER = 3.5
DEFAULT_J_PPM = 0.010


def lorentzian(x: np.ndarray, x0: float, line_width: float) -> np.ndarray:
    """Calculate a Lorentzian line shape.

    Parameters
    ----------
    x
        Chemical shift axis.
    x0
        Peak center.
    line_width
        Full width at half maximum.

    Returns
    -------
    np.ndarray
        Line intensity values.
    """
    return (line_width / 2) ** 2 / ((x - x0) ** 2 + (line_width / 2) ** 2)


def gaussian(x: np.ndarray, x0: float, sigma: float) -> np.ndarray:
    """Calculate a Gaussian line shape.

    Parameters
    ----------
    x
        Chemical shift axis.
    x0
        Peak center.
    sigma
        Standard deviation.

    Returns
    -------
    np.ndarray
        Line intensity values.
    """
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def pseudo_voigt(x: np.ndarray, x0: float, line_width: float, eta: float = 0.5) -> np.ndarray:
    """Calculate a pseudo-Voigt line shape.

    Parameters
    ----------
    x
        Chemical shift axis.
    x0
        Peak center.
    line_width
        Full width at half maximum.
    eta
        Lorentzian mixing fraction.

    Returns
    -------
    np.ndarray
        Mixed line intensity values.
    """
    sigma = line_width / (2 * np.sqrt(2 * np.log(2)))
    return eta * lorentzian(x, x0, line_width) + (1 - eta) * gaussian(x, x0, sigma)


def add_noise(y: np.ndarray, snr: float = 80.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add Gaussian white noise to a spectrum.

    Parameters
    ----------
    y
        Spectrum intensity array.
    snr
        Signal-to-noise ratio.
    rng
        Optional NumPy random generator.

    Returns
    -------
    np.ndarray
        Noisy spectrum intensity array.
    """
    generator = rng or np.random.default_rng(42)
    peak = np.max(np.abs(y))
    sigma_noise = peak / snr if snr > 0 else 0.0
    return y + generator.normal(0, sigma_noise, size=y.shape)


def apply_splitting(
    positions: list[float],
    heights: list[float],
    coupling_ppm: float,
    equivalent_protons: int,
) -> tuple[list[float], list[float]]:
    """Apply one splitting level to existing subpeaks.

    Parameters
    ----------
    positions
        Existing peak positions.
    heights
        Existing peak heights.
    coupling_ppm
        Coupling distance in ppm.
    equivalent_protons
        Number of equivalent neighboring protons.

    Returns
    -------
    tuple[list[float], list[float]]
        New subpeak positions and heights.
    """
    line_count = equivalent_protons + 1
    binomial = [comb(equivalent_protons, idx) for idx in range(line_count)]
    new_positions: list[float] = []
    new_heights: list[float] = []
    
    for position, height in zip(positions, heights):
        for idx in range(line_count):
            new_positions.append(position + (idx - equivalent_protons / 2) * coupling_ppm)
            new_heights.append(height * binomial[idx])
            
    return new_positions, new_heights


def couplings_to_ppm(couplings: Any, frequency_mhz: float) -> list[float]:
    """Convert coupling constants from Hz to ppm.

    Parameters
    ----------
    couplings
        Coupling constants in raw source format.
    frequency_mhz
        Spectrometer frequency in MHz.

    Returns
    -------
    list[float]
        Couplings in ppm.
    """
    return [value / frequency_mhz for value in parse_couplings(couplings) if frequency_mhz > 0]


def get_coupling(couplings_ppm: list[float], idx: int) -> float:
    """Select a coupling value for a splitting level.

    Parameters
    ----------
    couplings_ppm
        Couplings in ppm.
    idx
        Splitting level index.

    Returns
    -------
    float
        Coupling value in ppm.
    """
    if not couplings_ppm:
        return DEFAULT_J_PPM
    if idx < len(couplings_ppm):
        return couplings_ppm[idx]
    return couplings_ppm[-1] * (0.6 ** (idx - len(couplings_ppm) + 1))


def multiplet_peaks(
    shift_center: float,
    multiplicity: str,
    couplings: Any,
    frequency_mhz: float,
    rng: np.random.Generator,
) -> tuple[list[float], list[float], float]:
    """Calculate proton multiplet subpeak positions and relative heights.

    Parameters
    ----------
    shift_center
        Peak center in ppm.
    multiplicity
        Multiplicity label such as ``s``, ``d``, or ``dd``.
    couplings
        Raw coupling constants in Hz.
    frequency_mhz
        Spectrometer frequency in MHz.
    rng
        NumPy random generator for unknown multiplets.

    Returns
    -------
    tuple[list[float], list[float], float]
        Subpeak positions, heights, and line-width multiplier.
    """
    couplings_ppm = couplings_to_ppm(couplings, frequency_mhz)
    line_width_multiplier = 1.0
    multiplicity_norm = normalize_multiplicity(multiplicity)
    
    if multiplicity_norm.startswith(BROAD_PREFIX):
        line_width_multiplier = BROAD_LINE_WIDTH_MULTIPLIER
        core = multiplicity_norm[len(BROAD_PREFIX) :]
        multiplicity_norm = core if core else "s"
        
    if multiplicity_norm == "m" or multiplicity_norm not in MULTIPLET_SPLITS:
        subpeak_count = int(rng.integers(7, 15))
        span = max(couplings_ppm[0] if couplings_ppm else 0.025, 0.015)
        positions = rng.uniform(shift_center - span, shift_center + span, subpeak_count)
        heights = rng.uniform(0.4, 1.0, subpeak_count)
        
        return positions.tolist(), heights.tolist(), line_width_multiplier * 2.0
    
    split_sequence = MULTIPLET_SPLITS[multiplicity_norm]
    
    if not split_sequence:
        return [shift_center], [1.0], line_width_multiplier
    
    positions = [shift_center]
    heights = [1.0]
    
    for idx, equivalent_protons in enumerate(split_sequence):
        positions, heights = apply_splitting(
            positions, heights, get_coupling(couplings_ppm, idx), equivalent_protons
        )
        
    return positions, heights, line_width_multiplier
