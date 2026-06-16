"""Line-shape functions and plotting utilities for NMR spectra."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spectralm-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "spectralm-cache"))
    
import numpy as np
from matplotlib import ticker


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


def set_spectra_axes(ax, ppm_min: float, ppm_max: float) -> None:
    """Apply standard NMR axis styling.

    Parameters
    ----------
    ax
        Matplotlib axes object.
    ppm_min
        Minimum chemical shift value.
    ppm_max
        Maximum chemical shift value.
    """
    ax.set_xlim(ppm_max, ppm_min)
    ax.set_xlabel("Chemical Shift (ppm)", fontsize=16, labelpad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", direction="out", length=4, labelsize=14)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20 if ppm_max > 100 else 1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5 if ppm_max > 100 else 0.2))
