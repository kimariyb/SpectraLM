"""Render normalized 1H and 13C NMR samples as spectrum images.

Images are produced at a fixed resolution of ``WIDTH_PX`` × ``HEIGHT_PX``
(1280×720 at 100 DPI) via the Agg backend — no display required.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spectralm-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "spectralm-cache"))

import matplotlib
matplotlib.use("Agg")  # headless — must precede any other matplotlib import

import matplotlib.pyplot as _plt  # only for figure cleanup in figure_to_image
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib import ticker
from PIL import Image

from src.data.utils import parse_frequency_mhz
from src.io import write_json
from src.spectra.utils import add_noise, pseudo_voigt, multiplet_peaks


DPI = 100
WIDTH_PX = 1280
HEIGHT_PX = 720
SPECTRUM_POINTS = 16384


def compute_1h(
    sample: dict[str, Any],
    snr: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """Compute a continuous 1H NMR spectrum.

    Parameters
    ----------
    sample
        Normalized sample dictionary containing ``1H_NMR``.
    snr
        Signal-to-noise ratio.
    rng
        NumPy random generator.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]
        Chemical shift axis, intensity array, and integration annotations.
    """
    nmr = sample["1H_NMR"]
    frequency_mhz = parse_frequency_mhz(nmr.get("frequency"), default=400.0)
    x_axis = np.linspace(0.0, 12.0, SPECTRUM_POINTS)
    intensity = np.zeros_like(x_axis)
    line_width_base = 0.008
    eta = 0.55
    integral_data: list[tuple[float, float]] = []
    
    for entry in nmr.get("peaks", []):
        if isinstance(entry, dict):
            shift = float(entry["shift"])
            multiplicity = str(entry.get("multiplicity", "s"))
            couplings = entry.get("J", [])
            integration = float(entry.get("integration", 1.0))
        else:
            shift, multiplicity, couplings, integration = entry
            shift = float(shift)
            integration = float(integration)
        positions, heights, line_width_multiplier = multiplet_peaks(
            shift, multiplicity, couplings, frequency_mhz, rng
        )
        line_width = line_width_base * line_width_multiplier * float(rng.uniform(0.85, 1.15))
        height_sum = sum(heights) or 1.0
        for position, height in zip(positions, heights):
            intensity += (integration * height / height_sum) * pseudo_voigt(
                x_axis, position, line_width, eta
            )
        integral_data.append((shift, integration))
        
    intensity /= intensity.max() or 1.0
    intensity = add_noise(intensity, snr=snr, rng=rng)
    intensity = np.clip(intensity, -0.03, None)
    
    return x_axis, intensity, integral_data


def compute_13c(
    sample: dict[str, Any],
    snr: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a continuous 13C NMR spectrum.

    Parameters
    ----------
    sample
        Normalized sample dictionary containing ``13C_NMR``.
    snr
        Signal-to-noise ratio.
    rng
        NumPy random generator.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Chemical shift axis and intensity array.
    """
    x_axis = np.linspace(0.0, 220.0, SPECTRUM_POINTS)
    intensity = np.zeros_like(x_axis)
    line_width_base = 0.06
    eta = 0.60
    
    for entry in sample["13C_NMR"].get("peaks", []):
        shift = entry["shift"] if isinstance(entry, dict) else entry
        shifts = shift if isinstance(shift, list) else [shift]
        for value in shifts:
            line_width = line_width_base * float(rng.uniform(0.8, 1.2))
            intensity += pseudo_voigt(x_axis, float(value), line_width, eta)
            
    intensity /= intensity.max() or 1.0
    intensity = add_noise(intensity, snr=snr, rng=rng)
    intensity = np.clip(intensity, -0.03, None)
    
    return x_axis, intensity


def draw_1h(
    ax: Axes,
    x_axis: np.ndarray,
    intensity: np.ndarray,
    ppm_min: float = 0.0,
    ppm_max: float = 12.0,
    label: str = "",
) -> None:
    """Draw a 1H NMR spectrum on existing axes.

    Parameters
    ----------
    ax
        Matplotlib axes object.
    x_axis
        Chemical shift axis.
    intensity
        Spectrum intensity array.
    ppm_min
        Minimum chemical shift value.
    ppm_max
        Maximum chemical shift value.
    label
        Optional panel label.
    """
    ax.plot(x_axis, intensity, color="black", linewidth=1, zorder=3)
    set_spectra_axes(ax, ppm_min, ppm_max)
    if label:
        ax.text(
            0.01,
            0.95,
            label,
            transform=ax.transAxes,
            fontsize=18,
            va="top",
            color="black",
            fontweight="bold",
        )


def draw_13c(
    ax: Axes,
    x_axis: np.ndarray,
    intensity: np.ndarray,
    ppm_min: float = 0.0,
    ppm_max: float = 220.0,
    label: str = "",
) -> None:
    """Draw a 13C NMR spectrum on existing axes.

    Parameters
    ----------
    ax
        Matplotlib axes object.
    x_axis
        Chemical shift axis.
    intensity
        Spectrum intensity array.
    ppm_min
        Minimum chemical shift value.
    ppm_max
        Maximum chemical shift value.
    label
        Optional panel label.
    """
    ax.plot(x_axis, intensity, color="black", linewidth=1, zorder=3)
    set_spectra_axes(ax, ppm_min, ppm_max)
    if label:
        ax.text(
            0.01,
            0.95,
            label,
            transform=ax.transAxes,
            fontsize=18,
            va="top",
            color="black",
            fontweight="bold",
        )


def set_spectra_axes(ax: Axes, ppm_min: float, ppm_max: float) -> None:
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
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10 if ppm_max > 100 else 1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2 if ppm_max > 100 else 0.2))


def figure_to_image(fig: Figure) -> Image.Image:
    """Convert a Matplotlib figure to an RGB PIL image.

    Parameters
    ----------
    fig
        Matplotlib figure.

    Returns
    -------
    PIL.Image.Image
        RGB image.
    """
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buffer = np.asarray(canvas.buffer_rgba())
    image = Image.fromarray(buffer[..., :3]).convert("RGB")
    _plt.close(fig)

    return image


def hydrogen_to_spectra(
    sample: dict[str, Any],
    snr: float = 500.0,
    seed: int | None = None,
) -> Image.Image:
    """Render a standalone 1H NMR spectrum image.

    Parameters
    ----------
    sample
        Normalized sample dictionary.
    snr
        Signal-to-noise ratio.
    seed
        Optional seed for reproducible rendering.  When ``None`` (default)
        system entropy is used, producing slightly different line-width
        jitter on every call.

    Returns
    -------
    PIL.Image.Image
        RGB spectrum image at ``WIDTH_PX`` × ``HEIGHT_PX`` pixels.
    """
    rng = np.random.default_rng(seed)
    x_axis, intensity, _ = compute_1h(sample, snr, rng)
    fig = Figure(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)
    ax = fig.add_subplot(111)
    draw_1h(ax, x_axis, intensity)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.95)
    return figure_to_image(fig)


def carbon_to_spectra(
    sample: dict[str, Any],
    snr: float = 500.0,
    seed: int | None = None,
) -> Image.Image:
    """Render a standalone 13C NMR spectrum image.

    Parameters
    ----------
    sample
        Normalized sample dictionary.
    snr
        Signal-to-noise ratio.
    seed
        Optional seed for reproducible rendering.  When ``None`` (default)
        system entropy is used, producing slightly different line-width
        jitter on every call.

    Returns
    -------
    PIL.Image.Image
        RGB spectrum image at ``WIDTH_PX`` × ``HEIGHT_PX`` pixels.
    """
    rng = np.random.default_rng(seed)
    x_axis, intensity = compute_13c(sample, snr, rng)
    fig = Figure(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)
    ax = fig.add_subplot(111)
    draw_13c(ax, x_axis, intensity)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.95)
    return figure_to_image(fig)


def demo_sample() -> dict[str, Any]:
    """Return a small built-in NMR sample for rendering smoke tests.

    Returns
    -------
    dict[str, Any]
        Normalized sample dictionary.
    """
    return {
        "id": "demo",
        "smiles": "CCO",
        "canonical_smiles": "CCO",
        "meta": {"source": "demo"},
        "13C_NMR": {
            "frequency": "101 MHz",
            "solvent": "CDCl3",
            "peaks": [{"shift": 58.1}, {"shift": 18.2}],
        },
        "1H_NMR": {
            "frequency": "400 MHz",
            "solvent": "CDCl3",
            "peaks": [
                {"shift": 3.65, "multiplicity": "q", "J": [7.0], "integration": 2.0},
                {"shift": 1.18, "multiplicity": "t", "J": [7.0], "integration": 3.0},
                {"shift": 2.0, "multiplicity": "brs", "J": [], "integration": 1.0},
            ],
        },
    }


if __name__ == "__main__":
    output_dir = Path("img")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sample = demo_sample()
    hydrogen_to_spectra(sample).save(output_dir / "spectra_1H.png")
    carbon_to_spectra(sample).save(output_dir / "spectra_13C.png")

    write_json(output_dir / "render_demo_manifest.json", {"sample_id": sample["id"]})
    print(f"Wrote demo spectra to {output_dir}")
