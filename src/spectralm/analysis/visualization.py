"""Dataset and selected subset visualization utilities."""

from __future__ import annotations

import argparse
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spectralm-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "spectralm-cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from spectralm.config import add_config_argument, load_config
from spectralm.io import load_pickle_list


def collect_distribution_values(samples: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Collect chemical shift, solvent, and multiplicity values.

    Parameters
    ----------
    samples
        Normalized SpectraLM samples.

    Returns
    -------
    dict[str, list[Any]]
        Distribution values for plotting and reporting.
    """
    c_shifts = []
    h_shifts = []
    solvents_c = []
    solvents_h = []
    multiplicities = []
    for sample in samples:
        for peak in sample["13C_NMR"]["peaks"]:
            try:
                shift = peak["shift"]
                if isinstance(shift, list):
                    c_shifts.extend(float(value) for value in shift)
                else:
                    c_shifts.append(float(shift))
            except (ValueError, TypeError):
                pass
        solvents_c.append(sample["13C_NMR"].get("solvent"))
        for peak in sample["1H_NMR"]["peaks"]:
            try:
                h_shifts.append(float(peak["shift"]))
            except (ValueError, TypeError):
                pass
            multiplicity = peak.get("multiplicity")
            if multiplicity and isinstance(multiplicity, str) and multiplicity.strip():
                multiplicities.append(multiplicity.strip().lower())
        solvents_h.append(sample["1H_NMR"].get("solvent"))
    return {
        "c_shifts": c_shifts,
        "h_shifts": h_shifts,
        "solvents": solvents_c + solvents_h,
        "multiplicities": multiplicities,
    }


def configure_plot_style() -> None:
    """Configure Matplotlib and Seaborn style for publication-ready plots."""
    sns.set_theme(style="ticks", context="notebook", font_scale=1.1)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["mathtext.default"] = "regular"


def plot_spectra_distribution(values: dict[str, list[Any]], output_path: str | Path) -> None:
    """Plot NMR shift, solvent, and multiplicity distributions.

    Parameters
    ----------
    values
        Distribution values from :func:`collect_distribution_values`.
    output_path
        Output image path.
    """
    configure_plot_style()
    solvent_counts = pd.Series(values["solvents"]).value_counts().head(10).reset_index()
    solvent_counts.columns = ["Solvent", "Count"]
    multiplicity_counts = pd.DataFrame(
        Counter(values["multiplicities"]).most_common(15),
        columns=["Multiplicity", "Count"],
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    sns.histplot(values["c_shifts"], bins=100, binrange=(-20, 220), color="#1f77b4", edgecolor="white", alpha=0.7, ax=axes[0])
    axes[0].set_title("13C NMR Chemical Shift Distribution", fontsize=16)
    axes[0].set_xlabel("Chemical Shift (ppm)", fontsize=14)
    axes[0].set_ylabel("Number of Peaks", fontsize=14)
    axes[0].set_xlim(-20, 220)
    sns.histplot(values["h_shifts"], bins=100, binrange=(-2, 12), color="#ff7f0e", edgecolor="white", alpha=0.7, ax=axes[1])
    axes[1].set_title("1H NMR Chemical Shift Distribution", fontsize=16)
    axes[1].set_xlabel("Chemical Shift (ppm)", fontsize=14)
    axes[1].set_ylabel("Number of Peaks", fontsize=14)
    axes[1].set_xlim(-2, 12)
    sns.barplot(data=solvent_counts, y="Solvent", x="Count", color="#2ca02c", alpha=0.7, ax=axes[2])
    axes[2].set_title("Top 10 Solvent Distribution", fontsize=16)
    axes[2].set_xlabel("Number of Samples", fontsize=14)
    axes[2].set_ylabel("")
    sns.barplot(data=multiplicity_counts, y="Multiplicity", x="Count", color="#9467bd", alpha=0.7, ax=axes[3])
    axes[3].set_title("1H NMR Multiplicity Distribution (Top 15)", fontsize=16)
    axes[3].set_xlabel("Number of Peaks", fontsize=14)
    axes[3].set_ylabel("")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_fingerprint_matrix(path: str | Path) -> np.ndarray:
    """Load a Morgan fingerprint matrix from an ``.npz`` file.

    Parameters
    ----------
    path
        Fingerprint ``.npz`` path.

    Returns
    -------
    numpy.ndarray
        Fingerprint matrix.

    Raises
    ------
    KeyError
        If the file does not contain ``fingerprints``.
    """
    payload = np.load(path)
    if "fingerprints" not in payload:
        raise KeyError(f"Expected 'fingerprints' in {path}")
    return payload["fingerprints"].astype(np.float32)


def pca_projection(
    matrix: np.ndarray,
    n_components: int = 2,
    max_fit_points: int | None = 50000,
    chunk_size: int = 200000,
    seed: int = 3407,
) -> np.ndarray:
    """Project a matrix with PCA using covariance eigendecomposition.

    Parameters
    ----------
    matrix
        Input feature matrix.
    n_components
        Number of principal components.
    max_fit_points
        Optional maximum rows used to fit PCA components.
    chunk_size
        Number of rows projected at a time.
    seed
        Random seed for fitting-row subsampling.

    Returns
    -------
    numpy.ndarray
        PCA coordinates.

    Raises
    ------
    ValueError
        If fewer than two rows are provided.
    """
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("PCA projection requires a matrix with at least two rows")
    numeric = matrix.astype(np.float64, copy=False)
    if max_fit_points is not None and matrix.shape[0] > max_fit_points:
        rng = np.random.default_rng(seed)
        fit_indices = np.sort(rng.choice(matrix.shape[0], size=max_fit_points, replace=False))
        fit_matrix = numeric[fit_indices]
    else:
        fit_matrix = numeric
    mean = fit_matrix.mean(axis=0, keepdims=True)
    centered_fit = fit_matrix - mean
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        covariance = centered_fit.T @ centered_fit / max(fit_matrix.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, order[:n_components]]
    projected_parts = []
    for start in range(0, numeric.shape[0], chunk_size):
        chunk = numeric[start : start + chunk_size] - mean
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            projected_parts.append(chunk @ components)
    projected = np.vstack(projected_parts)
    if projected.shape[1] < n_components:
        padding = np.zeros((matrix.shape[0], n_components - projected.shape[1]), dtype=np.float32)
        projected = np.hstack([projected, padding])
    if not np.isfinite(projected).all():
        raise ValueError("PCA projection produced non-finite values")
    return projected.astype(np.float32)


def selected_lookup(selected_csv: str | Path) -> dict[str, str]:
    """Build a selected molecule lookup keyed by canonical SMILES.

    Parameters
    ----------
    selected_csv
        Selected samples CSV path.

    Returns
    -------
    dict[str, str]
        Mapping from canonical SMILES to split name.
    """
    frame = pd.read_csv(selected_csv)
    lookup = {}
    for _, row in frame.iterrows():
        smiles = str(row.get("canonical_smiles", "")).strip()
        if smiles:
            lookup[smiles] = str(row.get("split", "selected"))
    return lookup


def build_selection_projection(
    fingerprints: str | Path,
    index_csv: str | Path,
    selected_csv: str | Path,
    max_pca_fit_points: int | None = 50000,
) -> pd.DataFrame:
    """Build PCA projection data with selected molecules marked.

    Parameters
    ----------
    fingerprints
        Morgan fingerprint ``.npz`` path for all valid molecules.
    index_csv
        Fingerprint index CSV path.
    selected_csv
        Selected samples CSV path.
    max_pca_fit_points
        Maximum rows used to fit PCA components.

    Returns
    -------
    pandas.DataFrame
        Projection rows with ``pc1``, ``pc2``, ``selected``, and ``split`` columns.
    """
    matrix = load_fingerprint_matrix(fingerprints)
    index = pd.read_csv(index_csv)
    if len(index) != matrix.shape[0]:
        raise ValueError("Fingerprint matrix row count does not match index CSV")
    coords = pca_projection(matrix, max_fit_points=max_pca_fit_points)
    lookup = selected_lookup(selected_csv)
    frame = index.copy()
    frame["pc1"] = coords[:, 0]
    frame["pc2"] = coords[:, 1]
    frame["split"] = frame["canonical_smiles"].map(lookup).fillna("unselected")
    frame["selected"] = frame["split"] != "unselected"
    return frame


def plot_selection_projection(
    frame: pd.DataFrame,
    output_path: str | Path,
    max_background_points: int | None = 200000,
    seed: int = 3407,
) -> None:
    """Plot all molecules as background and selected molecules as highlighted points.

    Parameters
    ----------
    frame
        PCA projection dataframe from :func:`build_selection_projection`.
    output_path
        Output image path.
    max_background_points
        Optional maximum number of unselected background points to draw.
    seed
        Random seed for background subsampling.
    """
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    background = frame[~frame["selected"]]
    if max_background_points is not None and len(background) > max_background_points:
        background = background.sample(n=max_background_points, random_state=seed)
    selected = frame[frame["selected"]]
    ax.scatter(
        background["pc1"],
        background["pc2"],
        s=5,
        c="#c9c9c9",
        alpha=0.18,
        linewidths=0,
        label=f"Unselected ({len(background):,})",
    )
    palette = {"train": "#d62728", "test": "#1f77b4", "val": "#2ca02c", "selected": "#9467bd"}
    for split_name, split_frame in selected.groupby("split"):
        ax.scatter(
            split_frame["pc1"],
            split_frame["pc2"],
            s=28,
            c=palette.get(split_name, "#9467bd"),
            alpha=0.9,
            edgecolors="white",
            linewidths=0.35,
            label=f"{split_name} ({len(split_frame):,})",
        )
    ax.set_title("Selected Molecules Highlighted in Morgan Fingerprint PCA Space")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False, loc="best")
    sns.despine(ax=ax)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the visualization CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Plot SpectraLM dataset distributions.")
    add_config_argument(parser)
    parser.add_argument("--dataset", default="dataset/NMRexp_spectra_dataset.pkl")
    parser.add_argument("--output", default="img/spectra_distribution.png")
    parser.add_argument("--selected-csv", default=None, help="Selected samples CSV for Butina subset visualization.")
    parser.add_argument("--fingerprints", default=None, help="Morgan fingerprint NPZ for all valid molecules.")
    parser.add_argument("--index", default=None, help="Fingerprint index CSV for all valid molecules.")
    parser.add_argument("--max-background-points", type=int, default=None, help="Maximum unselected background points to draw.")
    parser.add_argument("--max-pca-fit-points", type=int, default=None, help="Maximum rows used to fit PCA components.")
    return parser


def main() -> None:
    """Run dataset visualization from the command line."""
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    selected_csv = args.selected_csv or config.get("selected_csv")
    output = args.output if args.output != "img/spectra_distribution.png" else config.get("output", args.output)
    dataset = args.dataset if args.dataset != "dataset/NMRexp_spectra_dataset.pkl" else config.get("dataset", args.dataset)
    fingerprints = args.fingerprints or config.get("fingerprints")
    index = args.index or config.get("index")
    max_background_points = args.max_background_points
    if max_background_points is None:
        max_background_points = config.get("max_background_points", 200000)
    max_pca_fit_points = args.max_pca_fit_points
    if max_pca_fit_points is None:
        max_pca_fit_points = config.get("max_pca_fit_points", 50000)
    if selected_csv:
        if not fingerprints or not index:
            raise ValueError("Selection PCA requires fingerprints, index, and selected_csv")
        frame = build_selection_projection(
            fingerprints,
            index,
            selected_csv,
            max_pca_fit_points=max_pca_fit_points,
        )
        plot_selection_projection(frame, output, max_background_points=max_background_points)
    else:
        values = collect_distribution_values(load_pickle_list(dataset))
        plot_spectra_distribution(values, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
