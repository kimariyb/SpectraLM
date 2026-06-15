"""Dataset distribution visualization utilities."""

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
import pandas as pd
import seaborn as sns

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
    sns.histplot(
        values["c_shifts"],
        bins=100,
        binrange=(-20, 220),
        color="#1f77b4",
        edgecolor="white",
        alpha=0.7,
        ax=axes[0],
    )
    axes[0].set_title("13C NMR Chemical Shift Distribution", fontsize=16)
    axes[0].set_xlabel("Chemical Shift (ppm)", fontsize=14)
    axes[0].set_ylabel("Number of Peaks", fontsize=14)
    axes[0].set_xlim(-20, 220)
    sns.histplot(
        values["h_shifts"],
        bins=100,
        binrange=(-2, 12),
        color="#ff7f0e",
        edgecolor="white",
        alpha=0.7,
        ax=axes[1],
    )
    axes[1].set_title("1H NMR Chemical Shift Distribution", fontsize=16)
    axes[1].set_xlabel("Chemical Shift (ppm)", fontsize=14)
    axes[1].set_ylabel("Number of Peaks", fontsize=14)
    axes[1].set_xlim(-2, 12)
    sns.barplot(data=solvent_counts, y="Solvent", x="Count", color="#2ca02c", alpha=0.7, ax=axes[2])
    axes[2].set_title("Top 10 Solvent Distribution", fontsize=16)
    axes[2].set_xlabel("Number of Samples", fontsize=14)
    axes[2].set_ylabel("")
    sns.barplot(
        data=multiplicity_counts,
        y="Multiplicity",
        x="Count",
        color="#9467bd",
        alpha=0.7,
        ax=axes[3],
    )
    axes[3].set_title("1H NMR Multiplicity Distribution (Top 15)", fontsize=16)
    axes[3].set_xlabel("Number of Peaks", fontsize=14)
    axes[3].set_ylabel("")
    for idx, (count, multiplicity) in enumerate(
        zip(multiplicity_counts["Count"], multiplicity_counts["Multiplicity"])
    ):
        axes[3].text(count + multiplicity_counts["Count"].max() * 0.01, idx, f"{count:,}", va="center", fontsize=10)
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
    parser.add_argument("--dataset", default="dataset/NMRexp_spectra_dataset.pkl")
    parser.add_argument("--output", default="img/spectra_distribution.png")
    return parser


def main() -> None:
    """Run dataset visualization from the command line."""
    args = build_arg_parser().parse_args()
    values = collect_distribution_values(load_pickle_list(args.dataset))
    plot_spectra_distribution(values, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
