"""Generate publication figures for the SpectraLM NMR LLM manuscript.

The figures are intentionally data-driven: training curves and benchmark metrics
are read from ``outputs/experiments/structure`` so the manuscript graphics track
the experiment artifacts rather than hand-copied numbers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/spectralm-mpl")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
EXP = ROOT / "outputs" / "experiments" / "structure"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green": "#5EAA62",
    "green_soft": "#DDF3DE",
    "red": "#B64342",
    "red_soft": "#F6CFCB",
    "gold": "#D99A20",
    "teal": "#42949E",
    "violet": "#8C5FBF",
    "neutral_light": "#E9E9E9",
    "neutral_mid": "#767676",
    "neutral_dark": "#333333",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_all(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=PALETTE["neutral_dark"],
    )


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    fc: str,
    ec: str = "#2B2B2B",
    fontsize: float = 7,
    weight: str = "normal",
) -> None:
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=0.9,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=PALETTE["neutral_dark"],
        fontweight=weight,
        linespacing=1.15,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.1,
            color=PALETTE["neutral_mid"],
        )
    )


def draw_workflow() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 4.25))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.02,
        0.96,
        "Constraint-aware text LLM pipeline for 1D NMR structure prediction",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
    )

    rounded_box(
        ax,
        (0.04, 0.68),
        (0.20, 0.16),
        "$^1$H peak table\nshift, mult., J, integral",
        PALETTE["green_soft"],
        fontsize=7,
    )
    rounded_box(
        ax,
        (0.04, 0.43),
        (0.20, 0.16),
        "$^{13}$C peak table\nchemical shifts",
        PALETTE["green_soft"],
        fontsize=7,
    )
    rounded_box(
        ax,
        (0.04, 0.18),
        (0.20, 0.16),
        "Optional formula\nuser-supplied at inference",
        "#FFF3CD",
        fontsize=7,
    )

    rounded_box(
        ax,
        (0.32, 0.48),
        (0.22, 0.22),
        "Qwen3-8B text LLM\nLoRA SFT\nresponse-only loss",
        "#D9E8F7",
        fontsize=7.2,
        weight="bold",
    )
    rounded_box(
        ax,
        (0.62, 0.63),
        (0.25, 0.14),
        'Direct JSON output\n{"smiles": "..."}',
        "#E8DAF5",
        fontsize=7,
    )
    rounded_box(
        ax,
        (0.62, 0.42),
        (0.25, 0.14),
        "Top-k candidate samples\nparse + deduplicate",
        "#E8DAF5",
        fontsize=7,
    )
    rounded_box(
        ax,
        (0.62, 0.19),
        (0.25, 0.14),
        "Chemistry constraints\nRDKit canonicalization\nformula/domain filters",
        "#FBE2DF",
        fontsize=6.7,
    )
    rounded_box(
        ax,
        (0.38, 0.12),
        (0.16, 0.14),
        "1D NMR rules\npre-rank / audit",
        "#F2F2F2",
        fontsize=7,
    )

    for y in (0.76, 0.51, 0.26):
        arrow(ax, (0.25, y), (0.32, 0.59))
    arrow(ax, (0.54, 0.61), (0.62, 0.70))
    arrow(ax, (0.54, 0.57), (0.62, 0.49))
    arrow(ax, (0.745, 0.42), (0.745, 0.33))
    arrow(ax, (0.54, 0.19), (0.62, 0.25))
    arrow(ax, (0.87, 0.26), (0.94, 0.26))

    ax.text(
        0.93,
        0.26,
        "metrics\nexact match\nformula accuracy\nTanimoto\nFG-F1\nbehavior",
        fontsize=6.4,
        va="center",
        ha="left",
        linespacing=1.2,
        color=PALETTE["neutral_dark"],
    )
    ax.text(
        0.32,
        0.40,
        "Training target masks prompt, formula, and peak-table tokens;\nonly assistant JSON is supervised.",
        fontsize=6.5,
        ha="left",
        va="top",
        color=PALETTE["neutral_mid"],
    )
    ax.text(
        0.62,
        0.86,
        "What is new: a controlled text-LLM benchmark that separates\nspectral text learning from hard chemical validity constraints.",
        fontsize=7,
        ha="left",
        va="top",
        color=PALETTE["blue_main"],
    )

    save_all(fig, "fig1_workflow")


def load_training_curves() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    curves = {}
    mapping = {
        "Stage 2 + formula": EXP / "stage2-formula-10k-seed3407" / "logs" / "training_log_live.json",
        "Stage 2 no formula": EXP
        / "stage2-no-formula-10k-seed3407"
        / "logs"
        / "training_log_live.json",
    }
    for label, path in mapping.items():
        data = load_json(path)
        eval_log = data.get("eval_log", [])
        curves[label] = (
            np.array([row["step"] for row in eval_log], dtype=float),
            np.array([row["eval_loss"] for row in eval_log], dtype=float),
        )
    return curves


def load_prediction_summaries() -> dict[str, dict]:
    names = {
        "Direct + formula": "direct-formula-10k.summary.json",
        "Direct no formula": "direct-no-formula-10k.summary.json",
        "Candidates + formula": "candidates-formula-10k.summary.json",
        "Candidates no formula": "candidates-no-formula-10k.summary.json",
    }
    return {label: load_json(EXP / "predictions" / fname) for label, fname in names.items()}


def draw_results() -> None:
    summaries = load_prediction_summaries()
    curves = load_training_curves()

    fig = plt.figure(figsize=(7.1, 5.45))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.34)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    for label, (steps, losses) in curves.items():
        color = PALETTE["blue_main"] if "formula" in label and "no" not in label else PALETTE["red"]
        ax1.plot(steps, losses, marker="o", markersize=3, linewidth=1.6, label=label, color=color)
    ax1.set_title("Validation loss", fontsize=8, pad=6)
    ax1.set_xlabel("Step", fontsize=7)
    ax1.set_ylabel("Loss", fontsize=7)
    ax1.tick_params(labelsize=6)
    ax1.legend(fontsize=6, frameon=False, loc="upper right")
    ax1.spines[["top", "right"]].set_visible(False)
    panel_label(ax1, "a")

    metrics = [
        ("Valid", "valid_smiles_rate"),
        ("Formula", "molecular_formula_accuracy"),
        ("Mean Tan.", "mean_tanimoto"),
        ("FG-F1", "functional_group_micro_f1"),
        ("Exact", "connectivity_exact_match"),
    ]
    x = np.arange(len(metrics))
    width = 0.36
    direct_formula = [summaries["Direct + formula"][k] for _, k in metrics]
    direct_no_formula = [summaries["Direct no formula"][k] for _, k in metrics]
    ax2.bar(x - width / 2, direct_formula, width, color=PALETTE["blue_main"], label="+ formula")
    ax2.bar(x + width / 2, direct_no_formula, width, color=PALETTE["red"], label="no formula")
    ax2.set_title("Direct generation", fontsize=8, pad=6)
    ax2.set_ylim(0, 1.04)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m[0] for m in metrics], rotation=25, ha="right", fontsize=6)
    ax2.tick_params(axis="y", labelsize=6)
    ax2.legend(fontsize=6, frameon=False, loc="upper right")
    ax2.spines[["top", "right"]].set_visible(False)
    panel_label(ax2, "b")

    cand_metrics = [
        ("Valid", "valid_smiles_rate"),
        ("Formula", "molecular_formula_accuracy"),
        ("Oracle@32", "candidate_oracle_connectivity_at_32"),
        ("FG-F1", "functional_group_micro_f1"),
        ("Exact", "connectivity_exact_match"),
    ]
    cand_formula = [summaries["Candidates + formula"][k] for _, k in cand_metrics]
    cand_no_formula = [summaries["Candidates no formula"][k] for _, k in cand_metrics]
    ax3.bar(x - width / 2, cand_formula, width, color=PALETTE["blue_main"], label="+ formula")
    ax3.bar(x + width / 2, cand_no_formula, width, color=PALETTE["red"], label="no formula")
    ax3.set_title("Candidate mode", fontsize=8, pad=6)
    ax3.set_ylim(0, 1.04)
    ax3.set_xticks(x)
    ax3.set_xticklabels([m[0] for m in cand_metrics], rotation=25, ha="right", fontsize=6)
    ax3.tick_params(axis="y", labelsize=6)
    ax3.legend(fontsize=6, frameon=False, loc="upper right")
    ax3.spines[["top", "right"]].set_visible(False)
    panel_label(ax3, "c")

    ax4.set_axis_off()
    direct = summaries["Direct + formula"]
    cand = summaries["Candidates + formula"]
    bullets = [
        ("valid SMILES", f"{direct['valid_smiles_rate'] * 100:.1f}%"),
        ("formula-correct direct outputs", f"{direct['molecular_formula_accuracy'] * 100:.1f}%"),
        ("candidate oracle@32", f"{cand['candidate_oracle_connectivity_at_32'] * 100:.1f}%"),
        ("formula-valid candidates / 32", f"{cand['mean_formula_valid_candidate_count']:.2f}"),
        ("exact connectivity", f"{direct['connectivity_exact_match'] * 100:.1f}%"),
    ]
    ax4.text(0.02, 0.98, "Primary diagnosis", fontsize=8, fontweight="bold", va="top")
    y = 0.78
    for label, value in bullets:
        ax4.text(0.02, y, value, fontsize=12.5, fontweight="bold", color=PALETTE["blue_main"], va="center")
        ax4.text(0.30, y, label, fontsize=7, color=PALETTE["neutral_dark"], va="center")
        y -= 0.145
    panel_label(ax4, "d")

    save_all(fig, "fig2_results")


def load_candidate_means() -> dict[str, dict[str, float]]:
    means = {}
    for label, fname in {
        "Formula": "candidates-formula-10k.jsonl",
        "No formula": "candidates-no-formula-10k.jsonl",
    }.items():
        path = EXP / "predictions" / fname
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        keys = [
            "raw_candidate_count",
            "unique_candidate_count",
            "domain_valid_candidate_count",
            "formula_valid_candidate_count",
            "candidate_oracle_connectivity",
            "formula_constraint_failed",
            "ranking_attempted",
            "ranking_failed",
        ]
        means[label] = {k: float(np.mean([row.get(k, 0.0) for row in rows])) for k in keys}
    return means


def draw_candidate_bottleneck() -> None:
    means = load_candidate_means()
    summary = load_prediction_summaries()["Candidates + formula"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.15), gridspec_kw={"width_ratios": [1.4, 1.0]})

    stages = ["Raw\nsamples", "Unique +\ndomain-valid", "Formula-valid", "Reference\npresent"]
    formula_vals = [
        means["Formula"]["raw_candidate_count"],
        means["Formula"]["domain_valid_candidate_count"],
        means["Formula"]["formula_valid_candidate_count"],
        means["Formula"]["candidate_oracle_connectivity"],
    ]
    no_formula_vals = [
        means["No formula"]["raw_candidate_count"],
        means["No formula"]["domain_valid_candidate_count"],
        means["No formula"]["domain_valid_candidate_count"],
        means["No formula"]["candidate_oracle_connectivity"],
    ]
    x = np.arange(len(stages))
    ax1.plot(x, formula_vals, marker="o", linewidth=2.0, color=PALETTE["blue_main"], label="+ formula filter")
    ax1.plot(x, no_formula_vals, marker="o", linewidth=1.6, color=PALETTE["red"], label="no formula filter")
    for xv, yv in zip(x, formula_vals):
        if yv < 0.1:
            label = f"{yv:.3f}"
        elif yv < 2:
            label = f"{yv:.2f}"
        else:
            label = f"{yv:.1f}"
        ax1.text(xv, yv + 1.1, label, fontsize=6, ha="center")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=6)
    ax1.set_ylabel("Mean candidates per test sample", fontsize=7)
    ax1.tick_params(axis="y", labelsize=6)
    ax1.set_ylim(-0.5, 35)
    ax1.set_title("Candidate-set attrition", fontsize=8, pad=6)
    ax1.legend(fontsize=6, frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)
    panel_label(ax1, "a")

    ax2.set_axis_off()
    rounded_box(
        ax2,
        (0.05, 0.65),
        (0.90, 0.25),
        f"{summary['formula_constraint_failure_rate'] * 100:.0f}%\nno formula-valid candidate",
        PALETTE["red_soft"],
        fontsize=8,
        weight="bold",
    )
    rounded_box(
        ax2,
        (0.05, 0.36),
        (0.90, 0.21),
        f"{summary['candidate_oracle_connectivity_at_32'] * 100:.1f}%\noracle@32 connectivity",
        "#FFF3CD",
        fontsize=8,
        weight="bold",
    )
    rounded_box(
        ax2,
        (0.05, 0.08),
        (0.90, 0.20),
        "Reranking cannot fix\nmissing candidates",
        PALETTE["green_soft"],
        fontsize=8,
        weight="bold",
    )
    ax2.text(
        0.05,
        0.98,
        "Bottleneck",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
        color=PALETTE["neutral_dark"],
    )
    panel_label(ax2, "b")

    save_all(fig, "fig3_candidate_bottleneck")


def main() -> None:
    # Fig. 1 is a manually prepared graphical overview stored as
    # paper/figures/fig1_workflow.png. Do not overwrite it here.
    draw_results()
    draw_candidate_bottleneck()


if __name__ == "__main__":
    main()
