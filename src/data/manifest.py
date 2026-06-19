"""QC metadata and scaffold-disjoint split construction."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.molecules import (
    canonicalize_smiles,
    heavy_atom_count,
    molecule_formula,
    murcko_scaffold,
)
from src.data.utils import peak_count


MANIFEST_FIELDS = [
    "id",
    "split",
    "qc_status",
    "qc_reason",
    "canonical_smiles",
    "molecular_formula",
    "murcko_scaffold",
    "heavy_atom_count",
    "h_peak_count",
    "c_peak_count",
    "h_solvent",
    "c_solvent",
    "h_frequency",
    "c_frequency",
]


def sample_manifest_row(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert one normalized sample into a manifest row."""
    canonical = canonicalize_smiles(
        sample.get("canonical_smiles") or sample.get("smiles")
    )
    h_peaks = peak_count(sample, "1H_NMR")
    c_peaks = peak_count(sample, "13C_NMR")
    formula = molecule_formula(canonical)
    scaffold = murcko_scaffold(canonical)

    reasons: list[str] = []
    if canonical is None:
        reasons.append("invalid_smiles")
    if formula is None:
        reasons.append("missing_formula")
    if scaffold is None:
        reasons.append("missing_scaffold")
    if h_peaks <= 0:
        reasons.append("missing_1h_peaks")
    if c_peaks <= 0:
        reasons.append("missing_13c_peaks")

    h_nmr = sample.get("1H_NMR", {})
    c_nmr = sample.get("13C_NMR", {})
    return {
        "id": sample.get("id", ""),
        "split": "",
        "qc_status": "pass" if not reasons else "fail",
        "qc_reason": ";".join(reasons),
        "canonical_smiles": canonical or "",
        "molecular_formula": formula or "",
        "murcko_scaffold": scaffold or "",
        "heavy_atom_count": heavy_atom_count(canonical),
        "h_peak_count": h_peaks,
        "c_peak_count": c_peaks,
        "h_solvent": h_nmr.get("solvent", ""),
        "c_solvent": c_nmr.get("solvent", ""),
        "h_frequency": h_nmr.get("frequency", ""),
        "c_frequency": c_nmr.get("frequency", ""),
    }


def assign_scaffold_splits(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 3407,
) -> list[dict[str, Any]]:
    """Assign train/validation/test splits with scaffold exclusivity."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    train_ratio /= total_ratio
    val_ratio /= total_ratio

    pass_rows = [row for row in rows if row["qc_status"] == "pass"]
    train_target = int(len(pass_rows) * train_ratio)
    val_target = int(len(pass_rows) * val_ratio)

    by_scaffold: dict[str, list[dict[str, Any]]] = {}
    for row in pass_rows:
        by_scaffold.setdefault(row["murcko_scaffold"], []).append(row)

    rng = random.Random(seed)
    decorated = [(rng.random(), scaffold) for scaffold in by_scaffold]
    scaffolds = [
        scaffold
        for _, scaffold in sorted(
            decorated,
            key=lambda item: (-len(by_scaffold[item[1]]), item[0]),
        )
    ]

    split_counts: Counter[str] = Counter()
    for scaffold in scaffolds:
        group = by_scaffold[scaffold]
        if split_counts["train"] < train_target:
            split = "train"
        elif split_counts["val"] < val_target:
            split = "val"
        else:
            split = "test"
        for row in group:
            row["split"] = split
        split_counts[split] += len(group)

    for row in rows:
        if row["qc_status"] != "pass":
            row["split"] = "excluded"
    return rows


def manifest_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize QC, split sizes, and scaffold overlap."""
    split_counts = Counter(row["split"] for row in rows)
    qc_counts = Counter(row["qc_status"] for row in rows)
    qc_reasons = Counter(
        reason
        for row in rows
        for reason in str(row["qc_reason"]).split(";")
        if reason
    )
    scaffolds_by_split: dict[str, set[str]] = {}
    for row in rows:
        if row["split"] in {"train", "val", "test"}:
            scaffolds_by_split.setdefault(row["split"], set()).add(
                row["murcko_scaffold"]
            )

    overlaps = {}
    for left, right in [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]:
        overlaps[f"{left}_{right}"] = len(
            scaffolds_by_split.get(left, set())
            & scaffolds_by_split.get(right, set())
        )

    return {
        "samples": len(rows),
        "qc_counts": dict(qc_counts),
        "qc_reasons": dict(qc_reasons),
        "split_counts": dict(split_counts),
        "scaffold_counts_by_split": {
            split: len(scaffolds)
            for split, scaffolds in sorted(scaffolds_by_split.items())
        },
        "scaffold_overlap_counts": overlaps,
    }


def build_manifest(
    samples: list[dict[str, Any]],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 3407,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build manifest rows and their aggregate summary."""
    rows = [sample_manifest_row(sample) for sample in samples]
    rows = assign_scaffold_splits(
        rows,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    return rows, manifest_summary(rows)


def write_split_files(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Write one sample-ID file per benchmark split."""
    for split in ["train", "val", "test", "excluded"]:
        ids = [row["id"] for row in rows if row["split"] == split]
        path = out_dir / f"{split}_ids.txt"
        path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
