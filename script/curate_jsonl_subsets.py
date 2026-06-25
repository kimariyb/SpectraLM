"""Curate training subsets from a paired JSONL dataset manifest.

This script does not copy ``samples.jsonl``.  It reads ``manifest.csv``,
applies lightweight QC filters, creates scaffold-balanced ranked id lists, and
writes named subset files such as:

- ``subsets/clean_10k_train_ids.txt``
- ``subsets/clean_10k_val_ids.txt``
- ``subsets/clean_10k_test_ids.txt``

Training configs can point ``train_split_name`` and ``eval_split_name`` at
these names while reusing the same paired JSONL mother dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Allow running from project root without PYTHONPATH.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.molecules import inspect_dataset_molecule


DEFAULT_SUBSET_SIZES = [10_000]


def _read_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read manifest rows from CSV."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_ids(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write sample ids, one per line."""
    ids = [str(row["id"]) for row in rows if row.get("id")]
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write selected manifest rows for inspection."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _to_int(value: Any, default: int = 0) -> int:
    """Parse integer-like manifest values."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _solvent_known(value: Any) -> bool:
    """Return whether a solvent value carries useful information."""
    text = str(value or "").strip()
    return bool(text and text.lower() not in {"nan", "none", "not_known", "unknown"})


def _row_passes_filters(
    row: dict[str, Any],
    *,
    min_heavy_atoms: int,
    max_heavy_atoms: int,
    min_h_peaks: int,
    max_h_peaks: int,
    min_c_peaks: int,
    max_c_peaks: int,
    solvent_policy: str,
) -> tuple[bool, str]:
    """Apply manifest-level QC filters to one row."""
    if row.get("qc_status") != "pass":
        return False, "manifest_qc_fail"

    inspection = inspect_dataset_molecule(row.get("canonical_smiles"))
    if not inspection.accepted:
        return False, inspection.violations[0]
    if (
        inspection.isotope_label_count
        or _to_int(row.get("isotope_label_count")) > 0
    ):
        return False, "isotope_labeled_structure"

    heavy = _to_int(row.get("heavy_atom_count"))
    h_peaks = _to_int(row.get("h_peak_count"))
    c_peaks = _to_int(row.get("c_peak_count"))

    if heavy < min_heavy_atoms:
        return False, "too_few_heavy_atoms"
    if heavy > max_heavy_atoms:
        return False, "too_many_heavy_atoms"
    if h_peaks < min_h_peaks:
        return False, "too_few_1h_peaks"
    if h_peaks > max_h_peaks:
        return False, "too_many_1h_peaks"
    if c_peaks < min_c_peaks:
        return False, "too_few_13c_peaks"
    if c_peaks > max_c_peaks:
        return False, "too_many_13c_peaks"

    h_solvent = str(row.get("h_solvent", "")).strip()
    c_solvent = str(row.get("c_solvent", "")).strip()
    if solvent_policy == "known" and not (
        _solvent_known(h_solvent) and _solvent_known(c_solvent)
    ):
        return False, "unknown_solvent"
    if solvent_policy == "matched" and h_solvent != c_solvent:
        return False, "solvent_mismatch"
    if solvent_policy == "cdcl3" and not (
        h_solvent == "CDCl3" and c_solvent == "CDCl3"
    ):
        return False, "not_cdcl3"

    return True, ""


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    min_heavy_atoms: int = 2,
    max_heavy_atoms: int = 60,
    min_h_peaks: int = 1,
    max_h_peaks: int = 80,
    min_c_peaks: int = 1,
    max_c_peaks: int = 120,
    solvent_policy: str = "any",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Filter manifest rows and return kept rows plus rejection counts."""
    kept: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()

    for row in rows:
        ok, reason = _row_passes_filters(
            row,
            min_heavy_atoms=min_heavy_atoms,
            max_heavy_atoms=max_heavy_atoms,
            min_h_peaks=min_h_peaks,
            max_h_peaks=max_h_peaks,
            min_c_peaks=min_c_peaks,
            max_c_peaks=max_c_peaks,
            solvent_policy=solvent_policy,
        )
        if ok:
            kept.append(row)
        else:
            rejected[reason] += 1

    return kept, dict(rejected)


def scaffold_balanced_order(
    rows: list[dict[str, Any]],
    *,
    seed: int = 3407,
) -> list[dict[str, Any]]:
    """Return rows in a deterministic scaffold-balanced order.

    The ordering provides a scaffold-diverse deterministic prefix.
    """
    rng = random.Random(seed)
    by_scaffold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scaffold = str(row.get("murcko_scaffold") or row.get("canonical_smiles") or row["id"])
        by_scaffold[scaffold].append(row)

    for group in by_scaffold.values():
        group.sort(
            key=lambda item: (
                _to_int(item.get("heavy_atom_count")),
                _to_int(item.get("h_peak_count")) + _to_int(item.get("c_peak_count")),
                str(item.get("id", "")),
            )
        )
        rng.shuffle(group)

    scaffolds = list(by_scaffold)
    rng.shuffle(scaffolds)

    ordered: list[dict[str, Any]] = []
    active = scaffolds
    while active:
        next_active: list[str] = []
        for scaffold in active:
            group = by_scaffold[scaffold]
            if group:
                ordered.append(group.pop())
            if group:
                next_active.append(scaffold)
        active = next_active
    return ordered


def assign_grouped_random_splits(
    rows: list[dict[str, Any]],
    *,
    seed: int = 3407,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, list[dict[str, Any]]]:
    """Assign complete canonical-structure groups to deterministic random splits.

    Parameters
    ----------
    rows
        Manifest rows to split.
    seed
        Random seed used for deterministic group shuffling.
    val_fraction
        Target validation fraction.
    test_fraction
        Target test fraction.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Rows grouped under ``train``, ``val``, and ``test`` keys.
    """
    if not 0.0 <= float(val_fraction) < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if not 0.0 <= float(test_fraction) < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if float(val_fraction) + float(test_fraction) >= 1.0:
        raise ValueError("validation and test fractions must leave training rows")

    by_structure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get("canonical_smiles") or row.get("id") or "")
        by_structure[key].append(row)

    groups = list(by_structure.values())
    for group in groups:
        group.sort(key=lambda item: str(item.get("id", "")))
    rng = random.Random(seed)
    rng.shuffle(groups)

    total = len(rows)
    targets = {
        "val": int(round(total * float(val_fraction))),
        "test": int(round(total * float(test_fraction))),
    }
    assigned: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for group in groups:
        split = min(
            ("val", "test"),
            key=lambda name: (
                len(assigned[name]) / max(targets[name], 1)
                if targets[name]
                else 1.0,
                len(assigned[name]),
            ),
        )
        if targets[split] == 0 or len(assigned[split]) >= targets[split]:
            split = "train"
        assigned[split].extend(group)

    return assigned


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize selected manifest rows."""
    split_counts = Counter(row.get("split", "") for row in rows)
    solvents = Counter(
        (
            str(row.get("h_solvent", "")).strip(),
            str(row.get("c_solvent", "")).strip(),
        )
        for row in rows
    )
    return {
        "samples": len(rows),
        "split_counts": dict(split_counts),
        "unique_scaffolds": len({row.get("murcko_scaffold", "") for row in rows}),
        "heavy_atom_min": min((_to_int(row.get("heavy_atom_count")) for row in rows), default=0),
        "heavy_atom_max": max((_to_int(row.get("heavy_atom_count")) for row in rows), default=0),
        "top_solvent_pairs": [
            {"h_solvent": pair[0], "c_solvent": pair[1], "count": count}
            for pair, count in solvents.most_common(10)
        ],
    }


def build_subsets(
    manifest_rows: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    subset_sizes: list[int],
    val_size: int = 5000,
    val_fraction: float | None = None,
    test_fraction: float = 0.1,
    prefix: str = "clean",
    seed: int = 3407,
    min_heavy_atoms: int = 2,
    max_heavy_atoms: int = 60,
    min_h_peaks: int = 1,
    max_h_peaks: int = 80,
    min_c_peaks: int = 1,
    max_c_peaks: int = 120,
    solvent_policy: str = "any",
) -> dict[str, Any]:
    """Build named subset id files from manifest rows."""
    if val_fraction is not None and not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if not 0.0 < float(test_fraction) < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    filtered, rejected = filter_rows(
        manifest_rows,
        min_heavy_atoms=min_heavy_atoms,
        max_heavy_atoms=max_heavy_atoms,
        min_h_peaks=min_h_peaks,
        max_h_peaks=max_h_peaks,
        min_c_peaks=min_c_peaks,
        max_c_peaks=max_c_peaks,
        solvent_policy=solvent_policy,
    )

    rows_by_split = {
        split: [row for row in filtered if row.get("split") == split]
        for split in ["train", "val", "test"]
    }
    ranked = {
        split: scaffold_balanced_order(rows, seed=seed + idx)
        for idx, (split, rows) in enumerate(rows_by_split.items())
    }

    summary: dict[str, Any] = {
        "filter": {
            "min_heavy_atoms": min_heavy_atoms,
            "max_heavy_atoms": max_heavy_atoms,
            "min_h_peaks": min_h_peaks,
            "max_h_peaks": max_h_peaks,
            "min_c_peaks": min_c_peaks,
            "max_c_peaks": max_c_peaks,
            "solvent_policy": solvent_policy,
        },
        "input_rows": len(manifest_rows),
        "filtered_rows": len(filtered),
        "rejected_counts": rejected,
        "available_by_split": {split: len(rows) for split, rows in ranked.items()},
        "subsets": {},
    }

    for requested_size in sorted(set(int(size) for size in subset_sizes)):
        requested_test_size = int(
            round(requested_size * float(test_fraction))
        )
        if val_fraction is None:
            requested_val_size = int(val_size)
            requested_train_size = (
                requested_size - requested_val_size - requested_test_size
            )
        else:
            requested_val_size = int(
                round(requested_size * float(val_fraction))
            )
            requested_train_size = (
                requested_size - requested_val_size - requested_test_size
            )
        if requested_train_size <= 0:
            raise ValueError(
                "subset size must exceed validation and test allocations"
            )
        selected_train = ranked["train"][: min(
            requested_train_size,
            len(ranked["train"]),
        )]
        selected_val = ranked["val"][: min(
            requested_val_size,
            len(ranked["val"]),
        )]
        selected_test = ranked["test"][: min(
            requested_test_size,
            len(ranked["test"]),
        )]
        name = f"{prefix}_{requested_size // 1000}k" if requested_size >= 1000 else f"{prefix}_{requested_size}"

        selected = {
            "train": selected_train,
            "val": selected_val,
            "test": selected_test,
        }
        for split, rows in selected.items():
            _write_ids(out_path / f"{name}_{split}_ids.txt", rows)
            _write_csv(out_path / f"{name}_{split}_manifest.csv", rows)

        summary["subsets"][name] = {
            "requested_total_size": requested_size,
            "requested_train_size": requested_train_size,
            "requested_val_size": requested_val_size,
            "requested_test_size": requested_test_size,
            **{
            split: summarize_rows(rows)
            for split, rows in selected.items()
            },
        }

    (out_path / "curation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        help="Paired JSONL dataset directory containing manifest.csv.",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--subset-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SUBSET_SIZES,
    )
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--prefix", default="clean")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--min-heavy-atoms", type=int, default=2)
    parser.add_argument("--max-heavy-atoms", type=int, default=60)
    parser.add_argument("--min-h-peaks", type=int, default=1)
    parser.add_argument("--max-h-peaks", type=int, default=80)
    parser.add_argument("--min-c-peaks", type=int, default=1)
    parser.add_argument("--max-c-peaks", type=int, default=120)
    parser.add_argument(
        "--solvent-policy",
        choices=["any", "known", "matched", "cdcl3"],
        default="any",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir) if args.out_dir else dataset_dir / "subsets"
    rows = _read_manifest(dataset_dir / "manifest.csv")
    summary = build_subsets(
        rows,
        out_dir,
        subset_sizes=args.subset_sizes,
        val_size=args.val_size,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        prefix=args.prefix,
        seed=args.seed,
        min_heavy_atoms=args.min_heavy_atoms,
        max_heavy_atoms=args.max_heavy_atoms,
        min_h_peaks=args.min_h_peaks,
        max_h_peaks=args.max_h_peaks,
        min_c_peaks=args.min_c_peaks,
        max_c_peaks=args.max_c_peaks,
        solvent_policy=args.solvent_policy,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote curated subset id files to {out_dir}")


if __name__ == "__main__":
    main()
