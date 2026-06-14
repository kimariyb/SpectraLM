import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .utils.io_utils import load_pickle_list, write_json, write_rows_csv
except ImportError:
    from utils.io_utils import load_pickle_list, write_json, write_rows_csv

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ModuleNotFoundError:
    Chem = None
    rdMolDescriptors = None
    MurckoScaffold = None


def require_rdkit() -> None:
    if Chem is None or rdMolDescriptors is None or MurckoScaffold is None:
        raise RuntimeError(
            "RDKit is required for scaffold splitting and molecule quality checks. "
            "Install it with `conda install -c conda-forge rdkit` in your project environment."
        )


def canonical_smiles(smiles: str | None) -> str | None:
    require_rdkit()
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def molecule_formula(smiles: str | None) -> str | None:
    require_rdkit()
    mol = Chem.MolFromSmiles(str(smiles)) if smiles else None
    if mol is None:
        return None
    return rdMolDescriptors.CalcMolFormula(mol)


def murcko_scaffold(smiles: str | None) -> str | None:
    require_rdkit()
    canon = canonical_smiles(smiles)
    if canon is None:
        return None

    mol = Chem.MolFromSmiles(canon)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or f"acyclic:{canon}"


def molecule_descriptors(smiles: str | None) -> dict[str, str | None]:
    canon = canonical_smiles(smiles)
    if canon is None:
        return {
            "canonical_smiles": None,
            "murcko_scaffold": None,
            "molecular_formula": None,
        }

    return {
        "canonical_smiles": canon,
        "murcko_scaffold": murcko_scaffold(canon),
        "molecular_formula": molecule_formula(canon),
    }


def sample_smiles(sample: dict[str, Any]) -> str | None:
    return (
        sample.get("canonical_smiles")
        or sample.get("canonical_SMILES")
        or sample.get("SMILES")
        or sample.get("smiles")
    )


def peak_count(sample: dict[str, Any], nucleus: str) -> int:
    nmr = sample.get(nucleus) or {}
    peaks = nmr.get("peaks", nmr.get("data", []))
    return len(peaks or [])


def is_complete_nmr_sample(sample: dict[str, Any]) -> bool:
    return peak_count(sample, "1H_NMR") > 0 and peak_count(sample, "13C_NMR") > 0


def analyze_samples(
    samples: list[dict[str, Any]],
    progress_every: int = 10000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache: dict[str, dict[str, str | None]] = {}
    valid_rows = []
    canonical_values = []
    scaffold_values = []
    h_counts = []
    c_counts = []
    missing_1h = 0
    missing_13c = 0

    total = len(samples)
    for idx, sample in enumerate(samples, 1):
        raw_smiles = sample_smiles(sample)
        cache_key = str(raw_smiles or "")
        if cache_key not in cache:
            cache[cache_key] = molecule_descriptors(raw_smiles)
        desc = cache[cache_key]

        h_count = peak_count(sample, "1H_NMR")
        c_count = peak_count(sample, "13C_NMR")
        h_counts.append(h_count)
        c_counts.append(c_count)
        missing_1h += h_count == 0
        missing_13c += c_count == 0

        canon = desc["canonical_smiles"]
        if canon is not None:
            canonical_values.append(canon)
            scaffold_values.append(desc["murcko_scaffold"])

        if canon is not None and h_count > 0 and c_count > 0:
            row = dict(sample)
            row.update(desc)
            valid_rows.append(row)

        if progress_every and (idx % progress_every == 0 or idx == total):
            print(
                f"Processed {idx:,}/{total:,} samples; valid complete rows: {len(valid_rows):,}",
                file=sys.stderr,
                flush=True,
            )

    report = {
        "total_samples": total,
        "valid_molecules": len(canonical_values),
        "invalid_molecules": total - len(canonical_values),
        "complete_1h_13c_samples": sum(1 for h, c in zip(h_counts, c_counts) if h > 0 and c > 0),
        "missing_1h": missing_1h,
        "missing_13c": missing_13c,
        "unique_canonical_smiles": len(set(canonical_values)),
        "duplicate_canonical_smiles": len(canonical_values) - len(set(canonical_values)),
        "unique_scaffolds": len(set(scaffold_values)),
        "top_scaffolds": Counter(scaffold_values).most_common(20),
        "avg_1h_peak_count": sum(h_counts) / len(h_counts) if h_counts else 0,
        "avg_13c_peak_count": sum(c_counts) / len(c_counts) if c_counts else 0,
    }
    return valid_rows, report


def valid_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows, _ = analyze_samples(samples, progress_every=0)
    return rows


def scaffold_split(
    samples: list[dict[str, Any]],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 3407,
) -> dict[str, list[dict[str, Any]]]:
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must contain train/val/test values that sum to 1.0")

    rows = samples if samples and "murcko_scaffold" in samples[0] else valid_samples(samples)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in rows:
        groups[sample["murcko_scaffold"]].append(sample)

    rng = random.Random(seed)
    grouped_rows = list(groups.values())
    rng.shuffle(grouped_rows)
    grouped_rows.sort(key=len, reverse=True)

    total = sum(len(group) for group in grouped_rows)
    targets = {
        "train": total * ratios[0],
        "val": total * ratios[1],
        "test": total * ratios[2],
    }
    split = {"train": [], "val": [], "test": []}

    for group in grouped_rows:
        name = min(split, key=lambda part: len(split[part]) / (targets[part] or 1))
        split[name].extend(group)

    return split


def build_quality_report(samples: list[dict[str, Any]]) -> dict[str, Any]:
    _, report = analyze_samples(samples, progress_every=0)
    return report


SPLIT_CSV_FIELDS = ["split", "id", "canonical_smiles", "murcko_scaffold", "molecular_formula"]


def split_csv_rows(split: dict[str, list[dict[str, Any]]]) -> list[dict[str, str]]:
    rows = []
    for name, split_rows in split.items():
        for row in split_rows:
            rows.append(
                {
                    "split": name,
                    "id": row.get("id", ""),
                    "canonical_smiles": row.get("canonical_smiles", ""),
                    "murcko_scaffold": row.get("murcko_scaffold", ""),
                    "molecular_formula": row.get("molecular_formula", ""),
                }
            )
    return rows


def split_summary(split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        name: {
            "samples": len(rows),
            "unique_scaffolds": len({row["murcko_scaffold"] for row in rows}),
            "unique_molecules": len({row["canonical_smiles"] for row in rows}),
        }
        for name, rows in split.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scaffold split and NMR data quality report.")
    parser.add_argument("--input", default="src/data/NMRexp_spectra_dataset.pkl")
    parser.add_argument("--out-dir", default="src/data/splits")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1))
    args = parser.parse_args()

    samples = load_pickle_list(Path(args.input))
    valid_rows, quality = analyze_samples(samples)
    split = scaffold_split(valid_rows, ratios=tuple(args.ratios), seed=args.seed)

    out_dir = Path(args.out_dir)
    quality["split_summary"] = split_summary(split)

    write_json(out_dir / "quality_report.json", quality)
    write_rows_csv(out_dir / "scaffold_split.csv", split_csv_rows(split), SPLIT_CSV_FIELDS)

    print(json.dumps(quality["split_summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {out_dir / 'quality_report.json'}")
    print(f"Wrote {out_dir / 'scaffold_split.csv'}")


if __name__ == "__main__":
    main()
