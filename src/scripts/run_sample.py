import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ModuleNotFoundError:
    Chem = None
    Descriptors = None

try:
    from ..utils.io_utils import load_pickle_list, write_json, write_pickle, write_rows_csv
    from ..utils.nmr_utils import peak_count, sample_smiles
except ImportError:
    from utils.io_utils import load_pickle_list, write_json, write_pickle, write_rows_csv
    from utils.nmr_utils import peak_count, sample_smiles


FUNCTIONAL_GROUP_SMARTS = {
    "aromatic": "a",
    "alkene": "C=C",
    "alkyne": "C#C",
    "alcohol": "[OX2H][#6]",
    "phenol": "c[OX2H]",
    "ether": "[OD2]([#6])[#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "ester": "[CX3](=O)[OX2][#6]",
    "amide": "[NX3][CX3](=O)",
    "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "nitrile": "C#N",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "halide": "[F,Cl,Br,I]",
    "boron": "[B]",
    "silicon": "[Si]",
    "phosphorus": "[P]",
    "sulfur": "[S]",
}


def require_rdkit() -> None:
    if Chem is None or Descriptors is None:
        raise RuntimeError(
            "RDKit is required for representative sampling. "
            "Install it with `conda install -c conda-forge rdkit`."
        )


def mol_from_smiles(smiles: str | None):
    require_rdkit()
    if not smiles:
        return None
    return Chem.MolFromSmiles(str(smiles))


def functional_group_labels(smiles: str | None) -> list[str]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ["invalid"]

    labels = []
    for label, smarts in FUNCTIONAL_GROUP_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            labels.append(label)
    return labels or ["none_detected"]


def bin_value(value: int | float, low: int | float, high: int | float) -> str:
    if value <= low:
        return "low"
    if value <= high:
        return "mid"
    return "high"


def assign_complexity_bin(row: dict[str, Any]) -> str:
    mol = mol_from_smiles(row.get("canonical_smiles") or sample_smiles(row))
    heavy_atoms = mol.GetNumHeavyAtoms() if mol is not None else 0
    ring_count = mol.GetRingInfo().NumRings() if mol is not None else 0
    h_peaks = peak_count(row, "1H_NMR")
    c_peaks = peak_count(row, "13C_NMR")

    return "|".join(
        [
            f"heavy_atoms:{bin_value(heavy_atoms, 20, 40)}",
            f"rings:{bin_value(ring_count, 0, 2)}",
            f"h_peaks:{bin_value(h_peaks, 6, 14)}",
            f"c_peaks:{bin_value(c_peaks, 10, 25)}",
        ]
    )


def enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    smiles = out.get("canonical_smiles") or sample_smiles(out)
    out["functional_groups"] = functional_group_labels(smiles)
    out["complexity_bin"] = assign_complexity_bin(out)

    mol = mol_from_smiles(smiles)
    out["heavy_atoms"] = mol.GetNumHeavyAtoms() if mol is not None else 0
    out["mol_weight"] = Descriptors.MolWt(mol) if mol is not None else 0
    out["h_peak_count"] = peak_count(out, "1H_NMR")
    out["c_peak_count"] = peak_count(out, "13C_NMR")
    return out


def coverage_gain(row: dict[str, Any], covered: Counter) -> int:
    keys = [f"complexity:{row['complexity_bin']}"]
    keys.extend(f"fg:{label}" for label in row["functional_groups"])
    keys.append(f"scaffold:{row['murcko_scaffold']}")
    return sum(1 for key in keys if covered[key] == 0)


def representative_sample(
    rows: list[dict[str, Any]],
    split_name: str,
    target_size: int,
    max_per_scaffold: int = 1,
    max_heavy_atoms: int | None = None,
    max_selfies_length: int | None = None,
    seed: int = 3407,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    candidates = []
    for row in rows:
        if row.get("split") != split_name:
            continue
        enriched = enrich_row(row)
        if max_heavy_atoms is not None and enriched["heavy_atoms"] > max_heavy_atoms:
            continue
        if max_selfies_length is not None and len(enriched.get("selfies", "")) > max_selfies_length:
            continue
        candidates.append(enriched)
    rng.shuffle(candidates)

    selected = []
    covered = Counter()
    scaffold_counts = Counter()
    molecule_seen = set()
    by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_bin[row["complexity_bin"]].append(row)

    bin_names = list(by_bin)
    rng.shuffle(bin_names)

    while by_bin and len(selected) < target_size:
        progressed = False

        for bin_name in list(bin_names):
            bucket = by_bin.get(bin_name, [])
            if not bucket:
                by_bin.pop(bin_name, None)
                if bin_name in bin_names:
                    bin_names.remove(bin_name)
                continue

            best_idx = None
            best_score = None
            for idx, row in enumerate(bucket):
                scaffold = row["murcko_scaffold"]
                smiles = row.get("canonical_smiles")
                if scaffold_counts[scaffold] >= max_per_scaffold or smiles in molecule_seen:
                    continue

                score = (
                    coverage_gain(row, covered),
                    len(row["functional_groups"]),
                    row["h_peak_count"] + row["c_peak_count"],
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                by_bin.pop(bin_name, None)
                if bin_name in bin_names:
                    bin_names.remove(bin_name)
                continue

            row = bucket.pop(best_idx)
            selected.append(row)
            scaffold_counts[row["murcko_scaffold"]] += 1
            molecule_seen.add(row.get("canonical_smiles"))
            covered[f"complexity:{row['complexity_bin']}"] += 1
            covered[f"scaffold:{row['murcko_scaffold']}"] += 1
            for label in row["functional_groups"]:
                covered[f"fg:{label}"] += 1
            progressed = True

            if len(selected) >= target_size:
                break

        if not progressed:
            break

    return selected


def load_split_csv(path: Path) -> dict[str, dict[str, str]]:
    by_id = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_id[row["id"]] = row
    return by_id


def attach_split_metadata(
    samples: list[dict[str, Any]],
    split_rows: dict[str, dict[str, str]],
    keep_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        sample_id = sample.get("id")
        if keep_ids is not None and sample_id not in keep_ids:
            continue
        split_row = split_rows.get(sample_id)
        if not split_row:
            continue
        row = dict(sample)
        row.update(
            {
                "split": split_row["split"],
                "canonical_smiles": split_row["canonical_smiles"],
                "murcko_scaffold": split_row["murcko_scaffold"],
                "molecular_formula": split_row["molecular_formula"],
            }
        )
        rows.append(row)
    return rows


def preselect_split_ids(
    split_rows: dict[str, dict[str, str]],
    split_name: str,
    pool_size: int,
    max_per_scaffold: int,
    seed: int,
) -> set[str]:
    rng = random.Random(seed)
    rows = [row for row in split_rows.values() if row.get("split") == split_name]
    rng.shuffle(rows)

    selected = []
    scaffold_counts = Counter()
    molecule_seen = set()
    covered_fg = set()

    for row in rows:
        smiles = row.get("canonical_smiles")
        scaffold = row.get("murcko_scaffold")
        if not smiles or not scaffold:
            continue
        if scaffold_counts[scaffold] >= max_per_scaffold or smiles in molecule_seen:
            continue
        fg = functional_group_labels(smiles)
        if set(fg).issubset(covered_fg):
            continue
        selected.append(row)
        scaffold_counts[scaffold] += 1
        molecule_seen.add(smiles)
        covered_fg.update(fg)
        if len(selected) >= pool_size:
            return {item["id"] for item in selected}

    for row in rows:
        if len(selected) >= pool_size:
            break
        smiles = row.get("canonical_smiles")
        scaffold = row.get("murcko_scaffold")
        if not smiles or not scaffold:
            continue
        if scaffold_counts[scaffold] >= max_per_scaffold or smiles in molecule_seen:
            continue
        selected.append(row)
        scaffold_counts[scaffold] += 1
        molecule_seen.add(smiles)

    return {row["id"] for row in selected}


def sample_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fg_counts = Counter()
    complexity_counts = Counter()
    scaffold_counts = Counter()
    for row in rows:
        fg_counts.update(row["functional_groups"])
        complexity_counts[row["complexity_bin"]] += 1
        scaffold_counts[row["murcko_scaffold"]] += 1

    return {
        "samples": len(rows),
        "unique_molecules": len({row.get("canonical_smiles") for row in rows}),
        "unique_scaffolds": len(scaffold_counts),
        "max_per_scaffold": max(scaffold_counts.values()) if scaffold_counts else 0,
        "functional_groups": fg_counts.most_common(),
        "complexity_bins": complexity_counts.most_common(),
        "avg_1h_peak_count": sum(row["h_peak_count"] for row in rows) / len(rows) if rows else 0,
        "avg_13c_peak_count": sum(row["c_peak_count"] for row in rows) / len(rows) if rows else 0,
        "avg_heavy_atoms": sum(row["heavy_atoms"] for row in rows) / len(rows) if rows else 0,
        "avg_mol_weight": sum(row["mol_weight"] for row in rows) / len(rows) if rows else 0,
    }


SAMPLE_CSV_FIELDS = [
    "split",
    "id",
    "canonical_smiles",
    "murcko_scaffold",
    "molecular_formula",
    "functional_groups",
    "complexity_bin",
    "heavy_atoms",
    "mol_weight",
    "h_peak_count",
    "c_peak_count",
]


def selected_sample_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "split": row["split"],
            "id": row.get("id", ""),
            "canonical_smiles": row.get("canonical_smiles", ""),
            "murcko_scaffold": row.get("murcko_scaffold", ""),
            "molecular_formula": row.get("molecular_formula", ""),
            "functional_groups": ";".join(row["functional_groups"]),
            "complexity_bin": row["complexity_bin"],
            "heavy_atoms": row["heavy_atoms"],
            "mol_weight": f"{row['mol_weight']:.4f}",
            "h_peak_count": row["h_peak_count"],
            "c_peak_count": row["c_peak_count"],
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small representative SpectraLM subset.")
    parser.add_argument("--dataset", default="dataset/NMRexp_spectra_dataset.pkl")
    parser.add_argument("--split-csv", default="dataset/splits/scaffold_split.csv")
    parser.add_argument("--out-dir", default="dataset/subsets/spectralm_500_100")
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--max-per-scaffold", type=int, default=1)
    parser.add_argument("--max-heavy-atoms", type=int, default=None)
    parser.add_argument("--max-selfies-length", type=int, default=None)
    parser.add_argument("--pool-multiplier", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    split_rows = load_split_csv(Path(args.split_csv))
    train_pool_size = args.train_size * args.pool_multiplier
    test_pool_size = args.test_size * args.pool_multiplier
    print(f"Preselecting {train_pool_size:,} train candidate IDs...", file=sys.stderr, flush=True)
    train_ids = preselect_split_ids(
        split_rows, "train", train_pool_size, args.max_per_scaffold, args.seed
    )
    print(f"Preselecting {test_pool_size:,} test candidate IDs...", file=sys.stderr, flush=True)
    test_ids = preselect_split_ids(
        split_rows, "test", test_pool_size, args.max_per_scaffold, args.seed + 1
    )

    keep_ids = train_ids | test_ids
    print(f"Loading dataset and attaching {len(keep_ids):,} candidate samples...", file=sys.stderr, flush=True)
    samples = load_pickle_list(Path(args.dataset))
    rows = attach_split_metadata(samples, split_rows, keep_ids=keep_ids)
    print(f"Attached {len(rows):,} candidate samples.", file=sys.stderr, flush=True)

    train = representative_sample(
        rows,
        split_name="train",
        target_size=args.train_size,
        max_per_scaffold=args.max_per_scaffold,
        max_heavy_atoms=args.max_heavy_atoms,
        max_selfies_length=args.max_selfies_length,
        seed=args.seed,
    )
    test = representative_sample(
        rows,
        split_name="test",
        target_size=args.test_size,
        max_per_scaffold=args.max_per_scaffold,
        max_heavy_atoms=args.max_heavy_atoms,
        max_selfies_length=args.max_selfies_length,
        seed=args.seed + 1,
    )

    out_dir = Path(args.out_dir)
    write_pickle(out_dir / "train.pkl", train)
    write_pickle(out_dir / "test.pkl", test)
    write_rows_csv(out_dir / "selected_samples.csv", selected_sample_csv_rows(train + test), SAMPLE_CSV_FIELDS)

    report = {
        "train": sample_report(train),
        "test": sample_report(test),
        "scaffold_overlap_train_test": len(
            {row["murcko_scaffold"] for row in train}
            & {row["murcko_scaffold"] for row in test}
        ),
    }
    write_json(out_dir / "sample_report.json", report)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
