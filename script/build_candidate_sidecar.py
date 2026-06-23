"""Build formula-matched hard candidate sets for NMR ranking supervision."""

from __future__ import annotations
import os
import sys
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

# Allow running from project root without PYTHONPATH.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.molecules import canonicalize_smiles, molecule_formula


_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def _fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid canonical SMILES in candidate pool: {smiles}")
    return _MORGAN_GENERATOR.GetFingerprint(mol)


def _stable_pool(
    target_id: str,
    candidates: list[dict[str, str]],
    *,
    seed: int,
    max_pool_size: int,
) -> list[dict[str, str]]:
    if len(candidates) <= max_pool_size:
        return candidates
    return sorted(
        candidates,
        key=lambda row: hashlib.sha256(
            f"{seed}:{target_id}:{row['id']}".encode("utf-8")
        ).digest(),
    )[:max_pool_size]


def build_candidate_sidecar(
    dataset_dir: str | Path,
    split: str,
    output_path: str | Path,
    *,
    candidates_per_sample: int = 8,
    max_pool_size: int = 512,
    seed: int = 3407,
) -> dict[str, int]:
    """Build same-formula candidate sets ordered by hard-negative similarity.

    Parameters
    ----------
    dataset_dir
        Dataset containing ``samples.jsonl`` and split ID files.
    split
        Split name such as ``clean_50k_train``.
    output_path
        Destination JSONL sidecar.
    candidates_per_sample
        Maximum total candidates including the target.
    max_pool_size
        Deterministic cap before similarity scoring.
    seed
        Candidate-position randomization seed.

    Returns
    -------
    dict[str, int]
        Sidecar construction counts.
    """
    if candidates_per_sample < 2:
        raise ValueError("candidates_per_sample must be at least 2.")
    if max_pool_size < candidates_per_sample - 1:
        raise ValueError("max_pool_size must cover requested negatives.")

    base = Path(dataset_dir)
    split_key = {"validation": "val"}.get(split, split)
    ids_path = base / f"{split_key}_ids.txt"
    if not ids_path.exists():
        ids_path = base / "subsets" / f"{split_key}_ids.txt"
    if not ids_path.exists():
        raise FileNotFoundError(f"Split ID file not found: {ids_path}")
    selected_ids = {
        line.strip()
        for line in ids_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    samples: list[dict[str, str]] = []
    with (base / "samples.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            sample_id = str(row.get("id", ""))
            if sample_id not in selected_ids:
                continue
            canonical = canonicalize_smiles(
                row.get("canonical_smiles") or row.get("smiles")
            )
            if canonical is None:
                continue
            formula = str(row.get("molecular_formula") or molecule_formula(canonical))
            samples.append({"id": sample_id, "smiles": canonical, "formula": formula})

    by_formula: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in samples:
        by_formula[row["formula"]].append(row)
    fingerprints = {row["smiles"]: _fingerprint(row["smiles"]) for row in samples}

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate_sets = 0
    omitted = 0
    with output.open("w", encoding="utf-8") as handle:
        for target in samples:
            unique: dict[str, dict[str, str]] = {}
            for row in by_formula[target["formula"]]:
                if row["smiles"] != target["smiles"]:
                    unique.setdefault(row["smiles"], row)
            negatives = _stable_pool(
                target["id"],
                list(unique.values()),
                seed=seed,
                max_pool_size=max_pool_size,
            )
            if not negatives:
                omitted += 1
                continue
            target_fp = fingerprints[target["smiles"]]
            scored = sorted(
                (
                    DataStructs.TanimotoSimilarity(
                        target_fp,
                        fingerprints[row["smiles"]],
                    ),
                    row["smiles"],
                )
                for row in negatives
            )
            scored.reverse()
            selected = scored[: candidates_per_sample - 1]
            displayed = [target["smiles"], *[smiles for _, smiles in selected]]
            shuffle_seed = int.from_bytes(
                hashlib.sha256(f"{seed}:{target['id']}".encode("utf-8")).digest()[:8],
                byteorder="big",
            )
            random.Random(shuffle_seed).shuffle(displayed)
            record = {
                "id": target["id"],
                "molecular_formula": target["formula"],
                "target": target["smiles"],
                "candidates": displayed,
                "negative_tanimoto": [round(score, 6) for score, _ in selected],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            candidate_sets += 1

    return {
        "input_samples": len(samples),
        "candidate_sets": candidate_sets,
        "omitted_without_negatives": omitted,
    }


def main() -> None:
    """Build a candidate-ranking sidecar from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir")
    parser.add_argument("--split", default="clean_50k_train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidates-per-sample", type=int, default=8)
    parser.add_argument("--max-pool-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    report = build_candidate_sidecar(
        args.dataset_dir,
        args.split,
        args.output,
        candidates_per_sample=args.candidates_per_sample,
        max_pool_size=args.max_pool_size,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
