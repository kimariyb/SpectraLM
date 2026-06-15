"""Tests for scaffold splitting and representative sampling."""

from spectralm.data.sampling import representative_sample
from spectralm.data.splitting import scaffold_split


def test_scaffold_split_has_no_scaffold_overlap(ethanol_sample) -> None:
    """Scaffold split should keep scaffold groups in only one split."""
    rows = []
    for idx, scaffold in enumerate(["acyclic:CCO", "c1ccccc1", "C1CCCCC1"]):
        row = dict(ethanol_sample)
        row["id"] = f"sample-{idx}"
        row["canonical_smiles"] = ["CCO", "c1ccccc1", "C1CCCCC1"][idx]
        row["murcko_scaffold"] = scaffold
        rows.append(row)
    split = scaffold_split(rows, ratios=(1 / 3, 1 / 3, 1 / 3), seed=1)
    seen = {}
    for split_name, split_rows in split.items():
        for row in split_rows:
            scaffold = row["murcko_scaffold"]
            assert scaffold not in seen
            seen[scaffold] = split_name


def test_representative_sample_respects_target_and_scaffold_limit(ethanol_sample) -> None:
    """Representative sampling should respect target size and scaffold caps."""
    rows = []
    for idx, smiles in enumerate(["CCO", "CCN", "CCCl"]):
        row = dict(ethanol_sample)
        row["id"] = f"sample-{idx}"
        row["split"] = "train"
        row["canonical_smiles"] = smiles
        row["murcko_scaffold"] = f"acyclic:{smiles}"
        rows.append(row)
    selected = representative_sample(rows, "train", target_size=2, max_per_scaffold=1, seed=1)
    assert len(selected) == 2
    assert len({row["murcko_scaffold"] for row in selected}) == 2

