"""Integration checks for the downloaded PyG ESOL dataset."""

import numpy as np
import pytest

from ai4sci_molecule.week1 import calculate_descriptors, load_esol, make_random_split


@pytest.mark.integration
def test_esol_dataset_descriptors_targets_and_split():
    dataset = load_esol("data/MoleculeNet")
    smiles = [graph.smiles for graph in dataset]
    targets = np.array([graph.y.item() for graph in dataset])
    descriptors = calculate_descriptors(smiles)
    split = make_random_split(len(dataset), seed=42)

    assert len(dataset) == 1_128
    assert not descriptors.isna().any().any()
    assert not np.isnan(targets).any()
    assert [len(split[name]) for name in ("train", "validation", "test")] == [
        902,
        113,
        113,
    ]
