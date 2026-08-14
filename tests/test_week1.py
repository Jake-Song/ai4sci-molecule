"""Network-independent tests for the week 1 public API."""

import numpy as np
import pytest
import torch

from ai4sci_molecule.week1 import (
    DESCRIPTOR_NAMES,
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    calculate_descriptors,
    make_random_split,
    regression_metrics,
    smiles_to_graph,
)


def test_ethanol_graph_has_expected_shape_and_bidirectional_edges():
    graph = smiles_to_graph("CCO", target=-0.3)

    assert graph.x.shape == (3, len(NODE_FEATURE_NAMES))
    assert graph.edge_index.shape == (2, 4)
    assert graph.edge_attr.shape == (4, len(EDGE_FEATURE_NAMES))
    assert set(map(tuple, graph.edge_index.t().tolist())) == {
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
    }
    assert graph.smiles == "CCO"
    torch.testing.assert_close(graph.y, torch.tensor([-0.3]))


def test_benzene_bonds_and_atoms_are_aromatic_and_in_rings():
    graph = smiles_to_graph("c1ccccc1")
    aromatic_atom = NODE_FEATURE_NAMES.index("is_aromatic")
    aromatic_bond = EDGE_FEATURE_NAMES.index("bond_AROMATIC")
    ring_bond = EDGE_FEATURE_NAMES.index("is_in_ring")

    assert graph.num_nodes == 6
    assert graph.num_edges == 12
    assert torch.all(graph.x[:, aromatic_atom] == 1)
    assert torch.all(graph.edge_attr[:, aromatic_bond] == 1)
    assert torch.all(graph.edge_attr[:, ring_bond] == 1)


def test_descriptor_values_are_chemically_reasonable():
    descriptors = calculate_descriptors(["CCO", "c1ccccc1"])

    assert tuple(descriptors.columns) == DESCRIPTOR_NAMES
    assert descriptors.loc[0, "MolWt"] == pytest.approx(46.069, abs=0.01)
    assert descriptors.loc[0, "MolLogP"] == pytest.approx(-0.0014, abs=0.01)
    assert descriptors.loc[0, "TPSA"] == pytest.approx(20.23, abs=0.01)
    assert descriptors.loc[0, "NumRotatableBonds"] == 0
    assert descriptors.loc[0, "AromaticAtomFraction"] == 0
    assert descriptors.loc[1, "AromaticAtomFraction"] == pytest.approx(1.0)


@pytest.mark.parametrize("invalid", ["not-a-smiles", "", None])
def test_invalid_smiles_raise_clear_value_error(invalid):
    with pytest.raises(ValueError, match="Invalid SMILES"):
        calculate_descriptors([invalid])

    with pytest.raises(ValueError, match="Invalid SMILES"):
        smiles_to_graph(invalid)


def test_split_is_deterministic_disjoint_and_covers_every_sample():
    split = make_random_split(1_128, seed=42)
    repeated = make_random_split(1_128, seed=42)

    assert {name: len(values) for name, values in split.items()} == {
        "train": 902,
        "validation": 113,
        "test": 113,
    }
    for name in split:
        np.testing.assert_array_equal(split[name], repeated[name])

    all_indices = np.concatenate(list(split.values()))
    assert len(np.unique(all_indices)) == 1_128
    np.testing.assert_array_equal(np.sort(all_indices), np.arange(1_128))


def test_regression_metrics_match_known_values():
    metrics = regression_metrics([0.0, 1.0, 2.0], [0.0, 2.0, 1.0])

    assert metrics["RMSE"] == pytest.approx(np.sqrt(2 / 3))
    assert metrics["MAE"] == pytest.approx(2 / 3)
    assert metrics["R2"] == pytest.approx(0.0)
