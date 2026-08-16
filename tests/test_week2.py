"""Network-independent tests for the week 2 GNN API."""

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils.smiles import from_smiles

from ai4sci_molecule.week2 import (
    ATOM_FEATURE_CARDINALITIES,
    BOND_FEATURE_CARDINALITIES,
    CategoricalFeatureEncoder,
    GCNRegressor,
    GINRegressor,
    TrainingConfig,
    build_gnn_models,
    predict_gnn,
    train_gnn,
)


def make_tiny_dataset():
    smiles_and_targets = [
        ("C", 0.5),
        ("CC", 0.1),
        ("CCC", -0.2),
        ("CCCC", -0.7),
        ("CO", 0.3),
        ("CCO", -0.3),
        ("CCCO", -0.6),
        ("CN", 0.2),
        ("CCN", -0.1),
        ("c1ccccc1", -1.5),
    ]
    dataset = []
    for smiles, target in smiles_and_targets:
        graph = from_smiles(smiles)
        graph.y = torch.tensor([target], dtype=torch.float32)
        dataset.append(graph)
    return dataset


def test_pyg_feature_cardinalities_match_esol_encoding():
    graph = from_smiles("CCO")

    assert graph.x.shape[1] == len(ATOM_FEATURE_CARDINALITIES) == 9
    assert graph.edge_attr.shape[1] == len(BOND_FEATURE_CARDINALITIES) == 3
    for column, cardinality in enumerate(ATOM_FEATURE_CARDINALITIES):
        assert int(graph.x[:, column].max()) < cardinality
    for column, cardinality in enumerate(BOND_FEATURE_CARDINALITIES):
        assert int(graph.edge_attr[:, column].max()) < cardinality


@pytest.mark.parametrize("model_type", [GCNRegressor, GINRegressor])
def test_graph_regressors_return_one_value_per_molecule(model_type):
    batch = next(iter(DataLoader(make_tiny_dataset()[:3], batch_size=3)))
    model = model_type(hidden_channels=16, num_layers=2, dropout=0)

    prediction = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)

    assert prediction.shape == (3,)
    assert torch.isfinite(prediction).all()
    prediction.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_feature_encoder_rejects_week1_float_features():
    encoder = CategoricalFeatureEncoder(ATOM_FEATURE_CARDINALITIES, 8)

    with pytest.raises(TypeError, match="integer dtype"):
        encoder(torch.zeros((2, len(ATOM_FEATURE_CARDINALITIES))))


def test_build_gnn_models_is_reproducible():
    first = build_gnn_models(hidden_channels=8, num_layers=1, seed=7)
    second = build_gnn_models(hidden_channels=8, num_layers=1, seed=7)

    for name in first:
        for first_parameter, second_parameter in zip(
            first[name].parameters(), second[name].parameters(), strict=True
        ):
            torch.testing.assert_close(first_parameter, second_parameter)


def test_train_and_predict_gnn_on_tiny_dataset():
    dataset = make_tiny_dataset()
    split = {
        "train": np.arange(0, 6),
        "validation": np.arange(6, 8),
        "test": np.arange(8, 10),
    }
    config = TrainingConfig(
        batch_size=3,
        learning_rate=5e-3,
        max_epochs=4,
        patience=2,
        seed=3,
    )

    result = train_gnn(
        GCNRegressor(hidden_channels=8, num_layers=1, dropout=0),
        dataset,
        split,
        config,
        device="cpu",
    )
    predictions = predict_gnn(
        result, dataset, split["test"], batch_size=2, device="cpu"
    )

    assert 1 <= result.best_epoch <= len(result.history) <= config.max_epochs
    assert set(result.best_validation_metrics) == {"RMSE", "MAE", "R2"}
    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()
