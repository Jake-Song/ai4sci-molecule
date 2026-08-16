"""Utilities for the AI for Science molecule exercises."""

from .week1 import (
    DESCRIPTOR_NAMES,
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    build_baselines,
    calculate_descriptors,
    load_esol,
    make_random_split,
    regression_metrics,
    smiles_to_graph,
)
from .week2 import (
    ATOM_FEATURE_CARDINALITIES,
    BOND_FEATURE_CARDINALITIES,
    GCNRegressor,
    GINRegressor,
    TrainingConfig,
    TrainingResult,
    build_gnn_models,
    predict_gnn,
    resolve_device,
    seed_everything,
    train_gnn,
)

__all__ = [
    "DESCRIPTOR_NAMES",
    "EDGE_FEATURE_NAMES",
    "NODE_FEATURE_NAMES",
    "build_baselines",
    "calculate_descriptors",
    "load_esol",
    "make_random_split",
    "regression_metrics",
    "smiles_to_graph",
    "ATOM_FEATURE_CARDINALITIES",
    "BOND_FEATURE_CARDINALITIES",
    "GCNRegressor",
    "GINRegressor",
    "TrainingConfig",
    "TrainingResult",
    "build_gnn_models",
    "predict_gnn",
    "resolve_device",
    "seed_everything",
    "train_gnn",
]
