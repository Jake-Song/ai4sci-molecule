"""Reusable data and baseline helpers for week 1 of the ESOL project."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.datasets import MoleculeNet


_HYBRIDIZATIONS = (
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
)

NODE_FEATURE_NAMES = (
    "atomic_number",
    "degree",
    "formal_charge",
    "total_hydrogens",
    "is_aromatic",
    "hybridization_SP",
    "hybridization_SP2",
    "hybridization_SP3",
    "hybridization_SP3D",
    "hybridization_SP3D2",
    "hybridization_OTHER",
)

EDGE_FEATURE_NAMES = (
    "bond_SINGLE",
    "bond_DOUBLE",
    "bond_TRIPLE",
    "bond_AROMATIC",
    "is_conjugated",
    "is_in_ring",
)

DESCRIPTOR_NAMES = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumRotatableBonds",
    "AromaticAtomFraction",
)


def load_esol(root: str | Path = "data/MoleculeNet") -> MoleculeNet:
    """Download (when necessary) and load PyG's ESOL MoleculeNet dataset."""

    return MoleculeNet(root=str(root), name="ESOL")


def _parse_smiles(smiles: str, *, position: int | None = None) -> Chem.Mol:
    if not isinstance(smiles, str) or not smiles.strip():
        location = "" if position is None else f" at position {position}"
        raise ValueError(f"Invalid SMILES{location}: {smiles!r}")

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        location = "" if position is None else f" at position {position}"
        raise ValueError(f"Invalid SMILES{location}: {smiles!r}")
    return molecule


def _atom_features(atom: Chem.Atom) -> list[float]:
    hybridization = atom.GetHybridization()
    one_hot = [float(hybridization == value) for value in _HYBRIDIZATIONS]
    one_hot.append(float(hybridization not in _HYBRIDIZATIONS))
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        float(atom.GetIsAromatic()),
        *one_hot,
    ]


def _bond_features(bond: Chem.Bond) -> list[float]:
    bond_type = bond.GetBondType()
    return [
        float(bond_type == Chem.BondType.SINGLE),
        float(bond_type == Chem.BondType.DOUBLE),
        float(bond_type == Chem.BondType.TRIPLE),
        float(bond_type == Chem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def smiles_to_graph(smiles: str, target: float | None = None) -> Data:
    """Convert a SMILES string to a directed PyG molecular graph.

    Each RDKit bond is stored twice, once in each direction. Node and edge
    feature columns are documented by ``NODE_FEATURE_NAMES`` and
    ``EDGE_FEATURE_NAMES``.
    """

    molecule = _parse_smiles(smiles)
    x = torch.tensor(
        [_atom_features(atom) for atom in molecule.GetAtoms()],
        dtype=torch.float32,
    )

    directed_edges: list[tuple[int, int]] = []
    edge_features: list[list[float]] = []
    for bond in molecule.GetBonds():
        source = bond.GetBeginAtomIdx()
        destination = bond.GetEndAtomIdx()
        features = _bond_features(bond)
        directed_edges.extend(((source, destination), (destination, source)))
        edge_features.extend((features, features.copy()))

    if directed_edges:
        edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(EDGE_FEATURE_NAMES)), dtype=torch.float32)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    if target is not None:
        graph.y = torch.tensor([float(target)], dtype=torch.float32)
    return graph


def calculate_descriptors(smiles_list: Sequence[str]) -> pd.DataFrame:
    """Calculate five interpretable RDKit descriptors for each SMILES string."""

    rows: list[dict[str, float]] = []
    for position, smiles in enumerate(smiles_list):
        molecule = _parse_smiles(smiles, position=position)
        heavy_atoms = molecule.GetNumHeavyAtoms()
        aromatic_atoms = sum(atom.GetIsAromatic() for atom in molecule.GetAtoms())
        rows.append(
            {
                "MolWt": float(Descriptors.MolWt(molecule)),
                "MolLogP": float(Crippen.MolLogP(molecule)),
                "TPSA": float(rdMolDescriptors.CalcTPSA(molecule)),
                "NumRotatableBonds": float(Lipinski.NumRotatableBonds(molecule)),
                "AromaticAtomFraction": (
                    float(aromatic_atoms / heavy_atoms) if heavy_atoms else 0.0
                ),
            }
        )
    return pd.DataFrame(rows, columns=DESCRIPTOR_NAMES, dtype=float)


def make_random_split(n_samples: int, seed: int = 42) -> dict[str, np.ndarray]:
    """Return deterministic, disjoint 80/10/10 positional index arrays."""

    if not isinstance(n_samples, (int, np.integer)) or n_samples < 0:
        raise ValueError("n_samples must be a non-negative integer")

    indices = np.random.default_rng(seed).permutation(int(n_samples))
    train_end = int(0.8 * n_samples)
    validation_end = train_end + (n_samples - train_end) // 2
    return {
        "train": indices[:train_end],
        "validation": indices[train_end:validation_end],
        "test": indices[validation_end:],
    }


def build_baselines(seed: int = 42) -> dict[str, object]:
    """Build the mean, linear, and MLP regression baselines."""

    return {
        "Dummy (mean)": DummyRegressor(strategy="mean"),
        "Linear": make_pipeline(StandardScaler(), LinearRegression()),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                early_stopping=True,
                random_state=seed,
                max_iter=1_000,
            ),
        ),
    }


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """Return RMSE, MAE, and R-squared for a set of predictions."""

    return {
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }
