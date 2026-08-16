"""Graph neural network models and training helpers for week 2."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool
from torch_geometric.utils.smiles import e_map, x_map

from .week1 import regression_metrics


ATOM_FEATURE_CARDINALITIES = tuple(len(values) for values in x_map.values())
BOND_FEATURE_CARDINALITIES = tuple(len(values) for values in e_map.values())


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""

    if device == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return resolved


class CategoricalFeatureEncoder(nn.Module):
    """Embed each categorical feature column and sum the embeddings."""

    def __init__(self, cardinalities: Sequence[int], hidden_channels: int) -> None:
        super().__init__()
        if not cardinalities or any(size <= 0 for size in cardinalities):
            raise ValueError("cardinalities must contain positive integers")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")

        self.embeddings = nn.ModuleList(
            nn.Embedding(size, hidden_channels) for size in cardinalities
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.size(1) != len(self.embeddings):
            raise ValueError(
                f"expected [num_items, {len(self.embeddings)}] categorical features, "
                f"got {tuple(features.shape)}"
            )
        if features.dtype not in (torch.int32, torch.int64):
            raise TypeError("categorical graph features must use an integer dtype")

        encoded = self.embeddings[0](features[:, 0])
        for column, embedding in enumerate(self.embeddings[1:], start=1):
            encoded = encoded + embedding(features[:, column])
        return encoded


class GCNRegressor(nn.Module):
    """Three-stage GCN molecular regressor with global mean pooling."""

    def __init__(
        self,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        _validate_model_options(hidden_channels, num_layers, dropout)
        self.atom_encoder = CategoricalFeatureEncoder(
            ATOM_FEATURE_CARDINALITIES, hidden_channels
        )
        self.convolutions = nn.ModuleList(
            GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)
        )
        self.normalizations = nn.ModuleList(
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        )
        self.dropout = dropout
        self.head = _regression_head(hidden_channels)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        del edge_attr  # A vanilla GCN uses graph topology, but not bond categories.
        hidden = self.atom_encoder(x)
        for convolution, normalization in zip(
            self.convolutions, self.normalizations, strict=True
        ):
            hidden = convolution(hidden, edge_index)
            hidden = normalization(hidden)
            hidden = F.relu(hidden)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        graph_embedding = global_mean_pool(hidden, batch)
        return self.head(graph_embedding).view(-1)


class GINRegressor(nn.Module):
    """Bond-aware GIN (GINE) molecular regressor with mean pooling."""

    def __init__(
        self,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        _validate_model_options(hidden_channels, num_layers, dropout)
        self.atom_encoder = CategoricalFeatureEncoder(
            ATOM_FEATURE_CARDINALITIES, hidden_channels
        )
        self.bond_encoder = CategoricalFeatureEncoder(
            BOND_FEATURE_CARDINALITIES, hidden_channels
        )
        self.convolutions = nn.ModuleList(
            GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                ),
                edge_dim=hidden_channels,
                train_eps=True,
            )
            for _ in range(num_layers)
        )
        self.normalizations = nn.ModuleList(
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        )
        self.dropout = dropout
        self.head = _regression_head(hidden_channels)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        hidden = self.atom_encoder(x)
        bond_embedding = self.bond_encoder(edge_attr)
        for convolution, normalization in zip(
            self.convolutions, self.normalizations, strict=True
        ):
            hidden = convolution(hidden, edge_index, bond_embedding)
            hidden = normalization(hidden)
            hidden = F.relu(hidden)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        graph_embedding = global_mean_pool(hidden, batch)
        return self.head(graph_embedding).view(-1)


def _validate_model_options(
    hidden_channels: int, num_layers: int, dropout: float
) -> None:
    if hidden_channels <= 0:
        raise ValueError("hidden_channels must be positive")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")


def _regression_head(hidden_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_channels, hidden_channels),
        nn.ReLU(),
        nn.Linear(hidden_channels, 1),
    )


@dataclass(frozen=True)
class TrainingConfig:
    """Shared optimization settings for the GCN and GIN experiments."""

    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 30
    min_delta: float = 1e-4
    seed: int = 42

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative")


@dataclass
class TrainingResult:
    """Best validation checkpoint and learning history from one training run."""

    model: nn.Module
    history: list[dict[str, float | int]]
    best_epoch: int
    best_validation_metrics: dict[str, float]
    target_mean: float
    target_std: float


def build_gnn_models(
    hidden_channels: int = 64,
    num_layers: int = 3,
    dropout: float = 0.2,
    seed: int = 42,
) -> dict[str, nn.Module]:
    """Build deterministic GCN and bond-aware GIN model initializations."""

    models: dict[str, nn.Module] = {}
    for name, model_type in (("GCN", GCNRegressor), ("GIN", GINRegressor)):
        seed_everything(seed)
        models[name] = model_type(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
    return models


def train_gnn(
    model: nn.Module,
    dataset: Sequence[Data],
    split: Mapping[str, Sequence[int]],
    config: TrainingConfig | None = None,
    device: str | torch.device = "auto",
) -> TrainingResult:
    """Train with normalized targets and select a checkpoint by validation RMSE."""

    config = config or TrainingConfig()
    train_indices = _split_indices(split, "train")
    validation_indices = _split_indices(split, "validation")
    resolved_device = resolve_device(device)
    seed_everything(config.seed)

    train_targets = _targets_for_indices(dataset, train_indices)
    target_mean = float(train_targets.mean())
    target_std = float(train_targets.std())
    if not np.isfinite(target_std) or target_std == 0:
        raise ValueError("training targets must have non-zero finite variance")

    train_loader = _make_loader(
        dataset,
        train_indices,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    validation_loader = _make_loader(
        dataset,
        validation_indices,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
    )

    model = model.to(resolved_device)
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, float | int]] = []
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_rmse = float("inf")
    best_metrics: dict[str, float] = {}
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_graphs = 0
        for batch in train_loader:
            batch = batch.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward_batch(model, batch)
            target = (batch.y.view(-1) - target_mean) / target_std
            loss = F.mse_loss(prediction, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch.num_graphs
            total_graphs += batch.num_graphs

        validation_target, validation_prediction = _predict_loader(
            model,
            validation_loader,
            resolved_device,
            target_mean,
            target_std,
        )
        metrics = regression_metrics(validation_target, validation_prediction)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / total_graphs,
                **{f"validation_{name}": value for name, value in metrics.items()},
            }
        )

        if metrics["RMSE"] < best_rmse - config.min_delta:
            best_rmse = metrics["RMSE"]
            best_epoch = epoch
            best_metrics = metrics
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:  # Defensive: at least epoch one should always improve infinity.
        raise RuntimeError("training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.cpu()
    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_validation_metrics=best_metrics,
        target_mean=target_mean,
        target_std=target_std,
    )


def predict_gnn(
    result: TrainingResult,
    dataset: Sequence[Data],
    indices: Sequence[int],
    *,
    batch_size: int = 256,
    device: str | torch.device = "auto",
) -> np.ndarray:
    """Predict logS in the original target scale for selected graph indices."""

    selected_indices = np.asarray(indices, dtype=int)
    if selected_indices.ndim != 1 or len(selected_indices) == 0:
        raise ValueError("indices must be a non-empty one-dimensional sequence")
    loader = _make_loader(
        dataset,
        selected_indices,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )
    resolved_device = resolve_device(device)
    model = result.model.to(resolved_device)
    _, predictions = _predict_loader(
        model,
        loader,
        resolved_device,
        result.target_mean,
        result.target_std,
    )
    model.cpu()
    return predictions


def _split_indices(
    split: Mapping[str, Sequence[int]], name: str
) -> np.ndarray:
    if name not in split:
        raise ValueError(f"split is missing the {name!r} indices")
    indices = np.asarray(split[name], dtype=int)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError(f"split {name!r} must be a non-empty one-dimensional sequence")
    return indices


def _targets_for_indices(
    dataset: Sequence[Data], indices: Sequence[int]
) -> np.ndarray:
    targets = [float(dataset[int(index)].y.view(-1)[0]) for index in indices]
    values = np.asarray(targets, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("dataset targets must be finite")
    return values


def _make_loader(
    dataset: Sequence[Data],
    indices: Sequence[int],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    subset = Subset(dataset, [int(index) for index in indices])
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _forward_batch(model: nn.Module, batch: Data) -> Tensor:
    return model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)


@torch.no_grad()
def _predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets: list[Tensor] = []
    predictions: list[Tensor] = []
    for batch in loader:
        batch = batch.to(device)
        standardized_prediction = _forward_batch(model, batch)
        prediction = standardized_prediction * target_std + target_mean
        targets.append(batch.y.view(-1).detach().cpu())
        predictions.append(prediction.detach().cpu())
    return (
        torch.cat(targets).numpy(),
        torch.cat(predictions).numpy(),
    )
