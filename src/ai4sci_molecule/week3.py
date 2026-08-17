"""Scaffold-split generalization experiments for week 3."""

from __future__ import annotations

import json
import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data

from .week1 import (
    _parse_smiles,
    build_baselines,
    calculate_descriptors,
    make_random_split,
    regression_metrics,
)
from .week2 import TrainingConfig, build_gnn_models, predict_gnn, resolve_device, train_gnn


ACYCLIC_SCAFFOLD = ""
ACYCLIC_POLICIES = ("group", "unique")
SPLIT_TYPES = ("random", "scaffold", "scaffold_shuffled")
SUBSET_NAMES = ("train", "validation", "test")
MODEL_ORDER = ("Dummy (mean)", "Linear", "MLP", "GCN", "GIN")
DESCRIPTOR_REPRESENTATION = "RDKit descriptors"
GRAPH_REPRESENTATION = "Molecular graph"
DEFAULT_SIMILARITY_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_RESULTS_DIRECTORY = Path("results/week3")

METRIC_NAMES = ("RMSE", "MAE", "R2")
METRICS_COLUMNS = (
    "split_type",
    "seed",
    "model",
    "representation",
    "subset",
    *METRIC_NAMES,
)
PREDICTION_COLUMNS = (
    "split_type",
    "seed",
    "model",
    "representation",
    "dataset_index",
    "smiles",
    "scaffold",
    "actual",
    "predicted",
    "absolute_error",
    "nearest_train_similarity",
)


def murcko_scaffold(smiles: str, *, include_chirality: bool = False) -> str:
    """Return the Murcko scaffold SMILES, or an empty string for acyclic molecules."""

    molecule = _parse_smiles(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule, includeChirality=include_chirality
    )


def scaffold_groups(
    smiles_list: Sequence[str],
    *,
    acyclic_policy: str = "group",
    include_chirality: bool = False,
) -> dict[str, list[int]]:
    """Group dataset indices by Murcko scaffold, honouring the acyclic policy."""

    _validate_acyclic_policy(acyclic_policy)
    if len(smiles_list) == 0:
        raise ValueError("smiles_list must contain at least one molecule")

    groups: dict[str, list[int]] = defaultdict(list)
    for position, smiles in enumerate(smiles_list):
        molecule = _parse_smiles(smiles, position=position)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=molecule, includeChirality=include_chirality
        )
        if scaffold == ACYCLIC_SCAFFOLD and acyclic_policy == "unique":
            scaffold = f"<acyclic:{position}>"
        groups[scaffold].append(position)
    return dict(groups)


def make_scaffold_split(
    smiles_list: Sequence[str],
    seed: int | None = None,
    *,
    train_fraction: float = 0.8,
    acyclic_policy: str = "group",
    include_chirality: bool = False,
) -> dict[str, np.ndarray]:
    """Return disjoint 80/10/10 index arrays whose scaffolds never cross subsets.

    With ``seed=None`` the assignment is fully deterministic: scaffold groups are
    filled largest-first, which is the DeepChem and MoleculeNet convention. Passing
    a seed shuffles the groups that are small enough to move, producing the
    "balanced" scaffold split used as a variance check.
    """

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")

    groups = scaffold_groups(
        smiles_list,
        acyclic_policy=acyclic_policy,
        include_chirality=include_chirality,
    )
    n_samples = len(smiles_list)
    train_size = int(train_fraction * n_samples)
    validation_size = (n_samples - train_size) // 2
    capacities = [train_size, validation_size, n_samples - train_size - validation_size]

    ordered_groups = _ordered_scaffold_groups(groups, seed=seed, capacities=capacities)
    assigned = _greedy_assign(ordered_groups, capacities)

    split = {
        name: np.sort(np.asarray(indices, dtype=int))
        for name, indices in zip(SUBSET_NAMES, assigned, strict=True)
    }
    for name, indices in split.items():
        if len(indices) == 0:
            raise ValueError(
                f"scaffold split produced an empty {name!r} subset; "
                "the dataset has too few scaffold groups for these fractions"
            )
    return split


def _validate_acyclic_policy(acyclic_policy: str) -> None:
    if acyclic_policy not in ACYCLIC_POLICIES:
        raise ValueError(f"acyclic_policy must be one of {ACYCLIC_POLICIES}")


def _ordered_scaffold_groups(
    groups: Mapping[str, Sequence[int]],
    *,
    seed: int | None,
    capacities: Sequence[int],
) -> list[list[int]]:
    # Largest first, then scaffold SMILES ascending, so ties break identically everywhere.
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    if seed is None:
        return [list(indices) for _, indices in ordered]

    # Groups too large for the validation or test bucket can only ever live in train,
    # so shuffling them would wreck the requested proportions.
    movable_limit = min(capacities[1], capacities[2])
    oversized = [list(indices) for _, indices in ordered if len(indices) > movable_limit]
    movable = [list(indices) for _, indices in ordered if len(indices) <= movable_limit]
    permutation = np.random.default_rng(seed).permutation(len(movable))
    return oversized + [movable[int(position)] for position in permutation]


def _greedy_assign(
    ordered_groups: Sequence[Sequence[int]], capacities: Sequence[int]
) -> list[list[int]]:
    assigned: list[list[int]] = [[] for _ in capacities]
    remaining = list(capacities)
    for group in ordered_groups:
        # Always take the emptiest bucket. First-fit would pour every leftover group
        # into validation and leave test empty once a dominant scaffold overflows train.
        bucket = int(np.argmax(remaining))
        assigned[bucket].extend(group)
        remaining[bucket] -= len(group)
    return assigned


def scaffold_statistics(
    smiles_list: Sequence[str],
    *,
    acyclic_policy: str = "group",
    include_chirality: bool = False,
) -> dict[str, float]:
    """Summarize scaffold diversity: counts, singleton share, and largest-group share."""

    groups = scaffold_groups(
        smiles_list,
        acyclic_policy=acyclic_policy,
        include_chirality=include_chirality,
    )
    sizes = np.asarray([len(indices) for indices in groups.values()], dtype=float)
    n_molecules = float(len(smiles_list))
    singleton_count = float((sizes == 1).sum())
    acyclic_count = float(
        sum(
            1
            for smiles in smiles_list
            if murcko_scaffold(smiles, include_chirality=include_chirality)
            == ACYCLIC_SCAFFOLD
        )
    )
    return {
        "num_molecules": n_molecules,
        "num_scaffolds": float(len(groups)),
        "molecules_per_scaffold_mean": float(sizes.mean()),
        "median_scaffold_size": float(np.median(sizes)),
        "largest_scaffold_size": float(sizes.max()),
        "largest_scaffold_fraction": float(sizes.max() / n_molecules),
        "singleton_scaffold_count": singleton_count,
        "singleton_scaffold_fraction": singleton_count / float(len(groups)),
        "singleton_molecule_fraction": singleton_count / n_molecules,
        "acyclic_molecule_count": acyclic_count,
        "acyclic_molecule_fraction": acyclic_count / n_molecules,
    }


def scaffold_table(
    smiles_list: Sequence[str],
    *,
    acyclic_policy: str = "group",
    include_chirality: bool = False,
) -> pd.DataFrame:
    """Return a scaffold SMILES / count / fraction table sorted by descending count."""

    groups = scaffold_groups(
        smiles_list,
        acyclic_policy=acyclic_policy,
        include_chirality=include_chirality,
    )
    rows = [
        {
            "scaffold": scaffold,
            "molecules": len(indices),
            "fraction": len(indices) / len(smiles_list),
        }
        for scaffold, indices in groups.items()
    ]
    table = pd.DataFrame(rows, columns=["scaffold", "molecules", "fraction"])
    return table.sort_values(
        ["molecules", "scaffold"], ascending=[False, True]
    ).reset_index(drop=True)


def split_scaffold_overlap(
    smiles_list: Sequence[str],
    split: Mapping[str, Sequence[int]],
    *,
    acyclic_policy: str = "group",
    include_chirality: bool = False,
) -> pd.DataFrame:
    """Report, per subset, how many scaffolds and molecules it shares with train."""

    groups = scaffold_groups(
        smiles_list,
        acyclic_policy=acyclic_policy,
        include_chirality=include_chirality,
    )
    scaffold_of = {
        index: scaffold for scaffold, indices in groups.items() for index in indices
    }
    train_scaffolds = {scaffold_of[int(index)] for index in split["train"]}

    rows = []
    for name in SUBSET_NAMES:
        indices = [int(index) for index in split[name]]
        scaffolds = [scaffold_of[index] for index in indices]
        shared_scaffolds = {value for value in scaffolds if value in train_scaffolds}
        shared_molecules = sum(1 for value in scaffolds if value in train_scaffolds)
        rows.append(
            {
                "subset": name,
                "molecules": len(indices),
                "scaffolds": len(set(scaffolds)),
                "scaffolds_shared_with_train": len(shared_scaffolds),
                "molecules_with_train_scaffold": shared_molecules,
                "shared_molecule_fraction": (
                    shared_molecules / len(indices) if indices else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def nearest_neighbour_similarity(
    query_smiles: Sequence[str],
    reference_smiles: Sequence[str],
    *,
    radius: int = 2,
    fingerprint_size: int = 2048,
) -> np.ndarray:
    """Return each query molecule's maximum Tanimoto similarity to any reference."""

    if len(query_smiles) == 0:
        raise ValueError("query_smiles must contain at least one molecule")
    if len(reference_smiles) == 0:
        raise ValueError("reference_smiles must contain at least one molecule")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=fingerprint_size
    )
    reference_fingerprints = _morgan_fingerprints(reference_smiles, generator)
    query_fingerprints = _morgan_fingerprints(query_smiles, generator)
    return np.asarray(
        [
            max(DataStructs.BulkTanimotoSimilarity(fingerprint, reference_fingerprints))
            for fingerprint in query_fingerprints
        ],
        dtype=float,
    )


def _morgan_fingerprints(
    smiles_list: Sequence[str],
    generator: rdFingerprintGenerator.FingerprintGenerator64,
) -> list[DataStructs.ExplicitBitVect]:
    return [
        generator.GetFingerprint(_parse_smiles(smiles, position=position))
        for position, smiles in enumerate(smiles_list)
    ]


def target_shift_table(
    targets: Sequence[float],
    splits: Mapping[str, Mapping[str, Sequence[int]]],
) -> pd.DataFrame:
    """Compare train and test logS statistics across splitting regimes."""

    values = np.asarray(targets, dtype=float)
    rows = []
    for split_type, split in splits.items():
        for name in SUBSET_NAMES:
            selected = values[np.asarray(split[name], dtype=int)]
            rows.append(
                {
                    "split_type": split_type,
                    "subset": name,
                    "molecules": len(selected),
                    "target_mean": float(selected.mean()),
                    "target_std": float(selected.std()),
                }
            )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class SweepConfig:
    """Grid definition for the random-versus-scaffold generalization sweep."""

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    # ``scaffold_shuffled`` is the appendix regime: it re-draws the scaffold split per
    # seed, so its spread shows how much of the gap depends on one particular split.
    split_types: tuple[str, ...] = ("random", "scaffold", "scaffold_shuffled")
    include_baselines: bool = True
    include_gnns: bool = True
    acyclic_policy: str = "group"
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(max_epochs=150, patience=20)
    )

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("seeds must contain at least one value")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if not self.split_types:
            raise ValueError("split_types must contain at least one value")
        unknown = set(self.split_types) - set(SPLIT_TYPES)
        if unknown:
            raise ValueError(f"unknown split types: {sorted(unknown)}")
        if not (self.include_baselines or self.include_gnns):
            raise ValueError("at least one model family must be enabled")
        _validate_acyclic_policy(self.acyclic_policy)


@dataclass
class SweepResult:
    """Metrics, per-molecule test predictions, and the splits from one sweep."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    splits: dict[str, dict[str, np.ndarray]]
    scaffold_overlap: pd.DataFrame
    config: SweepConfig


def run_generalization_sweep(
    dataset: Sequence[Data],
    *,
    config: SweepConfig | None = None,
    device: str | torch.device = "auto",
    verbose: bool = True,
) -> SweepResult:
    """Evaluate every baseline and GNN under each splitting regime and seed."""

    config = config or SweepConfig()
    smiles = _dataset_smiles(dataset)
    targets = _dataset_targets(dataset)
    scaffolds = [murcko_scaffold(value) for value in smiles]
    descriptors = calculate_descriptors(smiles).to_numpy(dtype=float)

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    splits: dict[str, dict[str, np.ndarray]] = {}
    overlap_frames: list[pd.DataFrame] = []

    for split_type in config.split_types:
        for seed in config.seeds:
            split = _build_split(split_type, smiles, seed, config)
            splits[f"{split_type}:{seed}"] = split
            test_indices = np.asarray(split["test"], dtype=int)
            similarity = nearest_neighbour_similarity(
                [smiles[int(index)] for index in test_indices],
                [smiles[int(index)] for index in np.asarray(split["train"], dtype=int)],
            )
            if seed == config.seeds[0]:
                overlap = split_scaffold_overlap(
                    smiles, split, acyclic_policy=config.acyclic_policy
                )
                overlap.insert(0, "split_type", split_type)
                overlap_frames.append(overlap)

            if config.include_baselines:
                metrics, predictions = _evaluate_baselines(
                    descriptors, targets, split, seed, verbose=verbose
                )
                metric_rows.extend(_tag_metrics(metrics, split_type, seed))
                prediction_rows.append(
                    _prediction_frame(
                        predictions,
                        split_type=split_type,
                        seed=seed,
                        representation=DESCRIPTOR_REPRESENTATION,
                        indices=test_indices,
                        smiles=smiles,
                        scaffolds=scaffolds,
                        targets=targets,
                        similarity=similarity,
                    )
                )

            if config.include_gnns:
                metrics, predictions = _evaluate_gnns(
                    dataset, split, seed, config, device, verbose=verbose
                )
                metric_rows.extend(_tag_metrics(metrics, split_type, seed))
                prediction_rows.append(
                    _prediction_frame(
                        predictions,
                        split_type=split_type,
                        seed=seed,
                        representation=GRAPH_REPRESENTATION,
                        indices=test_indices,
                        smiles=smiles,
                        scaffolds=scaffolds,
                        targets=targets,
                        similarity=similarity,
                    )
                )

    metrics_frame = pd.DataFrame(metric_rows, columns=list(METRICS_COLUMNS))
    predictions_frame = pd.concat(prediction_rows, ignore_index=True)
    return SweepResult(
        metrics=metrics_frame,
        predictions=predictions_frame[list(PREDICTION_COLUMNS)],
        splits=splits,
        scaffold_overlap=pd.concat(overlap_frames, ignore_index=True),
        config=config,
    )


def _dataset_smiles(dataset: Sequence[Data]) -> list[str]:
    return [str(graph.smiles) for graph in dataset]


def _dataset_targets(dataset: Sequence[Data]) -> np.ndarray:
    return np.asarray(
        [float(graph.y.view(-1)[0]) for graph in dataset], dtype=float
    )


def _build_split(
    split_type: str,
    smiles: Sequence[str],
    seed: int,
    config: SweepConfig,
) -> dict[str, np.ndarray]:
    if split_type == "random":
        return make_random_split(len(smiles), seed=seed)
    if split_type == "scaffold":
        # Deterministic on purpose: every seed sees the same split, so the seed
        # spread measures training noise only.
        return make_scaffold_split(smiles, seed=None, acyclic_policy=config.acyclic_policy)
    if split_type == "scaffold_shuffled":
        return make_scaffold_split(smiles, seed=seed, acyclic_policy=config.acyclic_policy)
    raise ValueError(f"unknown split type: {split_type!r}")


def _evaluate_baselines(
    descriptors: np.ndarray,
    targets: np.ndarray,
    split: Mapping[str, Sequence[int]],
    seed: int,
    *,
    verbose: bool,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    train_indices = np.asarray(split["train"], dtype=int)
    metrics: list[dict[str, object]] = []
    test_predictions: dict[str, np.ndarray] = {}

    for name, model in build_baselines(seed=seed).items():
        model.fit(descriptors[train_indices], targets[train_indices])
        for subset in ("validation", "test"):
            indices = np.asarray(split[subset], dtype=int)
            predicted = np.asarray(model.predict(descriptors[indices]), dtype=float)
            metrics.append(
                {
                    "model": name,
                    "representation": DESCRIPTOR_REPRESENTATION,
                    "subset": subset,
                    **regression_metrics(targets[indices], predicted),
                }
            )
            if subset == "test":
                test_predictions[name] = predicted
        if verbose:
            print(f"  {name}: 완료")
    return metrics, test_predictions


def _evaluate_gnns(
    dataset: Sequence[Data],
    split: Mapping[str, Sequence[int]],
    seed: int,
    config: SweepConfig,
    device: str | torch.device,
    *,
    verbose: bool,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    targets = _dataset_targets(dataset)
    metrics: list[dict[str, object]] = []
    test_predictions: dict[str, np.ndarray] = {}

    for name, model in build_gnn_models(seed=seed).items():
        result = train_gnn(
            model,
            dataset,
            split,
            config=replace(config.training, seed=seed),
            device=device,
        )
        for subset in ("validation", "test"):
            indices = np.asarray(split[subset], dtype=int)
            predicted = predict_gnn(result, dataset, indices, device=device)
            metrics.append(
                {
                    "model": name,
                    "representation": GRAPH_REPRESENTATION,
                    "subset": subset,
                    **regression_metrics(targets[indices], predicted),
                }
            )
            if subset == "test":
                test_predictions[name] = predicted
        if verbose:
            print(f"  {name}: best epoch {result.best_epoch}")
    return metrics, test_predictions


def _tag_metrics(
    metrics: Sequence[Mapping[str, object]], split_type: str, seed: int
) -> list[dict[str, object]]:
    return [{"split_type": split_type, "seed": seed, **row} for row in metrics]


def _prediction_frame(
    predictions: Mapping[str, np.ndarray],
    *,
    split_type: str,
    seed: int,
    representation: str,
    indices: np.ndarray,
    smiles: Sequence[str],
    scaffolds: Sequence[str],
    targets: np.ndarray,
    similarity: np.ndarray,
) -> pd.DataFrame:
    frames = []
    for model, predicted in predictions.items():
        actual = targets[indices]
        frames.append(
            pd.DataFrame(
                {
                    "split_type": split_type,
                    "seed": seed,
                    "model": model,
                    "representation": representation,
                    "dataset_index": indices,
                    "smiles": [smiles[int(index)] for index in indices],
                    "scaffold": [scaffolds[int(index)] for index in indices],
                    "actual": actual,
                    "predicted": predicted,
                    "absolute_error": np.abs(actual - predicted),
                    "nearest_train_similarity": similarity,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def aggregate_metrics(metrics: pd.DataFrame, *, subset: str = "test") -> pd.DataFrame:
    """Average each metric over seeds and report the seed standard deviation.

    The two standard deviations do not measure the same thing: for ``random`` the
    seed changes both the split and the training run, while for the deterministic
    ``scaffold`` split it changes only the training run.
    """

    selected = metrics[metrics["subset"] == subset]
    if selected.empty:
        raise ValueError(f"metrics contains no rows for subset {subset!r}")

    grouped = selected.groupby(
        ["split_type", "model", "representation"], sort=False, observed=True
    )
    summary = grouped[list(METRIC_NAMES)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary.insert(0, "seeds", grouped["seed"].nunique())
    return _order_by_model(summary.reset_index())


def generalization_gap(
    metrics: pd.DataFrame, *, metric: str = "RMSE", subset: str = "test"
) -> pd.DataFrame:
    """Contrast random-split and scaffold-split performance for every model.

    ``gap_over_pooled_std`` is a descriptive effect size, not a test statistic: with
    five seeds, different test molecules per regime, and a deterministic scaffold
    split, no standard significance test applies here.
    """

    if metric not in METRIC_NAMES:
        raise ValueError(f"metric must be one of {METRIC_NAMES}")
    summary = aggregate_metrics(metrics, subset=subset)
    for split_type in ("random", "scaffold"):
        if split_type not in set(summary["split_type"]):
            raise ValueError(f"metrics is missing the {split_type!r} split")

    pivot = summary.pivot_table(
        index=["model", "representation"],
        columns="split_type",
        values=[f"{metric}_mean", f"{metric}_std"],
        observed=True,
    )
    rows = []
    for (model, representation), values in pivot.iterrows():
        random_mean = float(values[(f"{metric}_mean", "random")])
        scaffold_mean = float(values[(f"{metric}_mean", "scaffold")])
        random_std = float(values[(f"{metric}_std", "random")])
        scaffold_std = float(values[(f"{metric}_std", "scaffold")])
        pooled_std = float(np.sqrt((random_std**2 + scaffold_std**2) / 2))
        rows.append(
            {
                "model": model,
                "representation": representation,
                "random_mean": random_mean,
                "random_std": random_std,
                "scaffold_mean": scaffold_mean,
                "scaffold_std": scaffold_std,
                "gap": scaffold_mean - random_mean,
                "gap_ratio": scaffold_mean / random_mean,
                "gap_over_pooled_std": (
                    (scaffold_mean - random_mean) / pooled_std
                    if pooled_std > 0
                    else float("nan")
                ),
            }
        )
    return _order_by_model(pd.DataFrame(rows))


def similarity_error_correlation(
    predictions: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("split_type", "model"),
) -> pd.DataFrame:
    """Correlate nearest-train-neighbour similarity with absolute error per group."""

    rows = []
    for keys, group in predictions.groupby(list(group_columns), sort=False, observed=True):
        similarity = group["nearest_train_similarity"].to_numpy(dtype=float)
        error = group["absolute_error"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(group_columns, _as_tuple(keys), strict=True)),
                "molecules": len(group),
                "pearson": _correlation(similarity, error),
                "spearman": _correlation(
                    group["nearest_train_similarity"].rank().to_numpy(dtype=float),
                    group["absolute_error"].rank().to_numpy(dtype=float),
                ),
                "mean_similarity": float(similarity.mean()),
                "mean_absolute_error": float(error.mean()),
            }
        )
    return _order_by_model(pd.DataFrame(rows))


def bin_similarity_errors(
    predictions: pd.DataFrame,
    *,
    bins: Sequence[float] = DEFAULT_SIMILARITY_BINS,
    group_columns: Sequence[str] = ("split_type", "model"),
) -> pd.DataFrame:
    """Return mean absolute error and molecule count per similarity bin."""

    binned = predictions.assign(
        similarity_bin=pd.cut(
            predictions["nearest_train_similarity"],
            bins=list(bins),
            include_lowest=True,
        ).astype(str)
    )
    grouped = binned.groupby(
        [*group_columns, "similarity_bin"], sort=False, observed=True
    )["absolute_error"]
    summary = grouped.agg(
        molecules="size", mean_absolute_error="mean", std_absolute_error="std"
    )
    return _order_by_model(summary.reset_index())


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or left.std() == 0 or right.std() == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _as_tuple(keys: object) -> tuple[object, ...]:
    return keys if isinstance(keys, tuple) else (keys,)


def _order_by_model(frame: pd.DataFrame) -> pd.DataFrame:
    if "model" not in frame.columns:
        return frame
    ordering = {name: position for position, name in enumerate(MODEL_ORDER)}
    key = frame["model"].map(lambda name: ordering.get(name, len(ordering)))
    return frame.assign(_order=key).sort_values(
        "_order", kind="stable"
    ).drop(columns="_order").reset_index(drop=True)


def save_sweep_results(
    result: SweepResult, directory: str | Path = DEFAULT_RESULTS_DIRECTORY
) -> dict[str, Path]:
    """Write metrics, predictions, splits, and the sweep configuration to disk."""

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": target / "metrics.csv",
        "predictions": target / "predictions.csv",
        "scaffold_overlap": target / "scaffold_overlap.csv",
        "splits": target / "splits.json",
        "sweep_config": target / "sweep_config.json",
    }
    result.metrics.to_csv(paths["metrics"], index=False, float_format="%.6f")
    result.predictions.to_csv(paths["predictions"], index=False, float_format="%.4f")
    result.scaffold_overlap.to_csv(paths["scaffold_overlap"], index=False)
    paths["splits"].write_text(
        json.dumps(
            {
                key: {name: [int(i) for i in indices] for name, indices in split.items()}
                for key, split in result.splits.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["sweep_config"].write_text(
        json.dumps(_config_to_dict(result.config), indent=2), encoding="utf-8"
    )
    return paths


def load_sweep_results(
    directory: str | Path = DEFAULT_RESULTS_DIRECTORY,
) -> SweepResult:
    """Load a previously saved sweep from disk."""

    target = Path(directory)
    splits_payload = json.loads((target / "splits.json").read_text(encoding="utf-8"))
    return SweepResult(
        metrics=_read_csv(target / "metrics.csv", ("seed", *METRIC_NAMES)),
        predictions=_read_csv(
            target / "predictions.csv",
            (
                "seed",
                "dataset_index",
                "actual",
                "predicted",
                "absolute_error",
                "nearest_train_similarity",
            ),
        ),
        splits={
            key: {
                name: np.asarray(indices, dtype=int) for name, indices in split.items()
            }
            for key, split in splits_payload.items()
        },
        scaffold_overlap=pd.read_csv(target / "scaffold_overlap.csv"),
        config=_config_from_dict(
            json.loads((target / "sweep_config.json").read_text(encoding="utf-8"))
        ),
    )


def load_or_run_sweep(
    dataset: Sequence[Data],
    *,
    directory: str | Path = DEFAULT_RESULTS_DIRECTORY,
    config: SweepConfig | None = None,
    device: str | torch.device = "auto",
    refresh: bool = False,
) -> SweepResult:
    """Reuse cached sweep results when they match the requested configuration."""

    config = config or SweepConfig()
    target = Path(directory)
    if not refresh and (target / "sweep_config.json").exists():
        cached = load_sweep_results(target)
        if cached.config == config:
            return cached
        warnings.warn(
            f"cached sweep in {target} was built with a different configuration; "
            "recomputing",
            stacklevel=2,
        )
    result = run_generalization_sweep(dataset, config=config, device=device)
    save_sweep_results(result, target)
    return result


def _read_csv(path: Path, numeric_columns: Sequence[str]) -> pd.DataFrame:
    # keep_default_na=False preserves the empty scaffold SMILES of acyclic molecules.
    frame = pd.read_csv(path, keep_default_na=False)
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column])
    return frame


def _config_to_dict(config: SweepConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["seeds"] = list(config.seeds)
    payload["split_types"] = list(config.split_types)
    return payload


def _config_from_dict(payload: Mapping[str, object]) -> SweepConfig:
    training = TrainingConfig(**payload["training"])
    return SweepConfig(
        seeds=tuple(payload["seeds"]),
        split_types=tuple(payload["split_types"]),
        include_baselines=bool(payload["include_baselines"]),
        include_gnns=bool(payload["include_gnns"]),
        acyclic_policy=str(payload["acyclic_policy"]),
        training=training,
    )


def plot_scaffold_distribution(
    smiles_list: Sequence[str],
    *,
    top_n: int = 15,
    acyclic_policy: str = "group",
    title: str | None = None,
) -> Figure:
    """Plot the scaffold rank-size curve and the most common scaffolds."""

    table = scaffold_table(smiles_list, acyclic_policy=acyclic_policy)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))

    sizes = table["molecules"].to_numpy()
    axes[0].plot(np.arange(1, len(sizes) + 1), sizes, marker="o", markersize=3)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Scaffold rank")
    axes[0].set_ylabel("Molecules per scaffold")
    axes[0].set_title(f"{(sizes == 1).sum()} singleton scaffolds of {len(sizes)}")

    head = table.head(top_n).iloc[::-1]
    labels = [
        "(acyclic)" if scaffold == ACYCLIC_SCAFFOLD else _shorten(scaffold)
        for scaffold in head["scaffold"]
    ]
    axes[1].barh(labels, head["molecules"])
    axes[1].set_xlabel("Molecules")
    axes[1].set_title(f"Top {top_n} scaffolds")

    if title:
        figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_similarity_distribution(
    predictions: pd.DataFrame, *, title: str | None = None
) -> Figure:
    """Compare nearest-train-neighbour similarity distributions across regimes."""

    figure, axis = plt.subplots(figsize=(9, 5))
    unique = predictions.drop_duplicates(["split_type", "seed", "dataset_index"])
    for split_type, group in unique.groupby("split_type", sort=False, observed=True):
        values = np.sort(group["nearest_train_similarity"].to_numpy(dtype=float))
        axis.plot(
            values,
            np.linspace(0, 1, len(values)),
            label=f"{split_type} (median {np.median(values):.2f})",
        )
    axis.set_xlabel("Max Tanimoto similarity to any training molecule")
    axis.set_ylabel("Cumulative fraction of test molecules")
    axis.set_xlim(0, 1)
    axis.legend()
    if title:
        axis.set_title(title)
    figure.tight_layout()
    return figure


def plot_split_comparison(
    metrics: pd.DataFrame,
    *,
    metric: str = "RMSE",
    subset: str = "test",
    title: str | None = None,
) -> Figure:
    """Compare per-model performance across splitting regimes with seed error bars."""

    summary = aggregate_metrics(metrics, subset=subset)
    split_types = list(dict.fromkeys(summary["split_type"]))
    models = list(dict.fromkeys(summary["model"]))
    positions = np.arange(len(models), dtype=float)
    width = 0.8 / len(split_types)

    figure, axis = plt.subplots(figsize=(10, 5))
    for offset, split_type in enumerate(split_types):
        rows = summary[summary["split_type"] == split_type].set_index("model")
        axis.bar(
            positions + offset * width - 0.4 + width / 2,
            [rows.loc[model, f"{metric}_mean"] for model in models],
            width=width,
            yerr=[rows.loc[model, f"{metric}_std"] for model in models],
            capsize=4,
            label=split_type,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(models)
    axis.set_ylabel(f"{subset} {metric} (logS units)")
    axis.legend(title="Split")
    if title:
        axis.set_title(title)
    figure.tight_layout()
    return figure


def plot_generalization_gap(
    metrics: pd.DataFrame, *, metric: str = "RMSE", title: str | None = None
) -> Figure:
    """Show how much each model degrades when scaffolds no longer overlap."""

    gaps = generalization_gap(metrics, metric=metric).iloc[::-1]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.barh(gaps["model"], gaps["gap"])
    baseline = gaps[gaps["model"] == "Dummy (mean)"]["gap"]
    if not baseline.empty:
        axis.axvline(
            float(baseline.iloc[0]),
            linestyle="--",
            color="0.3",
            label="Dummy baseline (target shift floor)",
        )
        axis.legend()
    axis.set_xlabel(f"Scaffold {metric} - random {metric} (logS units)")
    if title:
        axis.set_title(title)
    figure.tight_layout()
    return figure


def plot_similarity_error(
    predictions: pd.DataFrame,
    *,
    models: Sequence[str] | None = None,
    bins: Sequence[float] = DEFAULT_SIMILARITY_BINS,
    title: str | None = None,
) -> Figure:
    """Relate nearest-train-neighbour similarity to absolute prediction error."""

    selected = (
        predictions
        if models is None
        else predictions[predictions["model"].isin(list(models))]
    )
    split_types = list(dict.fromkeys(selected["split_type"]))
    binned = bin_similarity_errors(selected, bins=bins)
    correlations = similarity_error_correlation(selected)
    centres = (np.asarray(bins[:-1]) + np.asarray(bins[1:])) / 2
    labels = pd.cut(centres, bins=list(bins), include_lowest=True).astype(str)

    figure, axes = plt.subplots(
        1, len(split_types), figsize=(6 * len(split_types), 5), sharey=True, squeeze=False
    )
    # A handful of badly mispredicted molecules would otherwise flatten every binned
    # mean into the bottom of the panel, so clip the shared axis instead.
    upper_limit = float(np.percentile(selected["absolute_error"], 98))
    for axis, split_type in zip(axes[0], split_types, strict=True):
        panel = selected[selected["split_type"] == split_type]
        axis.scatter(
            panel["nearest_train_similarity"],
            panel["absolute_error"],
            s=6,
            alpha=0.15,
            color="0.5",
        )
        rho = correlations[correlations["split_type"] == split_type].set_index("model")[
            "spearman"
        ]
        for model, group in binned[binned["split_type"] == split_type].groupby(
            "model", sort=False, observed=True
        ):
            means = group.set_index("similarity_bin")["mean_absolute_error"]
            axis.plot(
                centres,
                [means.get(label, np.nan) for label in labels],
                marker="o",
                label=f"{model} (ρ {rho.get(model, float('nan')):+.2f})",
            )
        axis.set_title(
            f"{split_type} (median similarity "
            f"{panel['nearest_train_similarity'].median():.2f})"
        )
        axis.set_xlabel("Max Tanimoto similarity to training set")
        axis.set_xlim(0, 1)
        axis.legend(fontsize="small")
    axes[0][0].set_ylabel("Mean absolute error per bin (logS units)")
    axes[0][0].set_ylim(0, upper_limit)
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    return figure


def save_figures(
    figures: Mapping[str, Figure],
    directory: str | Path = DEFAULT_RESULTS_DIRECTORY / "figures",
) -> dict[str, Path]:
    """Write each figure to a PNG file named after its mapping key."""

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, figure in figures.items():
        path = target / f"{name}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        paths[name] = path
    return paths


def _shorten(text: str, *, limit: int = 28) -> str:
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def main() -> None:
    """Run the default sweep on ESOL and write every artifact under results/week3."""

    from .week1 import load_esol

    dataset = load_esol()
    smiles = _dataset_smiles(dataset)
    targets = _dataset_targets(dataset)

    result = load_or_run_sweep(dataset, refresh=True)
    gaps = generalization_gap(result.metrics)
    correlations = similarity_error_correlation(result.predictions)
    shift = target_shift_table(
        targets,
        {
            "random": result.splits[f"random:{result.config.seeds[0]}"],
            "scaffold": result.splits[f"scaffold:{result.config.seeds[0]}"],
        },
    )

    directory = DEFAULT_RESULTS_DIRECTORY
    scaffolds = scaffold_table(smiles, acyclic_policy=result.config.acyclic_policy)
    scaffolds.to_csv(directory / "scaffold_table.csv", index=False)
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "scaffold_statistics": scaffold_statistics(
                    smiles, acyclic_policy=result.config.acyclic_policy
                ),
                "aggregate": aggregate_metrics(result.metrics).to_dict("records"),
                "generalization_gap": gaps.to_dict("records"),
                "similarity_error_correlation": correlations.to_dict("records"),
                "similarity_bins": bin_similarity_errors(result.predictions).to_dict(
                    "records"
                ),
                "target_shift": shift.to_dict("records"),
                "environment": {
                    "torch": torch.__version__,
                    "device": str(resolve_device("auto")),
                    "device_name": (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available()
                        else "cpu"
                    ),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    figures = {
        "scaffold_distribution": plot_scaffold_distribution(smiles),
        "similarity_distribution": plot_similarity_distribution(result.predictions),
        "split_comparison": plot_split_comparison(result.metrics),
        "generalization_gap": plot_generalization_gap(result.metrics),
        "similarity_error": plot_similarity_error(result.predictions),
    }
    save_figures(figures)
    for figure in figures.values():
        plt.close(figure)

    print(gaps.to_string(index=False))


if __name__ == "__main__":
    main()
