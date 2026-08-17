"""Network-independent tests for the week 3 scaffold-split API."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from torch_geometric.utils.smiles import from_smiles  # noqa: E402

from ai4sci_molecule.week2 import TrainingConfig  # noqa: E402
from ai4sci_molecule.week3 import (  # noqa: E402
    ACYCLIC_SCAFFOLD,
    METRICS_COLUMNS,
    PREDICTION_COLUMNS,
    SweepConfig,
    aggregate_metrics,
    bin_similarity_errors,
    generalization_gap,
    load_or_run_sweep,
    load_sweep_results,
    make_scaffold_split,
    murcko_scaffold,
    nearest_neighbour_similarity,
    plot_generalization_gap,
    plot_scaffold_distribution,
    plot_similarity_distribution,
    plot_similarity_error,
    plot_split_comparison,
    run_generalization_sweep,
    save_sweep_results,
    scaffold_groups,
    scaffold_statistics,
    scaffold_table,
    similarity_error_correlation,
    split_scaffold_overlap,
    target_shift_table,
)


AROMATIC_SMILES = ("c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "COc1ccccc1", "Nc1ccccc1", "Clc1ccccc1")
PYRIDINE_SMILES = ("c1ccncc1", "Cc1ccncc1", "Nc1ccncc1")
ACYCLIC_SMILES = ("CCO", "CCCO", "CC(=O)O", "CCCCC", "ClCCl")


def make_tiny_smiles():
    """Twenty molecules whose Murcko scaffolds can be counted by inspection."""

    return [
        *AROMATIC_SMILES,  # scaffold c1ccccc1, six molecules
        *PYRIDINE_SMILES,  # scaffold c1ccncc1, three molecules
        "c1ccc2ccccc2c1",
        "Cc1ccc2ccccc2c1",  # naphthalene, two molecules
        "C1CCCCC1",
        "CC1CCCCC1",  # cyclohexane, two molecules
        "c1ccsc1",  # thiophene, one molecule
        "c1ccoc1",  # furan, one molecule
        *ACYCLIC_SMILES,  # no ring system, five molecules
    ]


def make_tiny_dataset():
    """Build graphs for ``make_tiny_smiles`` with spread-out synthetic logS targets."""

    dataset = []
    for position, smiles in enumerate(make_tiny_smiles()):
        graph = from_smiles(smiles)
        graph.y = torch.tensor([-0.5 * position + 1.0], dtype=torch.float32)
        dataset.append(graph)
    return dataset


def make_many_scaffold_smiles():
    """Return 100 molecules spread over thirty ring scaffolds plus ten acyclic ones."""

    smiles = []
    for ring_size in range(3, 33):  # thirty distinct cycloalkane scaffolds
        ring = "C1" + "C" * (ring_size - 1) + "1"
        smiles.extend("C" * substituent + ring for substituent in range(3))
    smiles.extend("C" * (length + 1) + "O" for length in range(10))  # acyclic alcohols
    return smiles


@pytest.fixture
def sweep_metrics():
    rows = []
    for split_type, offset in (("random", 0.0), ("scaffold", 0.5)):
        for seed in (0, 1):
            for model, base in (("MLP", 0.7), ("GCN", 0.8)):
                rows.append(
                    {
                        "split_type": split_type,
                        "seed": seed,
                        "model": model,
                        "representation": "RDKit descriptors",
                        "subset": "test",
                        "RMSE": base + offset + 0.1 * seed,
                        "MAE": base + offset,
                        "R2": 0.5 - offset,
                    }
                )
    return pd.DataFrame(rows)


def test_murcko_scaffold_groups_benzene_toluene_and_phenol_under_one_ring_system():
    assert murcko_scaffold("c1ccccc1") == "c1ccccc1"
    assert murcko_scaffold("Cc1ccccc1") == "c1ccccc1"
    assert murcko_scaffold("Oc1ccccc1") == "c1ccccc1"
    assert murcko_scaffold("c1ccc2ccccc2c1") == "c1ccc2ccccc2c1"
    assert murcko_scaffold("CCO") == ACYCLIC_SCAFFOLD


@pytest.mark.parametrize("invalid", ["not-a-smiles", "", None])
def test_murcko_scaffold_rejects_invalid_smiles(invalid):
    with pytest.raises(ValueError, match="Invalid SMILES"):
        murcko_scaffold(invalid)


@pytest.mark.parametrize("acyclic_policy", ["group", "unique"])
def test_scaffold_groups_treat_acyclic_molecules_according_to_policy(acyclic_policy):
    groups = scaffold_groups(make_tiny_smiles(), acyclic_policy=acyclic_policy)

    acyclic_keys = [
        key
        for key, indices in groups.items()
        if all(murcko_scaffold(make_tiny_smiles()[index]) == ACYCLIC_SCAFFOLD for index in indices)
    ]
    if acyclic_policy == "group":
        assert acyclic_keys == [ACYCLIC_SCAFFOLD]
        assert len(groups[ACYCLIC_SCAFFOLD]) == len(ACYCLIC_SMILES)
    else:
        assert len(acyclic_keys) == len(ACYCLIC_SMILES)
        assert all(len(groups[key]) == 1 for key in acyclic_keys)
    assert len(groups["c1ccccc1"]) == len(AROMATIC_SMILES)


@pytest.mark.parametrize("acyclic_policy", ["group", "unique"])
@pytest.mark.parametrize("seed", [None, 7])
def test_scaffold_split_covers_every_index_exactly_once(acyclic_policy, seed):
    smiles = make_tiny_smiles()

    split = make_scaffold_split(smiles, seed, acyclic_policy=acyclic_policy)

    combined = np.concatenate([split[name] for name in ("train", "validation", "test")])
    np.testing.assert_array_equal(np.sort(combined), np.arange(len(smiles)))


@pytest.mark.parametrize("acyclic_policy", ["group", "unique"])
@pytest.mark.parametrize("seed", [None, 7])
def test_scaffold_split_keeps_scaffolds_disjoint_across_subsets(acyclic_policy, seed):
    smiles = make_tiny_smiles()
    groups = scaffold_groups(smiles, acyclic_policy=acyclic_policy)
    scaffold_of = {index: key for key, indices in groups.items() for index in indices}

    split = make_scaffold_split(smiles, seed, acyclic_policy=acyclic_policy)

    subsets = {
        name: {scaffold_of[int(index)] for index in split[name]}
        for name in ("train", "validation", "test")
    }
    assert subsets["train"].isdisjoint(subsets["validation"])
    assert subsets["train"].isdisjoint(subsets["test"])
    assert subsets["validation"].isdisjoint(subsets["test"])


def test_scaffold_split_respects_requested_proportions():
    smiles = make_many_scaffold_smiles()

    split = make_scaffold_split(smiles)

    assert len(split["train"]) / len(smiles) == pytest.approx(0.8, abs=0.08)
    assert len(split["validation"]) > 0
    assert len(split["test"]) > 0


def test_scaffold_split_is_deterministic_without_a_seed():
    smiles = make_tiny_smiles()

    first = make_scaffold_split(smiles)
    second = make_scaffold_split(smiles)

    for name in ("train", "validation", "test"):
        np.testing.assert_array_equal(first[name], second[name])


def test_scaffold_split_reproduces_the_same_shuffle_for_a_given_seed():
    smiles = make_many_scaffold_smiles()

    first = make_scaffold_split(smiles, 11)
    second = make_scaffold_split(smiles, 11)

    for name in ("train", "validation", "test"):
        np.testing.assert_array_equal(first[name], second[name])


def test_seeded_scaffold_splits_differ_across_seeds():
    smiles = make_many_scaffold_smiles()

    test_sets = [tuple(make_scaffold_split(smiles, seed)["test"]) for seed in (0, 1, 2, 3)]

    assert len(set(test_sets)) > 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_fraction": 0.0},
        {"train_fraction": 1.0},
        {"train_fraction": 1.5},
        {"train_fraction": -0.1},
        {"acyclic_policy": "everything"},
    ],
)
def test_scaffold_split_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        make_scaffold_split(make_tiny_smiles(), **kwargs)


def test_scaffold_split_rejects_an_empty_molecule_list():
    with pytest.raises(ValueError, match="at least one molecule"):
        make_scaffold_split([])


def test_scaffold_split_never_leaves_a_subset_empty_for_a_dominant_scaffold():
    # Ninety benzene derivatives plus ten other ring systems.
    smiles = ["C" * (index + 1) + "c1ccccc1" for index in range(90)]
    smiles += [f"{'C' * index}c1ccc2ccccc2c1" for index in range(1, 6)]
    smiles += [f"{'C' * index}c1ccncc1" for index in range(1, 6)]

    split = make_scaffold_split(smiles)

    assert all(len(split[name]) > 0 for name in ("train", "validation", "test"))


def test_scaffold_statistics_match_a_hand_counted_set():
    statistics = scaffold_statistics(make_tiny_smiles())

    assert statistics["num_molecules"] == 20
    # benzene, pyridine, naphthalene, cyclohexane, thiophene, furan, acyclic
    assert statistics["num_scaffolds"] == 7
    assert statistics["largest_scaffold_size"] == len(AROMATIC_SMILES)
    assert statistics["acyclic_molecule_count"] == len(ACYCLIC_SMILES)
    assert statistics["singleton_scaffold_count"] == 2  # thiophene and furan


def test_scaffold_table_counts_sum_to_the_number_of_molecules():
    table = scaffold_table(make_tiny_smiles())

    assert list(table.columns) == ["scaffold", "molecules", "fraction"]
    assert table["molecules"].sum() == 20
    assert table["fraction"].sum() == pytest.approx(1.0)
    assert table["molecules"].is_monotonic_decreasing


def test_split_scaffold_overlap_reports_no_shared_test_scaffolds_for_a_scaffold_split():
    smiles = make_tiny_smiles()
    split = make_scaffold_split(smiles)

    overlap = split_scaffold_overlap(smiles, split).set_index("subset")

    assert overlap.loc["test", "scaffolds_shared_with_train"] == 0
    assert overlap.loc["test", "shared_molecule_fraction"] == 0.0
    assert overlap.loc["train", "shared_molecule_fraction"] == 1.0


def test_split_scaffold_overlap_reports_shared_test_scaffolds_for_a_leaky_split():
    smiles = ["c1ccccc1", "Oc1ccccc1", "c1ccncc1", "Cc1ccccc1"]
    leaky = {
        "train": np.array([0, 1]),  # benzene, phenol
        "validation": np.array([2]),  # pyridine
        "test": np.array([3]),  # toluene, same scaffold as train
    }

    overlap = split_scaffold_overlap(smiles, leaky).set_index("subset")

    assert overlap.loc["test", "scaffolds_shared_with_train"] == 1
    assert overlap.loc["test", "shared_molecule_fraction"] == 1.0
    assert overlap.loc["validation", "scaffolds_shared_with_train"] == 0


def test_nearest_neighbour_similarity_stays_within_the_unit_interval():
    smiles = make_tiny_smiles()

    similarity = nearest_neighbour_similarity(smiles, smiles)

    assert similarity.shape == (len(smiles),)
    assert ((similarity >= 0) & (similarity <= 1)).all()
    # Every molecule is its own nearest neighbour when the sets are identical.
    np.testing.assert_allclose(similarity, 1.0)


def test_nearest_neighbour_similarity_ranks_toluene_closer_to_benzene_than_to_ethanol():
    to_benzene = nearest_neighbour_similarity(["Cc1ccccc1"], ["c1ccccc1"])
    to_ethanol = nearest_neighbour_similarity(["Cc1ccccc1"], ["CCO"])

    assert to_benzene[0] > to_ethanol[0]


@pytest.mark.parametrize("query,reference", [(["CCO"], []), ([], ["CCO"])])
def test_nearest_neighbour_similarity_rejects_empty_inputs(query, reference):
    with pytest.raises(ValueError):
        nearest_neighbour_similarity(query, reference)


def test_similarity_error_correlation_recovers_a_planted_negative_relationship():
    similarity = np.linspace(0.1, 0.9, 25)
    predictions = pd.DataFrame(
        {
            "split_type": "scaffold",
            "model": "GCN",
            "nearest_train_similarity": similarity,
            "absolute_error": 1.0 - similarity,
        }
    )

    correlation = similarity_error_correlation(predictions)

    assert correlation["pearson"].iloc[0] == pytest.approx(-1.0)
    assert correlation["spearman"].iloc[0] == pytest.approx(-1.0)
    assert correlation["molecules"].iloc[0] == 25


def test_bin_similarity_errors_accounts_for_every_prediction():
    rng = np.random.default_rng(0)
    predictions = pd.DataFrame(
        {
            "split_type": "random",
            "model": "GIN",
            "nearest_train_similarity": rng.uniform(0, 1, 60),
            "absolute_error": rng.uniform(0, 2, 60),
        }
    )

    binned = bin_similarity_errors(predictions)

    assert binned["molecules"].sum() == 60
    assert "similarity_bin" in binned.columns


def test_target_shift_table_reports_train_and_test_statistics_per_split():
    targets = np.arange(10, dtype=float)
    splits = {
        "random": {
            "train": np.arange(0, 6),
            "validation": np.arange(6, 8),
            "test": np.arange(8, 10),
        }
    }

    shift = target_shift_table(targets, splits)

    assert list(shift["subset"]) == ["train", "validation", "test"]
    assert shift.set_index("subset").loc["test", "target_mean"] == pytest.approx(8.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seeds": ()},
        {"seeds": (0, 0)},
        {"split_types": ()},
        {"split_types": ("random", "kmeans")},
        {"include_baselines": False, "include_gnns": False},
        {"acyclic_policy": "everything"},
    ],
)
def test_sweep_config_rejects_invalid_grids(kwargs):
    with pytest.raises(ValueError):
        SweepConfig(**kwargs)


@pytest.mark.parametrize(
    "include_baselines,include_gnns,n_models",
    [(True, False, 3), (False, True, 2)],
)
def test_run_generalization_sweep_returns_the_documented_schema(
    include_baselines, include_gnns, n_models
):
    dataset = make_tiny_dataset()
    config = SweepConfig(
        seeds=(0, 1),
        split_types=("random", "scaffold"),
        include_baselines=include_baselines,
        include_gnns=include_gnns,
        training=TrainingConfig(batch_size=4, max_epochs=2, patience=1),
    )

    result = run_generalization_sweep(dataset, config=config, device="cpu", verbose=False)

    assert tuple(result.metrics.columns) == METRICS_COLUMNS
    assert len(result.metrics) == 2 * 2 * n_models * 2  # splits x seeds x models x subsets
    assert np.isfinite(result.metrics[["RMSE", "MAE", "R2"]].to_numpy()).all()
    assert tuple(result.predictions.columns) == PREDICTION_COLUMNS
    similarity = result.predictions["nearest_train_similarity"]
    assert ((similarity >= 0) & (similarity <= 1)).all()
    np.testing.assert_allclose(
        result.predictions["absolute_error"],
        (result.predictions["actual"] - result.predictions["predicted"]).abs(),
    )
    assert set(result.splits) == {"random:0", "random:1", "scaffold:0", "scaffold:1"}


def test_aggregate_metrics_and_generalization_gap_summarize_every_model(sweep_metrics):
    summary = aggregate_metrics(sweep_metrics)
    gaps = generalization_gap(sweep_metrics)

    assert len(summary) == 4  # two models under two split regimes
    assert set(gaps["model"]) == {"MLP", "GCN"}
    mlp = gaps.set_index("model").loc["MLP"]
    assert mlp["random_mean"] == pytest.approx(0.75)
    assert mlp["scaffold_mean"] == pytest.approx(1.25)
    assert mlp["gap"] == pytest.approx(mlp["scaffold_mean"] - mlp["random_mean"])
    assert mlp["gap_ratio"] == pytest.approx(mlp["scaffold_mean"] / mlp["random_mean"])


def test_generalization_gap_rejects_a_sweep_without_both_regimes(sweep_metrics):
    with pytest.raises(ValueError, match="scaffold"):
        generalization_gap(sweep_metrics[sweep_metrics["split_type"] == "random"])


def test_save_and_load_sweep_results_round_trip(tmp_path):
    dataset = make_tiny_dataset()
    config = SweepConfig(
        seeds=(0,),
        include_gnns=False,
        training=TrainingConfig(batch_size=4, max_epochs=2, patience=1),
    )
    result = run_generalization_sweep(dataset, config=config, device="cpu", verbose=False)

    save_sweep_results(result, tmp_path)
    loaded = load_sweep_results(tmp_path)

    pd.testing.assert_frame_equal(loaded.metrics, result.metrics, check_exact=False, atol=1e-4)
    pd.testing.assert_frame_equal(
        loaded.predictions, result.predictions, check_exact=False, atol=1e-3
    )
    assert loaded.config == result.config
    for key, split in result.splits.items():
        for name, indices in split.items():
            np.testing.assert_array_equal(loaded.splits[key][name], indices)


def test_load_or_run_sweep_reuses_a_cached_directory(tmp_path):
    dataset = make_tiny_dataset()
    config = SweepConfig(
        seeds=(0,),
        include_gnns=False,
        training=TrainingConfig(batch_size=4, max_epochs=2, patience=1),
    )
    first = load_or_run_sweep(dataset, directory=tmp_path, config=config, device="cpu")

    # An empty dataset would fail outright, so returning proves nothing was recomputed.
    second = load_or_run_sweep([], directory=tmp_path, config=config, device="cpu")

    pd.testing.assert_frame_equal(second.metrics, first.metrics, check_exact=False, atol=1e-4)


def test_load_or_run_sweep_warns_and_recomputes_on_a_configuration_mismatch(tmp_path):
    dataset = make_tiny_dataset()
    training = TrainingConfig(batch_size=4, max_epochs=2, patience=1)
    load_or_run_sweep(
        dataset,
        directory=tmp_path,
        config=SweepConfig(seeds=(0,), include_gnns=False, training=training),
        device="cpu",
    )

    with pytest.warns(UserWarning, match="different configuration"):
        refreshed = load_or_run_sweep(
            dataset,
            directory=tmp_path,
            config=SweepConfig(seeds=(0, 1), include_gnns=False, training=training),
            device="cpu",
        )

    assert set(refreshed.metrics["seed"]) == {0, 1}


@pytest.fixture
def plotting_predictions():
    rng = np.random.default_rng(1)
    frames = []
    for split_type in ("random", "scaffold"):
        for model in ("MLP", "GCN"):
            frames.append(
                pd.DataFrame(
                    {
                        "split_type": split_type,
                        "seed": 0,
                        "model": model,
                        "dataset_index": np.arange(40),
                        "nearest_train_similarity": rng.uniform(0, 1, 40),
                        "absolute_error": rng.uniform(0, 2, 40),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def test_plot_helpers_return_populated_figures(sweep_metrics, plotting_predictions):
    figures = [
        plot_scaffold_distribution(make_tiny_smiles(), top_n=5),
        plot_similarity_distribution(plotting_predictions),
        plot_split_comparison(sweep_metrics),
        plot_generalization_gap(sweep_metrics),
        plot_similarity_error(plotting_predictions),
    ]

    try:
        for figure in figures:
            assert figure.axes
            assert any(axis.has_data() for axis in figure.axes)
    finally:
        for figure in figures:
            plt.close(figure)
