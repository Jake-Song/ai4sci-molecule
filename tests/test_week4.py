"""Network-independent tests for the week 4 error-analysis API."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from ai4sci_molecule.week1 import DESCRIPTOR_NAMES, calculate_descriptors  # noqa: E402
from ai4sci_molecule.week3 import PREDICTION_COLUMNS  # noqa: E402
from ai4sci_molecule.week4 import (  # noqa: E402
    BASELINE_MODEL,
    CASE_STUDY_COLUMNS,
    CORRELATION_COLUMNS,
    IMPORTANCE_COLUMNS,
    MIN_BASELINE_ERROR,
    PROFILE_COLUMNS,
    PROPERTY_NAMES,
    SHRINKAGE_COLUMNS,
    AnalysisConfig,
    bin_property_errors,
    bootstrap_correlation_interval,
    build_error_profile,
    build_figures,
    case_studies,
    effective_sample_sizes,
    error_property_importance,
    load_error_analysis,
    load_or_run_error_analysis,
    model_error_agreement,
    model_error_disagreement,
    molecular_properties,
    partial_correlation,
    plot_case_study_structures,
    plot_error_property_importance,
    plot_model_error_agreement,
    plot_property_collinearity,
    plot_property_error_bins,
    plot_property_error_correlation,
    plot_residual_shrinkage,
    plot_scaffold_errors,
    pool_split_types,
    property_collinearity,
    property_correlation_matrix,
    property_error_correlation,
    property_reference_table,
    property_table,
    rank_correlation,
    run_error_analysis,
    save_error_analysis,
    scaffold_error_table,
    shrinkage_fit,
    summarise,
)


AROMATIC_SMILES = (
    "c1ccccc1",
    "Cc1ccccc1",
    "Oc1ccccc1",
    "COc1ccccc1",
    "Nc1ccccc1",
    "Clc1ccccc1",
    "CCc1ccccc1",
    "CCCc1ccccc1",
)
NAPHTHALENE_SMILES = ("c1ccc2ccccc2c1", "Cc1ccc2ccccc2c1", "Oc1ccc2ccccc2c1")
CYCLOHEXANE_SMILES = ("C1CCCCC1", "CC1CCCCC1", "OC1CCCCC1")
ACYCLIC_SMILES = ("CCO", "CCCO", "CC(=O)O", "CCCCC", "ClCCl", "CCCCCCO")


def make_smiles():
    """Twenty molecules whose scaffolds and properties can be checked by inspection."""

    return [*AROMATIC_SMILES, *NAPHTHALENE_SMILES, *CYCLOHEXANE_SMILES, *ACYCLIC_SMILES]


def make_predictions(*, seeds=(0, 1, 2), split_types=("random", "scaffold")):
    """Build a predictions frame with analytically known structure planted in it.

    ``GIN``'s absolute error rises monotonically with ``MolWt`` and its predictions
    shrink toward the mean with a slope of exactly 0.5. ``MLP``'s error is flat in
    ``MolWt`` instead, so the agreement and correlation tests have known answers. Two
    molecules carry a planted blow-up, one for each model, and the naphthalene scaffold
    carries an inflated ``GIN`` error.
    """

    smiles = make_smiles()
    properties = molecular_properties(smiles)
    weights = properties["MolWt"].to_numpy(dtype=float)
    order = weights.argsort().argsort().astype(float)
    scaffolds = []
    for value in smiles:
        if value in AROMATIC_SMILES:
            scaffolds.append("c1ccccc1")
        elif value in NAPHTHALENE_SMILES:
            scaffolds.append("c1ccc2ccccc2c1")
        elif value in CYCLOHEXANE_SMILES:
            scaffolds.append("C1CCCCC1")
        else:
            scaffolds.append("")

    # Targets spread over -6..0 and deliberately uncorrelated with MolWt rank, so the
    # planted error/property relationship cannot be an artefact of target difficulty.
    actual = np.linspace(-6.0, 0.0, len(smiles))[(np.arange(len(smiles)) * 7) % len(smiles)]
    train_mean = float(actual.mean())
    generator = np.random.default_rng(0)

    rows = []
    for split_type in split_types:
        for seed in seeds:
            if split_type == "random":
                # A different, overlapping subset per seed, so seed coverage is ragged.
                members = [
                    index
                    for index in range(len(smiles))
                    if (index + seed) % 3 != 0 or index < 6
                ]
            else:
                members = list(range(len(smiles)))
            for index in members:
                jitter = float(generator.normal(scale=0.01))
                planted = {
                    "Dummy (mean)": train_mean,
                    "Linear": 0.7 * actual[index] - 0.3 + jitter,
                    "MLP": actual[index] + 0.4 + 0.05 * properties["NumHDonors"][index] + jitter,
                    "GCN": 0.5 * actual[index] + 0.5 * train_mean + 0.12 * order[index] + jitter,
                    "GIN": 0.5 * actual[index] + 0.5 * train_mean + 0.1 * order[index] + jitter,
                }
                for model, predicted in planted.items():
                    if model == "GIN" and index == 0:
                        predicted = actual[index] + 6.0
                    if model == "MLP" and index == 1:
                        predicted = actual[index] - 6.0
                    if model == "GIN" and scaffolds[index] == "c1ccc2ccccc2c1":
                        predicted = actual[index] + 3.0
                    rows.append(
                        {
                            "split_type": split_type,
                            "seed": seed,
                            "model": model,
                            "representation": (
                                "Molecular graph" if model in ("GCN", "GIN")
                                else "RDKit descriptors"
                            ),
                            "dataset_index": index,
                            "smiles": smiles[index],
                            "scaffold": scaffolds[index],
                            "actual": float(actual[index]),
                            "predicted": float(predicted),
                            "absolute_error": abs(float(predicted) - float(actual[index])),
                            "nearest_train_similarity": 0.2 + 0.6 * (index / len(smiles)),
                        }
                    )
    return pd.DataFrame(rows, columns=list(PREDICTION_COLUMNS))


def make_importance_frame(*, molecules=240, signal=True, seed=0):
    """Build a minimal profile-shaped frame for the multivariate importance tests.

    ``error_property_importance`` only reads the property, control, and target columns,
    so a synthetic frame lets these tests use enough molecules for a stable ranking
    without inventing hundreds of SMILES strings.
    """

    generator = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {name: generator.normal(size=molecules) for name in PROPERTY_NAMES}
    )
    frame["regime"] = "out_of_scaffold"
    frame["model"] = "GIN"
    frame["baseline_absolute_error"] = generator.normal(size=molecules)
    frame["nearest_train_similarity"] = generator.normal(size=molecules)
    frame["absolute_error"] = (
        3.0 * frame["MolWt"] + generator.normal(scale=0.2, size=molecules)
        if signal
        else generator.normal(size=molecules)
    )
    return frame


@pytest.fixture(scope="module")
def predictions():
    return make_predictions()


@pytest.fixture(scope="module")
def profile(predictions):
    return build_error_profile(predictions)


def test_molecular_properties_returns_the_documented_columns():
    properties = molecular_properties(make_smiles())

    assert list(properties.columns) == list(PROPERTY_NAMES)
    assert len(properties) == len(make_smiles())
    assert properties.notna().all().all()


def test_molecular_properties_reuses_week1_descriptors():
    smiles = make_smiles()

    shared = molecular_properties(smiles)[list(DESCRIPTOR_NAMES)]

    pd.testing.assert_frame_equal(shared, calculate_descriptors(smiles))


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CCO", {"NumRings": 0.0, "NumAromaticRings": 0.0, "FractionCSP3": 1.0,
                 "HeavyAtomCount": 3.0, "NumHDonors": 1.0}),
        ("c1ccccc1", {"NumRings": 1.0, "NumAromaticRings": 1.0, "NumAliphaticRings": 0.0,
                      "AromaticAtomFraction": 1.0, "HeavyAtomCount": 6.0}),
        ("C1CCCCC1", {"NumRings": 1.0, "NumAromaticRings": 0.0, "NumAliphaticRings": 1.0,
                      "FractionCSP3": 1.0}),
        ("c1ccc2ccccc2c1", {"NumRings": 2.0, "NumAromaticRings": 2.0,
                            "HeavyAtomCount": 10.0}),
    ],
)
def test_molecular_properties_matches_hand_checked_values(smiles, expected):
    row = molecular_properties([smiles]).iloc[0]

    for name, value in expected.items():
        assert row[name] == pytest.approx(value)


@pytest.mark.parametrize("smiles", ["not-a-molecule", "", None])
def test_molecular_properties_rejects_invalid_smiles(smiles):
    with pytest.raises(ValueError, match="Invalid SMILES at position"):
        molecular_properties(["CCO", smiles])


def test_molecular_properties_accepts_an_empty_list():
    properties = molecular_properties([])

    assert list(properties.columns) == list(PROPERTY_NAMES)
    assert properties.empty


def test_property_table_computes_one_row_per_molecule(predictions):
    properties = property_table(predictions)

    assert len(properties) == predictions["dataset_index"].nunique()
    assert not properties["dataset_index"].duplicated().any()
    assert properties[list(PROPERTY_NAMES)].notna().all().all()


def test_property_reference_table_reports_ordered_quartiles(predictions):
    properties = property_table(predictions)

    reference = property_reference_table(properties).set_index("property")

    assert len(reference) == len(PROPERTY_NAMES)
    for name in PROPERTY_NAMES:
        row = reference.loc[name]
        assert row["minimum"] <= row["q25"] <= row["median"] <= row["q75"] <= row["maximum"]
        assert row["median"] == pytest.approx(properties[name].median())


def test_property_correlation_matrix_is_symmetric_with_unit_diagonal(predictions):
    matrix = property_correlation_matrix(property_table(predictions)).set_index("property")

    values = matrix.to_numpy(dtype=float)
    assert np.allclose(np.diag(values), 1.0)
    assert np.allclose(values, values.T, equal_nan=True)
    assert np.nanmax(np.abs(values)) <= 1.0 + 1e-12


def test_property_collinearity_finds_the_size_cluster(predictions):
    pairs = property_collinearity(property_table(predictions), threshold=0.6)

    found = {frozenset((row.property_a, row.property_b)) for row in pairs.itertuples()}
    assert frozenset(("MolWt", "HeavyAtomCount")) in found
    assert (pairs["spearman"].abs() > 0.6).all()


def test_property_collinearity_tolerates_a_constant_property(predictions):
    properties = property_table(predictions).assign(NumAliphaticRings=1.0)

    pairs = property_collinearity(properties)

    assert "NumAliphaticRings" not in set(pairs["property_a"]) | set(pairs["property_b"])


def test_rank_correlation_is_one_for_a_monotone_nonlinear_pair():
    values = np.arange(1.0, 11.0)

    assert rank_correlation(values, values**3) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (np.ones(5), np.arange(5.0)),
        (np.array([1.0]), np.array([2.0])),
        (np.array([1.0, 2.0]), np.array([2.0, 4.0])),
    ],
)
def test_rank_correlation_returns_nan_for_degenerate_input(left, right):
    assert np.isnan(rank_correlation(left, right))


def test_rank_correlation_ignores_non_finite_pairs():
    left = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    right = np.array([1.0, 2.0, 3.0, 4.0, 100.0])

    assert rank_correlation(left, right) == pytest.approx(1.0)


def test_partial_correlation_without_controls_matches_rank_correlation():
    generator = np.random.default_rng(1)
    target = generator.normal(size=40)
    feature = generator.normal(size=40)

    assert partial_correlation(target, feature, None) == pytest.approx(
        rank_correlation(target, feature)
    )
    assert partial_correlation(target, feature, np.ones((40, 1))) == pytest.approx(
        rank_correlation(target, feature)
    )


def test_partial_correlation_removes_a_shared_driver():
    generator = np.random.default_rng(2)
    control = generator.normal(size=200)
    # Both variables are functions of the control alone, so nothing should survive.
    target = control * 2.0 + generator.normal(scale=0.05, size=200)
    feature = control * -1.5 + generator.normal(scale=0.05, size=200)

    raw = rank_correlation(target, feature)
    partial = partial_correlation(target, feature, control.reshape(-1, 1))

    assert abs(raw) > 0.9
    assert abs(partial) < 0.3


def test_partial_correlation_keeps_an_independent_relationship():
    generator = np.random.default_rng(3)
    control = generator.normal(size=200)
    feature = generator.normal(size=200)
    target = feature * 2.0 + control * 3.0

    assert partial_correlation(target, feature, control.reshape(-1, 1)) > 0.9


def test_partial_correlation_returns_nan_for_too_few_rows():
    assert np.isnan(
        partial_correlation(np.array([1.0, 2.0]), np.array([2.0, 1.0]), np.ones((2, 1)))
    )


def test_bootstrap_correlation_interval_is_reproducible():
    generator = np.random.default_rng(4)
    left = generator.normal(size=80)
    right = left + generator.normal(scale=0.5, size=80)

    first = bootstrap_correlation_interval(left, right, samples=200, seed=7)
    second = bootstrap_correlation_interval(left, right, samples=200, seed=7)

    assert first == second


def test_bootstrap_correlation_interval_brackets_the_point_estimate():
    generator = np.random.default_rng(5)
    left = generator.normal(size=120)
    right = left * 2.0 + generator.normal(scale=0.3, size=120)

    point, low, high = bootstrap_correlation_interval(left, right, samples=300, seed=1)

    assert low <= point <= high
    assert low > 0


def test_bootstrap_correlation_interval_is_nan_for_a_constant_column():
    point, low, high = bootstrap_correlation_interval(
        np.ones(20), np.arange(20.0), samples=50
    )

    assert np.isnan(point) and np.isnan(low) and np.isnan(high)


def test_pool_split_types_merges_the_two_scaffold_regimes():
    predictions = make_predictions(split_types=("random", "scaffold", "scaffold_shuffled"))

    pooled = pool_split_types(predictions)

    assert len(pooled) == len(predictions)
    assert set(pooled["split_type"]) == set(predictions["split_type"])
    assert set(pooled[pooled["regime"] == "out_of_scaffold"]["split_type"]) == {
        "scaffold",
        "scaffold_shuffled",
    }
    assert set(pooled[pooled["regime"] == "in_domain"]["split_type"]) == {"random"}


def test_pool_split_types_passes_unmapped_names_through(predictions):
    renamed = predictions.assign(split_type="custom_split")

    pooled = pool_split_types(renamed)

    assert set(pooled["regime"]) == {"custom_split"}


def test_effective_sample_sizes_separates_full_and_ragged_seed_coverage(predictions):
    sizes = effective_sample_sizes(predictions).set_index("regime")

    scaffold = sizes.loc["out_of_scaffold"]
    assert scaffold["distinct_molecules"] == scaffold["molecules_in_every_seed"]
    assert scaffold["mean_seeds_per_molecule"] == pytest.approx(scaffold["seeds"])

    in_domain = sizes.loc["in_domain"]
    assert in_domain["molecules_in_every_seed"] < in_domain["distinct_molecules"]
    assert in_domain["mean_seeds_per_molecule"] < in_domain["seeds"]


def test_build_error_profile_returns_one_row_per_molecule_and_model(profile, predictions):
    expected = (
        pool_split_types(predictions)
        .groupby(["regime", "model"])["dataset_index"]
        .nunique()
        .sum()
    )

    assert list(profile.columns) == list(PROFILE_COLUMNS)
    assert len(profile) == expected


def test_build_error_profile_averages_over_seeds(profile, predictions):
    pooled = pool_split_types(predictions)
    row = profile[
        (profile["regime"] == "out_of_scaffold")
        & (profile["model"] == "GIN")
        & (profile["dataset_index"] == 4)
    ].iloc[0]
    source = pooled[
        (pooled["regime"] == "out_of_scaffold")
        & (pooled["model"] == "GIN")
        & (pooled["dataset_index"] == 4)
    ]

    assert row["absolute_error"] == pytest.approx(source["absolute_error"].mean())
    assert row["seeds"] == source["seed"].nunique()


def test_build_error_profile_uses_the_documented_residual_sign():
    frame = make_predictions(seeds=(0,), split_types=("scaffold",))
    frame.loc[frame.index, "actual"] = -4.0
    frame.loc[frame.index, "predicted"] = -1.7
    frame.loc[frame.index, "absolute_error"] = 2.3

    profile = build_error_profile(frame)

    assert profile["residual_mean"].unique() == pytest.approx([2.3])


def test_build_error_profile_absolute_error_dominates_the_signed_mean(profile):
    assert (profile["absolute_error"] >= profile["residual_mean"].abs() - 1e-9).all()


def test_build_error_profile_derives_the_baseline_error_from_the_dummy_rows(profile):
    baseline = profile[profile["model"] == BASELINE_MODEL]

    assert baseline["absolute_error"].to_numpy() == pytest.approx(
        baseline["baseline_absolute_error"].to_numpy()
    )
    defined = baseline[baseline["baseline_absolute_error"] >= MIN_BASELINE_ERROR]
    assert defined["normalized_error"].to_numpy() == pytest.approx(1.0)


def test_build_error_profile_leaves_the_ratio_undefined_near_the_training_mean(profile):
    below = profile["baseline_absolute_error"] < MIN_BASELINE_ERROR

    assert profile.loc[below, "normalized_error"].isna().all()
    assert profile.loc[~below, "normalized_error"].notna().all()


def test_build_error_profile_keeps_a_single_seed_group_without_a_deviation():
    frame = make_predictions(seeds=(0,), split_types=("scaffold",))

    profile = build_error_profile(frame)

    assert (profile["seeds"] == 1).all()
    assert profile["absolute_error_std"].isna().all()


def test_build_error_profile_requires_the_prediction_schema(predictions):
    with pytest.raises(ValueError, match="missing required columns"):
        build_error_profile(predictions.drop(columns="nearest_train_similarity"))


def test_build_error_profile_requires_a_baseline_model(predictions):
    without = predictions[predictions["model"] != BASELINE_MODEL]

    with pytest.raises(ValueError, match="intrinsic-difficulty control"):
        build_error_profile(without)


def test_property_error_correlation_recovers_the_planted_relationship(profile):
    correlations = property_error_correlation(profile, samples=200)

    panel = correlations[
        (correlations["regime"] == "out_of_scaffold")
        & (correlations["property"] == "MolWt")
    ].set_index("model")

    assert list(correlations.columns) == list(CORRELATION_COLUMNS)
    # GIN's error is planted to grow with MolWt while MLP's is not. Twenty molecules are
    # too few for the bootstrap interval to clear zero, which is the honest answer; the
    # planted contrast between the two models is what this fixture can establish.
    assert panel.loc["GIN", "spearman"] > 0.25
    assert abs(panel.loc["MLP", "spearman"]) < panel.loc["GIN", "spearman"]


def test_property_error_correlation_brackets_every_estimate(profile):
    correlations = property_error_correlation(profile, samples=200)

    finite = correlations.dropna(subset=["spearman", "ci_low", "ci_high"])
    assert (finite["ci_low"] <= finite["spearman"] + 1e-9).all()
    assert (finite["spearman"] <= finite["ci_high"] + 1e-9).all()
    signs = finite["ci_low"] * finite["ci_high"] > 0
    assert (finite["excludes_zero"] == signs).all()


def test_property_error_correlation_flags_the_trained_on_descriptors(profile):
    correlations = property_error_correlation(profile, samples=50)

    flagged = set(correlations[correlations["trained_on"]]["property"])
    assert flagged == set(DESCRIPTOR_NAMES)


def test_property_error_correlation_baseline_column_does_not_vary_by_model(profile):
    correlations = property_error_correlation(profile, samples=50)

    spread = correlations.groupby(["regime", "property"])["baseline_spearman"].nunique()
    assert (spread == 1).all()


def test_property_error_correlation_handles_a_constant_error_column(profile):
    flat = profile.assign(
        absolute_error=np.where(profile["model"] == "GIN", 1.0, profile["absolute_error"])
    )

    correlations = property_error_correlation(flat, samples=50)

    panel = correlations[correlations["model"] == "GIN"]
    assert panel["spearman"].isna().all()
    assert (panel["molecules"] > 0).all()


def test_property_error_correlation_drops_missing_properties_pairwise(profile):
    damaged = profile.copy()
    damaged.loc[damaged.index[0], "MolWt"] = np.nan

    correlations = property_error_correlation(damaged, samples=50)

    row = correlations[
        (correlations["regime"] == damaged.iloc[0]["regime"])
        & (correlations["model"] == damaged.iloc[0]["model"])
        & (correlations["property"] == "MolWt")
    ].iloc[0]
    reference = correlations[
        (correlations["regime"] == damaged.iloc[0]["regime"])
        & (correlations["model"] == damaged.iloc[0]["model"])
        & (correlations["property"] == "TPSA")
    ].iloc[0]
    assert row["molecules"] == reference["molecules"] - 1


def test_bin_property_errors_fills_every_quantile_bin(profile):
    bins = bin_property_errors(profile, "MolWt")

    counts = bins.groupby(["regime", "model"])["molecules"].sum()
    expected = profile.groupby(["regime", "model"]).size()
    assert (counts == expected.reindex(counts.index)).all()
    per_bin = bins.groupby(["regime", "model"])["molecules"].agg(["min", "max"])
    assert ((per_bin["max"] - per_bin["min"]) <= 2).all()


def test_bin_property_errors_normalises_by_a_ratio_of_means(profile):
    bins = bin_property_errors(profile, "MolWt")

    row = bins.iloc[0]
    assert row["normalized_mean_absolute_error"] == pytest.approx(
        row["mean_absolute_error"] / row["baseline_mean_absolute_error"]
    )


def test_bin_property_errors_recovers_the_planted_size_trend(profile):
    bins = bin_property_errors(profile, "MolWt")

    panel = bins[(bins["regime"] == "out_of_scaffold") & (bins["model"] == "GIN")]
    ordered = panel.sort_values("bin_index")["mean_absolute_error"].to_numpy()
    assert ordered[-1] > ordered[0]


def test_bin_property_errors_collapses_a_constant_property(profile):
    constant = profile.assign(MolWt=100.0)

    bins = bin_property_errors(constant, "MolWt")

    assert (bins.groupby(["regime", "model"])["bin_index"].nunique() == 1).all()


def test_bin_property_errors_rejects_an_unknown_property(profile):
    with pytest.raises(ValueError, match="property_name must be one of"):
        bin_property_errors(profile, "NotAProperty")


def test_shrinkage_fit_recovers_the_planted_slope(profile):
    shrinkage = shrinkage_fit(profile).set_index(["regime", "model"])

    assert list(shrinkage.reset_index().columns) == list(SHRINKAGE_COLUMNS)
    row = shrinkage.loc[("out_of_scaffold", "GCN")]
    assert row["slope"] == pytest.approx(0.5, abs=0.1)
    assert row["prediction_std_ratio"] < 1.0


def test_shrinkage_fit_reports_a_perfect_model_as_unbiased():
    frame = make_predictions(seeds=(0,), split_types=("scaffold",))
    frame["predicted"] = frame["actual"]
    frame["absolute_error"] = 0.0

    shrinkage = shrinkage_fit(build_error_profile(frame, min_baseline_error=0.0))

    graph = shrinkage[shrinkage["model"] == "GIN"].iloc[0]
    assert graph["slope"] == pytest.approx(1.0)
    assert graph["mean_residual"] == pytest.approx(0.0)


def test_shrinkage_fit_reports_the_baseline_as_fully_shrunk(profile):
    row = shrinkage_fit(profile).set_index(["regime", "model"]).loc[
        ("out_of_scaffold", BASELINE_MODEL)
    ]

    assert row["slope"] == pytest.approx(0.0, abs=1e-6)
    assert row["low_solubility_bias"] > 0 > row["high_solubility_bias"]


def test_shrinkage_fit_returns_nan_for_a_constant_target():
    frame = make_predictions(seeds=(0,), split_types=("scaffold",))
    frame["actual"] = -3.0
    frame["absolute_error"] = (frame["predicted"] - frame["actual"]).abs()

    shrinkage = shrinkage_fit(build_error_profile(frame, min_baseline_error=0.0))

    assert shrinkage["slope"].isna().all()


def test_model_error_agreement_reports_identical_models_as_perfectly_agreeing(profile):
    duplicated = pd.concat(
        [profile, profile[profile["model"] == "GIN"].assign(model="GIN copy")],
        ignore_index=True,
    )

    agreement = model_error_agreement(
        duplicated, models=("GIN", "GIN copy"), samples=50
    )

    row = agreement.iloc[0]
    assert row["spearman"] == pytest.approx(1.0)
    assert row["shared_worst_fraction"] == pytest.approx(1.0)


def test_model_error_agreement_separates_families(profile):
    agreement = model_error_agreement(profile, samples=100)

    panel = agreement[agreement["regime"] == "out_of_scaffold"].set_index(
        ["model_a", "model_b"]
    )
    assert panel.loc[("GCN", "GIN"), "same_family"]
    assert not panel.loc[("MLP", "GIN"), "same_family"]
    assert panel.loc[("GCN", "GIN"), "spearman"] > panel.loc[("MLP", "GIN"), "spearman"]


def test_model_error_agreement_emits_each_pair_once(profile):
    agreement = model_error_agreement(profile, samples=50)

    pairs = [
        frozenset((row.model_a, row.model_b))
        for row in agreement[agreement["regime"] == "in_domain"].itertuples()
    ]
    assert len(pairs) == len(set(pairs))


def test_model_error_agreement_rejects_an_unknown_model(profile):
    with pytest.raises(ValueError, match="has no rows for"):
        model_error_agreement(profile, models=("GIN", "Transformer"))


def test_model_error_disagreement_names_the_planted_blow_ups(profile):
    disagreement = model_error_disagreement(profile, top_n=3)

    favours_mlp = disagreement[disagreement["favours"] == "MLP"].sort_values("rank")
    favours_gin = disagreement[disagreement["favours"] == "GIN"].sort_values("rank")
    assert int(favours_mlp.iloc[0]["dataset_index"]) == 0
    assert int(favours_gin.iloc[0]["dataset_index"]) == 1
    assert (favours_mlp["error_difference"].diff().dropna() <= 1e-9).all()


def test_model_error_disagreement_rejects_a_missing_model(profile):
    with pytest.raises(ValueError, match="has no .* rows in regime"):
        model_error_disagreement(profile, models=("GIN", "Transformer"))


def test_scaffold_error_table_honours_the_minimum_and_finds_the_bad_scaffold(profile):
    scaffolds = scaffold_error_table(profile, minimum_molecules=3)

    assert (scaffolds["molecules"] >= 3).all()
    panel = scaffolds[
        (scaffolds["regime"] == "out_of_scaffold") & (scaffolds["model"] == "GIN")
    ]
    assert panel.iloc[0]["scaffold"] == "c1ccc2ccccc2c1"


def test_scaffold_error_table_keeps_the_acyclic_scaffold(profile):
    scaffolds = scaffold_error_table(profile, minimum_molecules=3)

    assert "" in set(scaffolds["scaffold"])


def test_scaffold_error_table_marks_unseen_scaffolds_as_missing(profile):
    scaffolds = scaffold_error_table(
        profile, scaffold_counts={"c1ccccc1": 254}, minimum_molecules=3
    )

    known = scaffolds[scaffolds["scaffold"] == "c1ccccc1"]
    unknown = scaffolds[scaffolds["scaffold"] == "c1ccc2ccccc2c1"]
    assert (known["dataset_molecules"] == 254).all()
    assert unknown["dataset_molecules"].isna().all()


def test_error_property_importance_reports_the_gate_columns(profile):
    importance = error_property_importance(
        profile, folds=3, repeats=3, minimum_molecules=10
    )

    assert list(importance.columns) == list(IMPORTANCE_COLUMNS)
    per_group = importance.groupby(["regime", "model"])["feature"].nunique()
    assert (per_group == len(PROPERTY_NAMES) + 2).all()
    assert importance["cross_validated_r2"].notna().all()
    assert set(importance[importance["is_control"]]["feature"]) == {
        "baseline_absolute_error",
        "nearest_train_similarity",
    }


def test_error_property_importance_marks_a_signal_free_fit_as_untrustworthy():
    noise = make_importance_frame(signal=False)

    importance = error_property_importance(noise, repeats=5)

    assert not importance["trustworthy"].any()
    assert (importance["cross_validated_r2"] <= 0).all()


def test_error_property_importance_ranks_the_planted_property_first():
    frame = make_importance_frame(signal=True)

    importance = error_property_importance(frame, repeats=5, control_features=())

    ranked = importance.sort_values("importance_mean", ascending=False)
    assert ranked.iloc[0]["feature"] == "MolWt"
    assert ranked.iloc[0]["trustworthy"]
    assert not ranked["controls_included"].any()


def test_error_property_importance_is_reproducible(profile):
    first = error_property_importance(profile, folds=3, repeats=3, minimum_molecules=10)
    second = error_property_importance(profile, folds=3, repeats=3, minimum_molecules=10)

    pd.testing.assert_frame_equal(first, second)


def test_error_property_importance_refuses_a_tiny_group(profile):
    with pytest.raises(ValueError, match="fewer than the required"):
        error_property_importance(profile, folds=3, minimum_molecules=10_000)


def test_case_studies_orders_worst_and_best(profile):
    cases = case_studies(profile, top_n=3)

    assert list(cases.columns) == list(CASE_STUDY_COLUMNS)
    worst = cases[cases["case"] == "worst"]["absolute_error"].to_numpy()
    best = cases[cases["case"] == "best"]["absolute_error"].to_numpy()
    assert (np.diff(worst) <= 1e-9).all()
    assert (np.diff(best) >= -1e-9).all()
    assert worst[0] > best[0]


def test_case_studies_percentiles_stay_within_range(profile):
    cases = case_studies(profile, top_n=3)

    for name in PROPERTY_NAMES:
        column = cases[f"{name}_percentile"]
        assert ((column >= 0.0) & (column <= 100.0)).all()


def test_case_studies_excludes_molecules_sitting_on_the_training_mean(profile):
    cases = case_studies(profile, top_n=5)

    best = cases[cases["case"] == "best"]
    assert (best["baseline_absolute_error"] >= MIN_BASELINE_ERROR).all()


def test_case_studies_returns_what_exists_when_top_n_is_large(profile):
    cases = case_studies(profile, top_n=500)

    available = profile[
        (profile["regime"] == "out_of_scaffold") & (profile["model"] == "GIN")
    ]
    assert len(cases[cases["case"] == "worst"]) == len(available)


def test_case_studies_rejects_an_empty_selection(profile):
    with pytest.raises(ValueError, match="has no rows for regime"):
        case_studies(profile, model="Transformer")


def test_analysis_config_validates_its_fields():
    with pytest.raises(ValueError, match="unknown properties"):
        AnalysisConfig(properties=("MolWt", "NotAProperty"))
    with pytest.raises(ValueError, match="unknown models"):
        AnalysisConfig(models=("GIN", "Transformer"))
    with pytest.raises(ValueError, match="exactly two models"):
        AnalysisConfig(diagnostic_models=("GIN",))
    with pytest.raises(ValueError, match="must not include"):
        AnalysisConfig(diagnostic_models=(BASELINE_MODEL, "GIN"))
    with pytest.raises(ValueError, match="confidence must lie"):
        AnalysisConfig(confidence=1.5)
    with pytest.raises(ValueError, match="property_quantiles must increase"):
        AnalysisConfig(property_quantiles=(0.0, 0.5))
    with pytest.raises(ValueError, match="must be unique"):
        AnalysisConfig(properties=("MolWt", "MolWt"))


def test_analysis_config_exposes_the_regime_map_as_a_dictionary():
    assert AnalysisConfig().regime_mapping["scaffold_shuffled"] == "out_of_scaffold"


@pytest.fixture(scope="module")
def analysis(predictions):
    return run_error_analysis(
        predictions,
        smiles_list=make_smiles(),
        config=AnalysisConfig(
            bootstrap_samples=50,
            importance_folds=3,
            importance_repeats=3,
            importance_minimum_molecules=10,
            minimum_scaffold_molecules=3,
            case_count=3,
        ),
        verbose=False,
    )


def test_run_error_analysis_fits_the_importance_model_twice(analysis):
    assert set(analysis.importance["controls_included"]) == {True, False}


def test_run_error_analysis_attaches_dataset_scaffold_counts(analysis):
    assert analysis.scaffold_errors["dataset_molecules"].notna().any()


def test_save_and_load_error_analysis_round_trips(analysis, tmp_path):
    paths = save_error_analysis(analysis, tmp_path)

    assert all(path.exists() for path in paths.values())
    loaded = load_error_analysis(tmp_path)

    assert loaded.config == analysis.config
    for name in ("profile", "correlations", "shrinkage", "agreement", "importance"):
        pd.testing.assert_frame_equal(
            getattr(loaded, name),
            getattr(analysis, name),
            check_exact=False,
            atol=1e-4,
            check_dtype=False,
        )


def test_load_error_analysis_preserves_the_acyclic_scaffold(analysis, tmp_path):
    save_error_analysis(analysis, tmp_path)

    loaded = load_error_analysis(tmp_path)

    assert "" in set(loaded.profile["scaffold"])
    assert loaded.profile["scaffold"].notna().all()


def test_load_or_run_error_analysis_reuses_a_complete_cache(analysis, tmp_path):
    save_error_analysis(analysis, tmp_path)

    # Passing no predictions proves nothing was recomputed.
    reused = load_or_run_error_analysis(directory=tmp_path, config=analysis.config)

    assert len(reused.profile) == len(analysis.profile)


def test_load_or_run_error_analysis_recomputes_after_a_config_change(
    analysis, predictions, tmp_path
):
    save_error_analysis(analysis, tmp_path)

    with pytest.warns(UserWarning, match="different configuration"):
        result = load_or_run_error_analysis(
            predictions,
            smiles_list=make_smiles(),
            directory=tmp_path,
            config=AnalysisConfig(
                bootstrap_samples=25,
                importance_folds=3,
                importance_repeats=3,
                importance_minimum_molecules=10,
                minimum_scaffold_molecules=3,
                case_count=2,
            ),
            verbose=False,
        )

    assert result.config.bootstrap_samples == 25


def test_load_or_run_error_analysis_recomputes_when_a_file_is_missing(
    analysis, predictions, tmp_path
):
    paths = save_error_analysis(analysis, tmp_path)
    paths["profile"].unlink()

    result = load_or_run_error_analysis(
        predictions,
        smiles_list=make_smiles(),
        directory=tmp_path,
        config=analysis.config,
        verbose=False,
    )

    assert paths["profile"].exists()
    assert len(result.profile) == len(analysis.profile)


def test_load_or_run_error_analysis_requires_predictions_on_a_cache_miss(tmp_path):
    with pytest.raises(ValueError, match="predictions are required"):
        load_or_run_error_analysis(directory=tmp_path)


def test_summarise_reports_both_importance_fits(analysis):
    summary = summarise(analysis, prediction_rows=100)

    headline = summary["headline"]
    assert "cross_validated_r2_with_controls" in headline
    assert "cross_validated_r2_properties_only" in headline
    assert summary["source"]["prediction_rows"] == 100
    assert "torch" not in summary["environment"]
    assert set(summary["environment"]) >= {"python", "numpy", "pandas", "rdkit"}


@pytest.mark.parametrize(
    "builder",
    [
        lambda analysis: plot_property_collinearity(analysis.properties),
        lambda analysis: plot_property_error_correlation(analysis.correlations),
        lambda analysis: plot_property_error_bins(analysis.property_bins),
        lambda analysis: plot_error_property_importance(analysis.importance),
        lambda analysis: plot_model_error_agreement(analysis.profile),
        lambda analysis: plot_residual_shrinkage(analysis.profile),
        lambda analysis: plot_scaffold_errors(analysis.scaffold_errors),
        lambda analysis: plot_case_study_structures(analysis.case_studies),
    ],
)
def test_plot_helpers_return_a_populated_figure(analysis, builder):
    figure = builder(analysis)
    try:
        assert figure.get_axes()
        assert any(axis.has_data() for axis in figure.get_axes())
    finally:
        plt.close(figure)


def test_plot_error_property_importance_shows_the_gate_in_the_title(analysis):
    figure = plot_error_property_importance(analysis.importance)
    try:
        titles = [axis.get_title() for axis in figure.get_axes()]
        assert all("held-out R2" in title for title in titles)
        assert all(("readable" in title) for title in titles)
    finally:
        plt.close(figure)


def test_plot_case_study_structures_handles_a_single_pair(analysis):
    cases = pd.concat(
        [
            analysis.case_studies[analysis.case_studies["case"] == case].head(1)
            for case in ("worst", "best")
        ],
        ignore_index=True,
    )

    figure = plot_case_study_structures(cases, top_n=4)
    try:
        assert figure.get_axes()
    finally:
        plt.close(figure)


def test_build_figures_covers_every_saved_name(analysis):
    figures = build_figures(analysis)
    try:
        assert set(figures) == {
            "property_collinearity",
            "property_error_correlation",
            "property_error_bins_in_domain",
            "property_error_bins_out_of_scaffold",
            "property_importance",
            "model_error_agreement",
            "residual_shrinkage",
            "scaffold_errors",
            "case_study_structures",
        }
    finally:
        for figure in figures.values():
            plt.close(figure)


@pytest.mark.integration
def test_molecular_properties_runs_on_real_esol_molecules():
    from ai4sci_molecule.week1 import load_esol

    dataset = load_esol()
    smiles = [str(graph.smiles) for graph in dataset[:20]]

    properties = molecular_properties(smiles)

    assert len(properties) == 20
    assert properties.notna().all().all()
    assert (properties["MolWt"] > 0).all()
