"""Per-molecule error diagnostics for week 4.

Week 3 answered *when* the models fail: below a nearest-train Tanimoto similarity of
roughly 0.4 the graph models collapse to mean-prediction accuracy. This module answers
*which molecules* fail and whether the failure is chemical or representational, using
week 3's cached per-molecule predictions as its only evidence base. Nothing here trains
a model, so the whole analysis is CPU-deterministic given ``results/week3``.
"""

from __future__ import annotations

import json
import sys
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.figure import Figure
from rdkit import Chem, rdBase
from rdkit.Chem import Draw, Lipinski, rdMolDescriptors
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_predict

from .week1 import DESCRIPTOR_NAMES, _parse_smiles, calculate_descriptors
from .week3 import (
    MODEL_ORDER,
    PREDICTION_COLUMNS,
    _order_by_model,
    _read_csv,
    _shorten,
    save_figures,
    scaffold_table,
)


BASELINE_MODEL = "Dummy (mean)"
DEFAULT_RESULTS_DIRECTORY = Path("results/week4")
WEEK3_RESULTS_DIRECTORY = Path("results/week3")

# The five week-1 descriptors come first so ``calculate_descriptors`` can supply them
# verbatim; the eight that follow are the properties week 4 adds.
PROPERTY_NAMES = (
    *DESCRIPTOR_NAMES,
    "HeavyAtomCount",
    "NumRings",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumHeteroatoms",
    "FractionCSP3",
    "NumHDonors",
    "NumHAcceptors",
)
EXTENDED_PROPERTY_NAMES = tuple(
    name for name in PROPERTY_NAMES if name not in DESCRIPTOR_NAMES
)
HEADLINE_PROPERTIES = ("MolWt", "MolLogP", "NumAromaticRings", "FractionCSP3")

# Pooling the two scaffold regimes lifts the out-of-scaffold sample from 113 to a few
# hundred molecules. ``random`` is deliberately kept apart: it is the leaky regime, and
# mixing it in would blend in-domain and out-of-domain evidence.
IN_DOMAIN_REGIME = "in_domain"
OUT_OF_SCAFFOLD_REGIME = "out_of_scaffold"
DEFAULT_REGIME_MAP = {
    "random": IN_DOMAIN_REGIME,
    "scaffold": OUT_OF_SCAFFOLD_REGIME,
    "scaffold_shuffled": OUT_OF_SCAFFOLD_REGIME,
}
REGIME_ORDER = (IN_DOMAIN_REGIME, OUT_OF_SCAFFOLD_REGIME)

# Representation families, used to test whether models that share a representation fail
# on the same molecules. The baseline belongs to neither.
MODEL_FAMILIES = {
    "Linear": "descriptor",
    "MLP": "descriptor",
    "GCN": "graph",
    "GIN": "graph",
}
DIAGNOSTIC_MODELS = ("MLP", "GIN")

# Below this the molecule sits essentially on the training mean, so dividing by the
# baseline error would explode. Such rows get a NaN ratio instead.
MIN_BASELINE_ERROR = 0.25
CONTROL_FEATURES = ("baseline_absolute_error", "nearest_train_similarity")
DEFAULT_PROPERTY_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_COLLINEARITY_THRESHOLD = 0.6
MINIMUM_SCAFFOLD_MOLECULES = 3
DEFAULT_WORST_FRACTION = 0.1
DEFAULT_CASE_COUNT = 5
DEFAULT_IMPORTANCE_FOLDS = 5
DEFAULT_IMPORTANCE_REPEATS = 20
MINIMUM_IMPORTANCE_MOLECULES = 60

PROFILE_COLUMNS = (
    "regime",
    "split_type",
    "model",
    "representation",
    "family",
    "dataset_index",
    "smiles",
    "scaffold",
    "seeds",
    "actual",
    "predicted_mean",
    "residual_mean",
    "absolute_error",
    "absolute_error_std",
    "baseline_absolute_error",
    "normalized_error",
    "nearest_train_similarity",
    *PROPERTY_NAMES,
)
SAMPLE_SIZE_COLUMNS = (
    "regime",
    "split_types",
    "seeds",
    "rows_per_model",
    "distinct_molecules",
    "molecules_in_every_seed",
    "mean_seeds_per_molecule",
    "distinct_scaffolds",
    "scaffolds_over_minimum",
)
CORRELATION_COLUMNS = (
    "regime",
    "model",
    "property",
    "molecules",
    "spearman",
    "ci_low",
    "ci_high",
    "excludes_zero",
    "baseline_spearman",
    "partial_spearman",
    "trained_on",
)
PROPERTY_BIN_COLUMNS = (
    "regime",
    "model",
    "property",
    "property_bin",
    "bin_index",
    "molecules",
    "property_median",
    "actual_median",
    "mean_absolute_error",
    "baseline_mean_absolute_error",
    "normalized_mean_absolute_error",
    "mean_residual",
)
IMPORTANCE_COLUMNS = (
    "regime",
    "model",
    "controls_included",
    "feature",
    "is_control",
    "importance_mean",
    "importance_std",
    "cross_validated_r2",
    "out_of_fold_spearman",
    "trustworthy",
    "molecules",
)
AGREEMENT_COLUMNS = (
    "regime",
    "model_a",
    "model_b",
    "same_family",
    "molecules",
    "spearman",
    "ci_low",
    "ci_high",
    "shared_worst_fraction",
    "residual_spearman",
)
DISAGREEMENT_COLUMNS = (
    "regime",
    "favours",
    "rank",
    "dataset_index",
    "smiles",
    "scaffold",
    "actual",
    "absolute_error_a",
    "absolute_error_b",
    "error_difference",
    "baseline_absolute_error",
    "nearest_train_similarity",
    *PROPERTY_NAMES,
)
SHRINKAGE_COLUMNS = (
    "regime",
    "model",
    "molecules",
    "slope",
    "intercept",
    "prediction_std_ratio",
    "mean_residual",
    "residual_actual_spearman",
    "low_solubility_bias",
    "high_solubility_bias",
)
SCAFFOLD_ERROR_COLUMNS = (
    "regime",
    "model",
    "scaffold",
    "molecules",
    "dataset_molecules",
    "mean_absolute_error",
    "baseline_mean_absolute_error",
    "normalized_mean_absolute_error",
    "mean_residual",
)
CASE_STUDY_COLUMNS = (
    "regime",
    "model",
    "case",
    "rank",
    "dataset_index",
    "smiles",
    "scaffold",
    "actual",
    "predicted_mean",
    "absolute_error",
    "residual_mean",
    "baseline_absolute_error",
    "nearest_train_similarity",
    *PROPERTY_NAMES,
    *(f"{name}_percentile" for name in PROPERTY_NAMES),
)
COLLINEARITY_COLUMNS = ("property_a", "property_b", "spearman")
REFERENCE_COLUMNS = ("property", "median", "q25", "q75", "minimum", "maximum")

# name -> (filename, numeric columns to coerce on load). Single source of truth for the
# save/load round trip, so adding a table does not mean editing three functions.
ANALYSIS_TABLES: dict[str, tuple[str, tuple[str, ...]]] = {
    "profile": (
        "molecule_errors.csv",
        (
            "seeds",
            "dataset_index",
            "actual",
            "predicted_mean",
            "residual_mean",
            "absolute_error",
            "absolute_error_std",
            "baseline_absolute_error",
            "normalized_error",
            "nearest_train_similarity",
            *PROPERTY_NAMES,
        ),
    ),
    "properties": ("properties.csv", ("dataset_index", *PROPERTY_NAMES)),
    "sample_sizes": (
        "effective_sample_sizes.csv",
        (
            "seeds",
            "rows_per_model",
            "distinct_molecules",
            "molecules_in_every_seed",
            "mean_seeds_per_molecule",
            "distinct_scaffolds",
            "scaffolds_over_minimum",
        ),
    ),
    "collinearity": ("property_collinearity.csv", ("spearman",)),
    "reference": ("property_reference.csv", ("median", "q25", "q75", "minimum", "maximum")),
    "correlations": (
        "property_error_correlations.csv",
        ("molecules", "spearman", "ci_low", "ci_high", "baseline_spearman", "partial_spearman"),
    ),
    "property_bins": (
        "property_bins.csv",
        (
            "bin_index",
            "molecules",
            "property_median",
            "actual_median",
            "mean_absolute_error",
            "baseline_mean_absolute_error",
            "normalized_mean_absolute_error",
            "mean_residual",
        ),
    ),
    "importance": (
        "property_importance.csv",
        (
            "importance_mean",
            "importance_std",
            "cross_validated_r2",
            "out_of_fold_spearman",
            "molecules",
        ),
    ),
    "agreement": (
        "model_agreement.csv",
        ("molecules", "spearman", "ci_low", "ci_high", "shared_worst_fraction",
         "residual_spearman"),
    ),
    "disagreement": (
        "model_disagreement.csv",
        (
            "rank",
            "dataset_index",
            "actual",
            "absolute_error_a",
            "absolute_error_b",
            "error_difference",
            "baseline_absolute_error",
            "nearest_train_similarity",
            *PROPERTY_NAMES,
        ),
    ),
    "shrinkage": (
        "shrinkage.csv",
        (
            "molecules",
            "slope",
            "intercept",
            "prediction_std_ratio",
            "mean_residual",
            "residual_actual_spearman",
            "low_solubility_bias",
            "high_solubility_bias",
        ),
    ),
    "scaffold_errors": (
        "scaffold_errors.csv",
        (
            "molecules",
            "dataset_molecules",
            "mean_absolute_error",
            "baseline_mean_absolute_error",
            "normalized_mean_absolute_error",
            "mean_residual",
        ),
    ),
    "case_studies": (
        "case_studies.csv",
        (
            "rank",
            "dataset_index",
            "actual",
            "predicted_mean",
            "absolute_error",
            "residual_mean",
            "baseline_absolute_error",
            "nearest_train_similarity",
            *PROPERTY_NAMES,
            *(f"{name}_percentile" for name in PROPERTY_NAMES),
        ),
    ),
}
ANALYSIS_FILENAMES = (
    *(filename for filename, _ in ANALYSIS_TABLES.values()),
    "analysis_config.json",
)
BOOLEAN_COLUMNS = (
    "excludes_zero",
    "trained_on",
    "is_control",
    "controls_included",
    "trustworthy",
    "same_family",
)


def molecular_properties(smiles_list: Sequence[str]) -> pd.DataFrame:
    """Calculate thirteen interpretable RDKit properties for each SMILES string.

    The first five columns are delegated to :func:`week1.calculate_descriptors` so the
    two modules can never drift apart; the remaining eight describe molecular size,
    ring content, saturation, and hydrogen-bonding capacity.
    """

    smiles_list = list(smiles_list)
    if not smiles_list:
        return pd.DataFrame(columns=list(PROPERTY_NAMES), dtype=float)

    rows: list[dict[str, float]] = []
    for position, smiles in enumerate(smiles_list):
        molecule = _parse_smiles(smiles, position=position)
        rows.append(
            {
                "HeavyAtomCount": float(molecule.GetNumHeavyAtoms()),
                "NumRings": float(rdMolDescriptors.CalcNumRings(molecule)),
                "NumAromaticRings": float(
                    rdMolDescriptors.CalcNumAromaticRings(molecule)
                ),
                "NumAliphaticRings": float(
                    rdMolDescriptors.CalcNumAliphaticRings(molecule)
                ),
                "NumHeteroatoms": float(rdMolDescriptors.CalcNumHeteroatoms(molecule)),
                "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(molecule)),
                "NumHDonors": float(Lipinski.NumHDonors(molecule)),
                "NumHAcceptors": float(Lipinski.NumHAcceptors(molecule)),
            }
        )

    extended = pd.DataFrame(rows, columns=list(EXTENDED_PROPERTY_NAMES), dtype=float)
    descriptors = calculate_descriptors(smiles_list)
    combined = pd.concat([descriptors, extended], axis=1)
    return combined[list(PROPERTY_NAMES)]


def property_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute properties once per unique molecule in a predictions frame."""

    _require_columns(predictions, ("dataset_index", "smiles"))
    unique = (
        predictions[["dataset_index", "smiles"]]
        .drop_duplicates("dataset_index")
        .sort_values("dataset_index")
        .reset_index(drop=True)
    )
    properties = molecular_properties(unique["smiles"])
    return pd.concat([unique, properties], axis=1)


def property_reference_table(properties: pd.DataFrame) -> pd.DataFrame:
    """Summarise each property's dataset-level spread so case studies have a yardstick."""

    rows = []
    for name in PROPERTY_NAMES:
        values = properties[name].to_numpy(dtype=float)
        rows.append(
            {
                "property": name,
                "median": float(np.nanmedian(values)),
                "q25": float(np.nanpercentile(values, 25)),
                "q75": float(np.nanpercentile(values, 75)),
                "minimum": float(np.nanmin(values)),
                "maximum": float(np.nanmax(values)),
            }
        )
    return pd.DataFrame(rows, columns=list(REFERENCE_COLUMNS))


def property_collinearity(
    properties: pd.DataFrame, *, threshold: float = DEFAULT_COLLINEARITY_THRESHOLD
) -> pd.DataFrame:
    """List property pairs correlated beyond ``threshold``.

    This table exists to disqualify the naive reading before it is offered: with
    ``MolWt`` and ``HeavyAtomCount`` at Spearman 0.93, no single-property claim about
    prediction error is identifiable on its own.
    """

    rows = []
    for position, first in enumerate(PROPERTY_NAMES):
        for second in PROPERTY_NAMES[position + 1 :]:
            value = rank_correlation(
                properties[first].to_numpy(dtype=float),
                properties[second].to_numpy(dtype=float),
            )
            if np.isfinite(value) and abs(value) > threshold:
                rows.append(
                    {"property_a": first, "property_b": second, "spearman": value}
                )
    frame = pd.DataFrame(rows, columns=list(COLLINEARITY_COLUMNS))
    return frame.sort_values("spearman", key=np.abs, ascending=False).reset_index(
        drop=True
    )


def property_correlation_matrix(properties: pd.DataFrame) -> pd.DataFrame:
    """Return the full Spearman matrix between properties, first column ``property``."""

    values = {
        name: properties[name].to_numpy(dtype=float) for name in PROPERTY_NAMES
    }
    matrix = pd.DataFrame(index=list(PROPERTY_NAMES), columns=list(PROPERTY_NAMES), dtype=float)
    for first in PROPERTY_NAMES:
        for second in PROPERTY_NAMES:
            matrix.loc[first, second] = (
                1.0 if first == second else rank_correlation(values[first], values[second])
            )
    return matrix.reset_index(names="property")


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman correlation via ranks, ``nan`` for degenerate input.

    Mirrors ``week3._correlation``'s contract but drops non-finite pairs first, because
    a single missing property value would otherwise poison the whole column.
    """

    left, right = _finite_pairs(left, right)
    if len(left) < 3:
        return float("nan")
    left_ranks = pd.Series(left).rank().to_numpy(dtype=float)
    right_ranks = pd.Series(right).rank().to_numpy(dtype=float)
    if left_ranks.std() == 0 or right_ranks.std() == 0:
        return float("nan")
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def partial_correlation(
    target: np.ndarray, feature: np.ndarray, controls: np.ndarray | None
) -> float:
    """Rank-correlate ``target`` and ``feature`` after removing the controls' influence.

    Both variables are rank-transformed and then residualised on the (rank-transformed)
    control columns by least squares. With ``baseline_absolute_error`` and
    ``nearest_train_similarity`` as controls this answers the question week 3 leaves
    open: does a molecular property explain error *beyond* intrinsic difficulty and
    similarity to the training set?
    """

    if controls is None:
        return rank_correlation(target, feature)

    controls = np.asarray(controls, dtype=float)
    if controls.ndim == 1:
        controls = controls.reshape(-1, 1)
    stacked = np.column_stack([np.asarray(target, dtype=float), np.asarray(feature, dtype=float), controls])
    stacked = stacked[np.isfinite(stacked).all(axis=1)]
    if len(stacked) < 3:
        return float("nan")

    ranked = np.column_stack(
        [pd.Series(stacked[:, column]).rank().to_numpy(dtype=float) for column in range(stacked.shape[1])]
    )
    control_block = ranked[:, 2:]
    # Drop constant controls; they contribute nothing and make the design singular.
    control_block = control_block[:, control_block.std(axis=0) > 0]
    if control_block.size == 0:
        return rank_correlation(stacked[:, 0], stacked[:, 1])

    design = np.column_stack([np.ones(len(ranked)), control_block])
    target_residual = _least_squares_residual(design, ranked[:, 0])
    feature_residual = _least_squares_residual(design, ranked[:, 1])
    if target_residual.std() == 0 or feature_residual.std() == 0:
        return float("nan")
    return float(np.corrcoef(target_residual, feature_residual)[0, 1])


def bootstrap_correlation_interval(
    left: np.ndarray,
    right: np.ndarray,
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return ``(point, ci_low, ci_high)`` from a percentile bootstrap over molecules."""

    point = rank_correlation(left, right)
    left, right = _finite_pairs(left, right)
    if not np.isfinite(point) or len(left) < 3 or samples <= 0:
        return point, float("nan"), float("nan")

    generator = np.random.default_rng(seed)
    values = np.empty(samples, dtype=float)
    for position in range(samples):
        picks = generator.integers(0, len(left), len(left))
        values[position] = rank_correlation(left[picks], right[picks])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return point, float("nan"), float("nan")

    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(values, [tail, 100.0 - tail])
    return point, float(low), float(high)


def pool_split_types(
    predictions: pd.DataFrame,
    *,
    regime_map: Mapping[str, str] = DEFAULT_REGIME_MAP,
) -> pd.DataFrame:
    """Add a ``regime`` column that pools the two scaffold splits into one sample.

    ``scaffold`` alone leaves 113 test molecules and barely any scaffold with enough
    members to analyse. Pooling it with ``scaffold_shuffled`` multiplies both counts at
    zero training cost. The raw ``split_type`` is kept so the deterministic split can
    still be reported unpooled, and unmapped split types fall back to their own name.
    """

    _require_columns(predictions, ("split_type",))
    regime = predictions["split_type"].map(lambda name: regime_map.get(name, name))
    return predictions.assign(regime=regime)


def effective_sample_sizes(
    predictions: pd.DataFrame,
    *,
    minimum_molecules: int = MINIMUM_SCAFFOLD_MOLECULES,
) -> pd.DataFrame:
    """Report how much independent evidence each regime actually carries.

    8,410 prediction rows are not 8,410 observations: they are a few hundred molecules
    seen under five seeds, and the seed axis means different things per regime. This is
    the table that stops the rest of the study being over-read.
    """

    frame = predictions if "regime" in predictions.columns else pool_split_types(predictions)
    rows = []
    for regime, group in frame.groupby("regime", sort=False):
        single_model = group[group["model"] == group["model"].iloc[0]]
        per_molecule = single_model.groupby("dataset_index")["seed"].nunique()
        scaffolds = single_model.drop_duplicates("dataset_index").groupby("scaffold").size()
        rows.append(
            {
                "regime": regime,
                "split_types": " + ".join(sorted(set(group["split_type"]))),
                "seeds": int(group["seed"].nunique()),
                "rows_per_model": int(len(single_model)),
                "distinct_molecules": int(single_model["dataset_index"].nunique()),
                "molecules_in_every_seed": int(
                    (per_molecule == group["seed"].nunique()).sum()
                ),
                "mean_seeds_per_molecule": float(per_molecule.mean()),
                "distinct_scaffolds": int(len(scaffolds)),
                "scaffolds_over_minimum": int((scaffolds >= minimum_molecules).sum()),
            }
        )
    frame = pd.DataFrame(rows, columns=list(SAMPLE_SIZE_COLUMNS))
    return _order_by_regime(frame)


def build_error_profile(
    predictions: pd.DataFrame,
    *,
    properties: pd.DataFrame | None = None,
    regime_map: Mapping[str, str] = DEFAULT_REGIME_MAP,
    baseline_model: str = BASELINE_MODEL,
    min_baseline_error: float = MIN_BASELINE_ERROR,
) -> pd.DataFrame:
    """Collapse the seed axis to one row per molecule and attach molecular properties.

    ``residual_mean`` is ``predicted - actual``, so a molecule whose true logS is -4.0
    and whose prediction is -1.7 has a residual of +2.3: the model pulled it toward the
    training mean. ``baseline_absolute_error`` is the ``Dummy (mean)`` model's own error
    on the same molecule, which is exactly ``|y - train_mean|`` and therefore serves as
    the intrinsic-difficulty control. ``normalized_error`` divides by it, and is NaN
    where that denominator is too small to be meaningful.
    """

    _require_columns(predictions, PREDICTION_COLUMNS)
    frame = pool_split_types(predictions, regime_map=regime_map)
    if baseline_model not in set(frame["model"]):
        raise ValueError(
            f"predictions contains no {baseline_model!r} rows; the intrinsic-difficulty "
            "control cannot be built"
        )

    grouped = frame.groupby(["regime", "model", "dataset_index"], sort=False)
    aggregated = grouped.agg(
        split_type=("split_type", lambda values: " + ".join(sorted(set(values)))),
        representation=("representation", "first"),
        smiles=("smiles", "first"),
        scaffold=("scaffold", "first"),
        seeds=("seed", "nunique"),
        actual=("actual", "mean"),
        predicted_mean=("predicted", "mean"),
        absolute_error=("absolute_error", "mean"),
        absolute_error_std=("absolute_error", "std"),
        nearest_train_similarity=("nearest_train_similarity", "mean"),
    ).reset_index()
    aggregated["residual_mean"] = aggregated["predicted_mean"] - aggregated["actual"]
    aggregated["family"] = aggregated["model"].map(
        lambda name: MODEL_FAMILIES.get(name, "baseline")
    )

    baseline = (
        aggregated[aggregated["model"] == baseline_model]
        .set_index(["regime", "dataset_index"])["absolute_error"]
        .rename("baseline_absolute_error")
    )
    aggregated = aggregated.join(
        baseline, on=["regime", "dataset_index"], how="left"
    )
    denominator = aggregated["baseline_absolute_error"].where(
        aggregated["baseline_absolute_error"] >= min_baseline_error
    )
    aggregated["normalized_error"] = aggregated["absolute_error"] / denominator

    properties = property_table(predictions) if properties is None else properties
    merged = aggregated.merge(
        properties.drop(columns="smiles", errors="ignore"),
        on="dataset_index",
        how="left",
    )
    missing = merged[list(PROPERTY_NAMES)].isna().any(axis=1)
    if bool(missing.any()):
        warnings.warn(
            f"{int(missing.sum())} rows have missing properties and are kept with NaN; "
            "correlations drop them pairwise",
            stacklevel=2,
        )
    return _order_by_regime(_order_by_model(merged))[list(PROFILE_COLUMNS)]


def property_error_correlation(
    profile: pd.DataFrame,
    *,
    properties: Sequence[str] = PROPERTY_NAMES,
    target: str = "absolute_error",
    control_features: Sequence[str] = CONTROL_FEATURES,
    baseline_model: str = BASELINE_MODEL,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> pd.DataFrame:
    """Correlate every property with per-molecule error, with three honesty columns.

    ``spearman`` is the raw relationship, ``baseline_spearman`` is the same correlation
    for the ``Dummy (mean)`` control (i.e. how much of the trend is just intrinsic
    difficulty), and ``partial_spearman`` is what survives after residualising on
    difficulty and nearest-train similarity. ``trained_on`` flags the five descriptors
    the Linear and MLP models were fitted on, whose correlations are therefore partly
    circular.
    """

    rows = []
    for (regime, model), group in profile.groupby(["regime", "model"], sort=False):
        baseline_group = profile[
            (profile["regime"] == regime) & (profile["model"] == baseline_model)
        ]
        controls = (
            group[list(control_features)].to_numpy(dtype=float)
            if control_features
            else None
        )
        errors = group[target].to_numpy(dtype=float)
        for name in properties:
            values = group[name].to_numpy(dtype=float)
            point, low, high = bootstrap_correlation_interval(
                values, errors, samples=samples, confidence=confidence, seed=seed
            )
            rows.append(
                {
                    "regime": regime,
                    "model": model,
                    "property": name,
                    "molecules": int(np.isfinite(values * errors).sum()),
                    "spearman": point,
                    "ci_low": low,
                    "ci_high": high,
                    "excludes_zero": bool(
                        np.isfinite(low) and np.isfinite(high) and low * high > 0
                    ),
                    "baseline_spearman": rank_correlation(
                        baseline_group[name].to_numpy(dtype=float),
                        baseline_group[target].to_numpy(dtype=float),
                    ),
                    "partial_spearman": partial_correlation(errors, values, controls),
                    "trained_on": name in DESCRIPTOR_NAMES,
                }
            )
    return _order_by_regime(_order_by_model(pd.DataFrame(rows, columns=list(CORRELATION_COLUMNS))))


def bin_property_errors(
    profile: pd.DataFrame,
    property_name: str,
    *,
    quantiles: Sequence[float] = DEFAULT_PROPERTY_QUANTILES,
) -> pd.DataFrame:
    """Report error per property quantile bin, normalised by the baseline's own error.

    This is week 3's Dummy-normalisation moved onto a property axis. Every aggregate is
    a ratio of means (bin MAE divided by bin baseline MAE), never a mean of per-molecule
    ratios, which would be dominated by molecules sitting near the training mean.
    """

    if property_name not in PROPERTY_NAMES:
        raise ValueError(f"property_name must be one of {PROPERTY_NAMES}")

    rows = []
    for (regime, model), group in profile.groupby(["regime", "model"], sort=False):
        values = group[property_name]
        # duplicates="drop" keeps a constant property from raising; it collapses to one bin.
        try:
            bins = pd.qcut(values, q=list(quantiles), duplicates="drop")
            # qcut returns an ordered Categorical, so its categories give ascending bins.
            # Appearance order would put the panels' x-axis out of sequence.
            labels = [str(category) for category in bins.cat.categories]
            bins = bins.astype(str)
        except ValueError:
            bins = pd.Series(["all"] * len(group), index=group.index, dtype=object)
            labels = ["all"]
        for index, label in enumerate(labels):
            panel = group[bins == label]
            if panel.empty:
                continue
            baseline_mae = float(panel["baseline_absolute_error"].mean())
            mae = float(panel["absolute_error"].mean())
            rows.append(
                {
                    "regime": regime,
                    "model": model,
                    "property": property_name,
                    "property_bin": label,
                    "bin_index": index,
                    "molecules": int(len(panel)),
                    "property_median": float(panel[property_name].median()),
                    "actual_median": float(panel["actual"].median()),
                    "mean_absolute_error": mae,
                    "baseline_mean_absolute_error": baseline_mae,
                    "normalized_mean_absolute_error": (
                        mae / baseline_mae if baseline_mae > 0 else float("nan")
                    ),
                    "mean_residual": float(panel["residual_mean"].mean()),
                }
            )
    return _order_by_regime(_order_by_model(pd.DataFrame(rows, columns=list(PROPERTY_BIN_COLUMNS))))


def property_bin_summary(
    profile: pd.DataFrame,
    *,
    properties: Sequence[str] = PROPERTY_NAMES,
    quantiles: Sequence[float] = DEFAULT_PROPERTY_QUANTILES,
) -> pd.DataFrame:
    """Stack :func:`bin_property_errors` over every property."""

    frames = [
        bin_property_errors(profile, name, quantiles=quantiles) for name in properties
    ]
    return pd.concat(frames, ignore_index=True)


def error_property_importance(
    profile: pd.DataFrame,
    *,
    properties: Sequence[str] = PROPERTY_NAMES,
    control_features: Sequence[str] = CONTROL_FEATURES,
    models: Sequence[str] | None = None,
    target: str = "absolute_error",
    folds: int = DEFAULT_IMPORTANCE_FOLDS,
    repeats: int = DEFAULT_IMPORTANCE_REPEATS,
    seed: int = 0,
    minimum_molecules: int = MINIMUM_IMPORTANCE_MOLECULES,
) -> pd.DataFrame:
    """Fit ``|error| ~ properties + controls`` and report importance behind an R2 gate.

    The cross-validated R2 comes first on purpose. Permutation importance from a model
    fitted on thirteen collinear features will always produce a confident-looking
    ranking, even when the model has no predictive power at all; ``trustworthy`` marks
    whether the ranking may be read as a finding. The controls enter as features so the
    question becomes whether properties add anything on top of week 3's similarity
    story rather than restating it. Pass ``control_features=()`` for the properties-only
    contrast; the ``controls_included`` column distinguishes the two fits.
    """

    models = tuple(models) if models is not None else tuple(
        name for name in MODEL_ORDER if name != BASELINE_MODEL
    )
    features = (*properties, *control_features)
    rows = []
    for (regime, model), group in profile.groupby(["regime", "model"], sort=False):
        if model not in models:
            continue
        usable = group.dropna(subset=[*features, target])
        if len(usable) < minimum_molecules:
            raise ValueError(
                f"regime {regime!r} model {model!r} has {len(usable)} usable molecules, "
                f"fewer than the required {minimum_molecules}"
            )

        matrix = usable[list(features)].to_numpy(dtype=float)
        errors = usable[target].to_numpy(dtype=float)
        estimator = GradientBoostingRegressor(
            random_state=seed, max_depth=2, n_estimators=200
        )
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        out_of_fold = cross_val_predict(estimator, matrix, errors, cv=splitter)
        residual_sum = float(((errors - out_of_fold) ** 2).sum())
        total_sum = float(((errors - errors.mean()) ** 2).sum())
        r2 = 1.0 - residual_sum / total_sum if total_sum > 0 else float("nan")
        trustworthy = bool(np.isfinite(r2) and r2 > 0)

        # Importance is measured fold by fold on held-out molecules, then averaged, so a
        # model that only memorises its training rows cannot manufacture a ranking.
        collected: list[np.ndarray] = []
        for train_index, test_index in splitter.split(matrix):
            estimator.fit(matrix[train_index], errors[train_index])
            result = permutation_importance(
                estimator,
                matrix[test_index],
                errors[test_index],
                n_repeats=repeats,
                random_state=seed,
            )
            collected.append(result.importances_mean)
        importances = np.vstack(collected)

        for position, name in enumerate(features):
            rows.append(
                {
                    "regime": regime,
                    "model": model,
                    "controls_included": bool(control_features),
                    "feature": name,
                    "is_control": name in control_features,
                    "importance_mean": float(importances[:, position].mean()),
                    "importance_std": float(importances[:, position].std()),
                    "cross_validated_r2": r2,
                    "out_of_fold_spearman": rank_correlation(out_of_fold, errors),
                    "trustworthy": trustworthy,
                    "molecules": int(len(usable)),
                }
            )
    return _order_by_regime(_order_by_model(pd.DataFrame(rows, columns=list(IMPORTANCE_COLUMNS))))


def model_error_agreement(
    profile: pd.DataFrame,
    *,
    models: Sequence[str] | None = None,
    worst_fraction: float = DEFAULT_WORST_FRACTION,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> pd.DataFrame:
    """Correlate per-molecule errors between every pair of models.

    This is the study's discriminating test. If two models fail on the same molecules
    the difficulty lives in the chemistry or the labels; if they fail on different
    molecules it lives in the representation. ``same_family`` marks pairs that share a
    representation, so the within-family and across-family correlations can be read
    against each other.
    """

    models = tuple(models) if models is not None else tuple(
        name for name in MODEL_ORDER if name in set(profile["model"])
    )
    unknown = set(models) - set(profile["model"])
    if unknown:
        raise ValueError(
            f"profile has no rows for {sorted(unknown)}; available models are "
            f"{sorted(set(profile['model']))}"
        )

    rows = []
    for regime, group in profile.groupby("regime", sort=False):
        errors = group.pivot_table(
            index="dataset_index", columns="model", values="absolute_error"
        )
        residuals = group.pivot_table(
            index="dataset_index", columns="model", values="residual_mean"
        )
        for position, first in enumerate(models):
            for second in models[position + 1 :]:
                if first not in errors.columns or second not in errors.columns:
                    continue
                pair = errors[[first, second]].dropna()
                point, low, high = bootstrap_correlation_interval(
                    pair[first].to_numpy(dtype=float),
                    pair[second].to_numpy(dtype=float),
                    samples=samples,
                    confidence=confidence,
                    seed=seed,
                )
                rows.append(
                    {
                        "regime": regime,
                        "model_a": first,
                        "model_b": second,
                        "same_family": (
                            MODEL_FAMILIES.get(first, "baseline")
                            == MODEL_FAMILIES.get(second, "baseline")
                        ),
                        "molecules": int(len(pair)),
                        "spearman": point,
                        "ci_low": low,
                        "ci_high": high,
                        "shared_worst_fraction": _shared_worst_fraction(
                            pair[first], pair[second], worst_fraction
                        ),
                        "residual_spearman": rank_correlation(
                            residuals[first].to_numpy(dtype=float),
                            residuals[second].to_numpy(dtype=float),
                        ),
                    }
                )
    return _order_by_regime(pd.DataFrame(rows, columns=list(AGREEMENT_COLUMNS)))


def model_error_disagreement(
    profile: pd.DataFrame,
    *,
    models: Sequence[str] = DIAGNOSTIC_MODELS,
    regime: str = OUT_OF_SCAFFOLD_REGIME,
    top_n: int = 10,
) -> pd.DataFrame:
    """Name the molecules where one model beats the other by the widest margin."""

    first, second = models
    panel = profile[profile["regime"] == regime]
    errors = panel.pivot_table(
        index="dataset_index", columns="model", values="absolute_error"
    )
    for name in (first, second):
        if name not in errors.columns:
            raise ValueError(
                f"profile has no {name!r} rows in regime {regime!r}; available models "
                f"are {sorted(errors.columns)}"
            )

    metadata = panel.drop_duplicates("dataset_index").set_index("dataset_index")
    pair = errors[[first, second]].dropna()
    difference = pair[first] - pair[second]

    rows = []
    for favours, ordered in (
        (second, difference.sort_values(ascending=False)),
        (first, difference.sort_values()),
    ):
        for rank, (index, value) in enumerate(ordered.head(top_n).items(), start=1):
            record = metadata.loc[index]
            rows.append(
                {
                    "regime": regime,
                    "favours": favours,
                    "rank": rank,
                    "dataset_index": int(index),
                    "smiles": record["smiles"],
                    "scaffold": record["scaffold"],
                    "actual": float(record["actual"]),
                    "absolute_error_a": float(pair.loc[index, first]),
                    "absolute_error_b": float(pair.loc[index, second]),
                    "error_difference": float(abs(value)),
                    "baseline_absolute_error": float(record["baseline_absolute_error"]),
                    "nearest_train_similarity": float(record["nearest_train_similarity"]),
                    **{name: float(record[name]) for name in PROPERTY_NAMES},
                }
            )
    return pd.DataFrame(rows, columns=list(DISAGREEMENT_COLUMNS))


def shrinkage_fit(
    profile: pd.DataFrame,
    *,
    low_quantile: float = 0.25,
    high_quantile: float = 0.75,
) -> pd.DataFrame:
    """Fit ``predicted ~ slope * actual`` to quantify regression toward the mean.

    A slope below one means the model compresses the target range. The two bias columns
    show which tail pays for it: a positive ``low_solubility_bias`` says the least
    soluble molecules are predicted as more soluble than they are.
    """

    rows = []
    for (regime, model), group in profile.groupby(["regime", "model"], sort=False):
        actual = group["actual"].to_numpy(dtype=float)
        predicted = group["predicted_mean"].to_numpy(dtype=float)
        residual = group["residual_mean"].to_numpy(dtype=float)
        if len(actual) < 3 or actual.std() == 0:
            slope = intercept = float("nan")
        else:
            slope, intercept = (float(value) for value in np.polyfit(actual, predicted, 1))
        low_cut = float(np.quantile(actual, low_quantile))
        high_cut = float(np.quantile(actual, high_quantile))
        rows.append(
            {
                "regime": regime,
                "model": model,
                "molecules": int(len(group)),
                "slope": slope,
                "intercept": intercept,
                "prediction_std_ratio": (
                    float(predicted.std() / actual.std()) if actual.std() > 0 else float("nan")
                ),
                "mean_residual": float(residual.mean()),
                "residual_actual_spearman": rank_correlation(actual, residual),
                "low_solubility_bias": float(residual[actual <= low_cut].mean()),
                "high_solubility_bias": float(residual[actual >= high_cut].mean()),
            }
        )
    return _order_by_regime(_order_by_model(pd.DataFrame(rows, columns=list(SHRINKAGE_COLUMNS))))


def scaffold_error_table(
    profile: pd.DataFrame,
    *,
    scaffold_counts: Mapping[str, int] | pd.Series | None = None,
    minimum_molecules: int = MINIMUM_SCAFFOLD_MOLECULES,
) -> pd.DataFrame:
    """Aggregate error per Murcko scaffold for scaffolds with enough test molecules.

    Counts are per regime, which is why the pooled ``out_of_scaffold`` regime exists:
    the deterministic scaffold split on its own leaves almost no scaffold with more than
    a couple of test molecules.
    """

    counts = pd.Series(dict(scaffold_counts)) if scaffold_counts is not None else None
    rows = []
    for (regime, model), group in profile.groupby(["regime", "model"], sort=False):
        for scaffold, panel in group.groupby("scaffold", sort=False):
            if len(panel) < minimum_molecules:
                continue
            baseline_mae = float(panel["baseline_absolute_error"].mean())
            mae = float(panel["absolute_error"].mean())
            rows.append(
                {
                    "regime": regime,
                    "model": model,
                    "scaffold": scaffold,
                    "molecules": int(len(panel)),
                    "dataset_molecules": (
                        float(counts.get(scaffold, float("nan")))
                        if counts is not None
                        else float("nan")
                    ),
                    "mean_absolute_error": mae,
                    "baseline_mean_absolute_error": baseline_mae,
                    "normalized_mean_absolute_error": (
                        mae / baseline_mae if baseline_mae > 0 else float("nan")
                    ),
                    "mean_residual": float(panel["residual_mean"].mean()),
                }
            )
    frame = pd.DataFrame(rows, columns=list(SCAFFOLD_ERROR_COLUMNS))
    frame = frame.sort_values(
        ["regime", "model", "normalized_mean_absolute_error"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    return _order_by_regime(_order_by_model(frame))


def case_studies(
    profile: pd.DataFrame,
    *,
    regime: str = OUT_OF_SCAFFOLD_REGIME,
    model: str = "GIN",
    top_n: int = DEFAULT_CASE_COUNT,
    min_baseline_error: float = MIN_BASELINE_ERROR,
) -> pd.DataFrame:
    """Return the worst and best predicted molecules with property percentiles.

    "Best" is restricted to molecules the baseline finds non-trivial. Without that
    guard the best-predicted set is simply the molecules whose logS happens to equal the
    training mean, which teaches nothing about the model.
    """

    panel = profile[(profile["regime"] == regime) & (profile["model"] == model)]
    if panel.empty:
        raise ValueError(f"profile has no rows for regime {regime!r} model {model!r}")

    percentiles = {
        name: panel[name].rank(pct=True) * 100.0 for name in PROPERTY_NAMES
    }
    annotated = panel.assign(**{f"{name}_percentile": values for name, values in percentiles.items()})
    worst = annotated.sort_values("absolute_error", ascending=False).head(top_n)
    candidates = annotated[annotated["baseline_absolute_error"] >= min_baseline_error]
    best = candidates.sort_values("absolute_error").head(top_n)

    rows = []
    for case, selection in (("worst", worst), ("best", best)):
        for rank, (_, record) in enumerate(selection.iterrows(), start=1):
            rows.append(
                {
                    "regime": regime,
                    "model": model,
                    "case": case,
                    "rank": rank,
                    "dataset_index": int(record["dataset_index"]),
                    "smiles": record["smiles"],
                    "scaffold": record["scaffold"],
                    "actual": float(record["actual"]),
                    "predicted_mean": float(record["predicted_mean"]),
                    "absolute_error": float(record["absolute_error"]),
                    "residual_mean": float(record["residual_mean"]),
                    "baseline_absolute_error": float(record["baseline_absolute_error"]),
                    "nearest_train_similarity": float(record["nearest_train_similarity"]),
                    **{name: float(record[name]) for name in PROPERTY_NAMES},
                    **{
                        f"{name}_percentile": float(record[f"{name}_percentile"])
                        for name in PROPERTY_NAMES
                    },
                }
            )
    return pd.DataFrame(rows, columns=list(CASE_STUDY_COLUMNS))


@dataclass(frozen=True)
class AnalysisConfig:
    """Settings for the whole week 4 error-analysis pipeline."""

    regimes: tuple[str, ...] = REGIME_ORDER
    models: tuple[str, ...] = MODEL_ORDER
    properties: tuple[str, ...] = PROPERTY_NAMES
    headline_properties: tuple[str, ...] = HEADLINE_PROPERTIES
    diagnostic_models: tuple[str, ...] = DIAGNOSTIC_MODELS
    control_features: tuple[str, ...] = CONTROL_FEATURES
    regime_map: tuple[tuple[str, str], ...] = tuple(sorted(DEFAULT_REGIME_MAP.items()))
    min_baseline_error: float = MIN_BASELINE_ERROR
    property_quantiles: tuple[float, ...] = DEFAULT_PROPERTY_QUANTILES
    minimum_scaffold_molecules: int = MINIMUM_SCAFFOLD_MOLECULES
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES
    confidence: float = DEFAULT_CONFIDENCE
    importance_folds: int = DEFAULT_IMPORTANCE_FOLDS
    importance_repeats: int = DEFAULT_IMPORTANCE_REPEATS
    importance_minimum_molecules: int = MINIMUM_IMPORTANCE_MOLECULES
    case_count: int = DEFAULT_CASE_COUNT
    case_model: str = "GIN"
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("regimes", "models", "properties", "diagnostic_models"):
            values = getattr(self, name)
            if not values:
                raise ValueError(f"{name} must contain at least one value")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique")
        unknown = set(self.properties) - set(PROPERTY_NAMES)
        if unknown:
            raise ValueError(f"unknown properties: {sorted(unknown)}")
        unknown = set(self.models) - set(MODEL_ORDER)
        if unknown:
            raise ValueError(f"unknown models: {sorted(unknown)}")
        if len(self.diagnostic_models) != 2:
            raise ValueError("diagnostic_models must name exactly two models")
        if BASELINE_MODEL in self.diagnostic_models:
            raise ValueError(f"diagnostic_models must not include {BASELINE_MODEL!r}")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie strictly between 0 and 1")
        if self.min_baseline_error < 0:
            raise ValueError("min_baseline_error must be non-negative")
        if self.minimum_scaffold_molecules < 2:
            raise ValueError("minimum_scaffold_molecules must be at least 2")
        quantiles = self.property_quantiles
        if list(quantiles) != sorted(quantiles) or quantiles[0] != 0.0 or quantiles[-1] != 1.0:
            raise ValueError("property_quantiles must increase from 0.0 to 1.0")

    @property
    def regime_mapping(self) -> dict[str, str]:
        """Return ``regime_map`` as a plain dictionary."""

        return dict(self.regime_map)


@dataclass
class ErrorAnalysis:
    """Every table produced by :func:`run_error_analysis`."""

    profile: pd.DataFrame
    properties: pd.DataFrame
    sample_sizes: pd.DataFrame
    collinearity: pd.DataFrame
    reference: pd.DataFrame
    correlations: pd.DataFrame
    property_bins: pd.DataFrame
    importance: pd.DataFrame
    agreement: pd.DataFrame
    disagreement: pd.DataFrame
    shrinkage: pd.DataFrame
    scaffold_errors: pd.DataFrame
    case_studies: pd.DataFrame
    config: AnalysisConfig = field(default_factory=AnalysisConfig)


def run_error_analysis(
    predictions: pd.DataFrame,
    *,
    smiles_list: Sequence[str] | None = None,
    config: AnalysisConfig | None = None,
    verbose: bool = True,
) -> ErrorAnalysis:
    """Run the whole week 4 pipeline over week 3's cached predictions."""

    config = config or AnalysisConfig()
    properties = property_table(predictions)
    if verbose:
        print(f"  properties: {len(properties)} molecules")

    profile = build_error_profile(
        predictions,
        properties=properties,
        regime_map=config.regime_mapping,
        min_baseline_error=config.min_baseline_error,
    )
    profile = profile[profile["regime"].isin(config.regimes)]
    if verbose:
        print(f"  profile: {len(profile)} rows")

    counts = (
        scaffold_table(smiles_list).set_index("scaffold")["molecules"]
        if smiles_list is not None
        else None
    )
    return ErrorAnalysis(
        profile=profile,
        properties=properties,
        sample_sizes=effective_sample_sizes(
            pool_split_types(predictions, regime_map=config.regime_mapping),
            minimum_molecules=config.minimum_scaffold_molecules,
        ),
        collinearity=property_collinearity(properties),
        reference=property_reference_table(properties),
        correlations=property_error_correlation(
            profile,
            properties=config.properties,
            control_features=config.control_features,
            samples=config.bootstrap_samples,
            confidence=config.confidence,
            seed=config.seed,
        ),
        property_bins=property_bin_summary(
            profile, properties=config.properties, quantiles=config.property_quantiles
        ),
        # Fitted twice: with the week 3 controls, and on properties alone. The pair of
        # R2 values is what says whether properties carry error signal of their own.
        importance=pd.concat(
            [
                error_property_importance(
                    profile,
                    properties=config.properties,
                    control_features=controls,
                    folds=config.importance_folds,
                    repeats=config.importance_repeats,
                    seed=config.seed,
                    minimum_molecules=config.importance_minimum_molecules,
                )
                for controls in (config.control_features, ())
            ],
            ignore_index=True,
        ),
        agreement=model_error_agreement(
            profile,
            samples=config.bootstrap_samples,
            confidence=config.confidence,
            seed=config.seed,
        ),
        disagreement=model_error_disagreement(
            profile, models=config.diagnostic_models, regime=OUT_OF_SCAFFOLD_REGIME
        ),
        shrinkage=shrinkage_fit(profile),
        scaffold_errors=scaffold_error_table(
            profile,
            scaffold_counts=counts,
            minimum_molecules=config.minimum_scaffold_molecules,
        ),
        case_studies=case_studies(
            profile,
            model=config.case_model,
            top_n=config.case_count,
            min_baseline_error=config.min_baseline_error,
        ),
        config=config,
    )


def save_error_analysis(
    analysis: ErrorAnalysis, directory: str | Path = DEFAULT_RESULTS_DIRECTORY
) -> dict[str, Path]:
    """Write every analysis table plus the configuration to ``directory``."""

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, (filename, _) in ANALYSIS_TABLES.items():
        path = target / filename
        frame = getattr(analysis, name)
        # The per-molecule table is the large one; four decimals match week 3's choice.
        precision = "%.4f" if name == "profile" else "%.6f"
        frame.to_csv(path, index=False, float_format=precision)
        paths[name] = path
    paths["analysis_config"] = target / "analysis_config.json"
    paths["analysis_config"].write_text(
        json.dumps(_config_to_dict(analysis.config), indent=2), encoding="utf-8"
    )
    return paths


def load_error_analysis(
    directory: str | Path = DEFAULT_RESULTS_DIRECTORY,
) -> ErrorAnalysis:
    """Load a previously saved analysis from disk."""

    target = Path(directory)
    tables = {
        name: _read_analysis_csv(target / filename, numeric)
        for name, (filename, numeric) in ANALYSIS_TABLES.items()
    }
    config = _config_from_dict(
        json.loads((target / "analysis_config.json").read_text(encoding="utf-8"))
    )
    return ErrorAnalysis(config=config, **tables)


def load_or_run_error_analysis(
    predictions: pd.DataFrame | None = None,
    *,
    smiles_list: Sequence[str] | None = None,
    directory: str | Path = DEFAULT_RESULTS_DIRECTORY,
    config: AnalysisConfig | None = None,
    refresh: bool = False,
    verbose: bool = True,
) -> ErrorAnalysis:
    """Reuse a cached analysis when it matches the requested configuration."""

    config = config or AnalysisConfig()
    target = Path(directory)
    if not refresh and all((target / name).exists() for name in ANALYSIS_FILENAMES):
        cached = load_error_analysis(target)
        if cached.config == config:
            return cached
        warnings.warn(
            f"cached analysis in {target} was built with a different configuration; "
            "recomputing",
            stacklevel=2,
        )
    if predictions is None:
        raise ValueError(
            "predictions are required when the cache is missing, stale, or refreshed"
        )
    analysis = run_error_analysis(
        predictions, smiles_list=smiles_list, config=config, verbose=verbose
    )
    save_error_analysis(analysis, target)
    return analysis


def plot_property_collinearity(
    properties: pd.DataFrame, *, title: str | None = None
) -> Figure:
    """Heatmap of property-property Spearman correlations, trained-on columns starred."""

    matrix = property_correlation_matrix(properties).set_index("property")
    figure, axis = plt.subplots(figsize=(9, 7.5))
    image = axis.imshow(matrix.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1)
    labels = [
        f"{name}*" if name in DESCRIPTOR_NAMES else name for name in matrix.columns
    ]
    axis.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize="small")
    axis.set_yticks(range(len(labels)), labels, fontsize="small")
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = matrix.to_numpy(dtype=float)[row, column]
            if np.isfinite(value):
                axis.text(
                    column,
                    row,
                    f"{value:.2f}".replace("0.", "."),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if abs(value) > 0.6 else "black",
                )
    figure.colorbar(image, ax=axis, label="Spearman correlation")
    axis.set_title(title or "Property collinearity (* = a Linear/MLP input feature)")
    figure.tight_layout()
    return figure


def plot_property_error_correlation(
    correlations: pd.DataFrame,
    *,
    models: Sequence[str] | None = None,
    title: str | None = None,
) -> Figure:
    """Heatmap of property-error Spearman per model, one panel per regime.

    Cells whose bootstrap interval spans zero are hatched, so the eye is not drawn to
    correlations the data cannot distinguish from noise.
    """

    selected = (
        correlations
        if models is None
        else correlations[correlations["model"].isin(list(models))]
    )
    regimes = list(dict.fromkeys(selected["regime"]))
    figure, axes = plt.subplots(
        1, len(regimes), figsize=(1 + 4.5 * len(regimes), 6), squeeze=False, sharey=True
    )
    for axis, regime in zip(axes[0], regimes, strict=True):
        panel = selected[selected["regime"] == regime]
        grid = panel.pivot_table(
            index="property", columns="model", values="spearman", sort=False
        )
        significant = panel.pivot_table(
            index="property", columns="model", values="excludes_zero", sort=False
        )
        grid = grid.reindex([name for name in PROPERTY_NAMES if name in grid.index])
        significant = significant.reindex(grid.index)
        image = axis.imshow(grid.to_numpy(dtype=float), cmap="coolwarm", vmin=-0.5, vmax=0.5)
        axis.set_xticks(
            range(len(grid.columns)), list(grid.columns), rotation=45, ha="right"
        )
        axis.set_yticks(range(len(grid.index)), list(grid.index), fontsize="small")
        for row in range(len(grid.index)):
            for column in range(len(grid.columns)):
                value = grid.to_numpy(dtype=float)[row, column]
                if not np.isfinite(value):
                    continue
                clears = bool(significant.to_numpy()[row, column])
                axis.text(
                    column,
                    row,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold" if clears else "normal",
                    color="white" if abs(value) > 0.3 else "black",
                    alpha=1.0 if clears else 0.45,
                )
        axis.set_title(regime)
        figure.colorbar(image, ax=axis, fraction=0.046, label="Spearman rho")
    figure.suptitle(
        title or "Property vs absolute error (bold = bootstrap CI excludes zero)"
    )
    figure.tight_layout()
    return figure


def plot_property_error_bins(
    property_bins: pd.DataFrame,
    *,
    properties: Sequence[str] = HEADLINE_PROPERTIES,
    regime: str = OUT_OF_SCAFFOLD_REGIME,
    models: Sequence[str] | None = None,
    title: str | None = None,
) -> Figure:
    """Normalised error across property quartiles, one panel per property.

    The dashed line at 1.0 is mean-prediction accuracy: a model above it is doing worse
    than predicting the training mean for those molecules.
    """

    selected = property_bins[property_bins["regime"] == regime]
    if models is not None:
        selected = selected[selected["model"].isin(list(models))]
    else:
        selected = selected[selected["model"] != BASELINE_MODEL]

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    for axis, name in zip(axes.flat, properties, strict=False):
        panel = selected[selected["property"] == name]
        for model, group in panel.groupby("model", sort=False, observed=True):
            group = group.sort_values("bin_index")
            axis.plot(
                group["property_median"],
                group["normalized_mean_absolute_error"],
                marker="o",
                label=model,
            )
        axis.axhline(1.0, color="0.4", linestyle="--", linewidth=1)
        axis.set_title(name)
        axis.legend(fontsize="small")
        counts = panel.groupby("bin_index", observed=True)["molecules"].first()
        axis.set_xlabel(
            f"{name} (bin median; n = {', '.join(str(int(v)) for v in counts)})"
        )
    for axis in axes[:, 0]:
        axis.set_ylabel("MAE / Dummy MAE")
    figure.suptitle(title or f"Normalised error across property quartiles ({regime})")
    figure.tight_layout()
    return figure


def plot_error_property_importance(
    importance: pd.DataFrame,
    *,
    regime: str = OUT_OF_SCAFFOLD_REGIME,
    models: Sequence[str] = DIAGNOSTIC_MODELS,
    controls_included: bool = True,
    title: str | None = None,
) -> Figure:
    """Held-out permutation importance, with the cross-validated R2 in every title.

    The R2 is in the title because the ranking must not be read when it is at or below
    zero: a model with no predictive power still produces a confident-looking ordering.
    """

    selected = importance[importance["regime"] == regime]
    if "controls_included" in selected.columns:
        selected = selected[selected["controls_included"] == controls_included]
    figure, axes = plt.subplots(
        1, len(models), figsize=(6 * len(models), 6), squeeze=False, sharex=True
    )
    for axis, model in zip(axes[0], models, strict=True):
        panel = selected[selected["model"] == model].sort_values("importance_mean")
        colors = ["0.6" if flag else "tab:blue" for flag in panel["is_control"]]
        axis.barh(
            range(len(panel)),
            panel["importance_mean"],
            xerr=panel["importance_std"],
            color=colors,
            hatch="",
        )
        for position, flag in enumerate(panel["is_control"]):
            if flag:
                axis.patches[position].set_hatch("//")
        axis.set_yticks(range(len(panel)), list(panel["feature"]), fontsize="small")
        r2 = float(panel["cross_validated_r2"].iloc[0]) if len(panel) else float("nan")
        verdict = "readable" if r2 > 0 else "NOT readable"
        axis.set_title(f"{model} — held-out R2 {r2:+.3f} ({verdict})")
        axis.set_xlabel("Permutation importance on held-out molecules")
    figure.suptitle(title or f"Does any property predict error? ({regime}; hatched = control)")
    figure.tight_layout()
    return figure


def plot_model_error_agreement(
    profile: pd.DataFrame,
    *,
    pairs: Sequence[tuple[str, str]] = (("GCN", "GIN"), ("MLP", "GIN"), ("Linear", "MLP")),
    regime: str = OUT_OF_SCAFFOLD_REGIME,
    agreement: pd.DataFrame | None = None,
    title: str | None = None,
) -> Figure:
    """Per-molecule error of one model against another, one panel per pair.

    Within-family pairs clustering on the diagonal while the across-family pair scatters
    is the signature of a representation-specific failure.
    """

    panel = profile[profile["regime"] == regime]
    errors = panel.pivot_table(
        index="dataset_index", columns="model", values="absolute_error"
    )
    figure, axes = plt.subplots(
        1, len(pairs), figsize=(5 * len(pairs), 5), squeeze=False, sharex=True, sharey=True
    )
    limit = float(np.nanpercentile(errors.to_numpy(dtype=float), 99))
    for axis, (first, second) in zip(axes[0], pairs, strict=True):
        if first not in errors.columns or second not in errors.columns:
            continue
        pair = errors[[first, second]].dropna()
        axis.scatter(pair[first], pair[second], s=10, alpha=0.35, color="tab:blue")
        axis.plot([0, limit], [0, limit], color="0.4", linestyle="--", linewidth=1)
        rho = rank_correlation(
            pair[first].to_numpy(dtype=float), pair[second].to_numpy(dtype=float)
        )
        same = MODEL_FAMILIES.get(first) == MODEL_FAMILIES.get(second)
        axis.set_title(
            f"{first} vs {second} — rho {rho:+.2f}"
            f" ({'same' if same else 'different'} family)"
        )
        axis.set_xlabel(f"{first} |error| (logS)")
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
    axes[0][0].set_ylabel("Second model |error| (logS)")
    figure.suptitle(title or f"Do models fail on the same molecules? ({regime})")
    figure.tight_layout()
    return figure


def plot_residual_shrinkage(
    profile: pd.DataFrame,
    *,
    models: Sequence[str] = DIAGNOSTIC_MODELS,
    regime: str = OUT_OF_SCAFFOLD_REGIME,
    title: str | None = None,
) -> Figure:
    """Predicted versus actual logS and the signed bias against actual logS."""

    panel = profile[profile["regime"] == regime]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    bounds = (
        float(panel["actual"].min()) - 0.5,
        float(panel["actual"].max()) + 0.5,
    )
    for model in models:
        group = panel[panel["model"] == model]
        if group.empty:
            continue
        actual = group["actual"].to_numpy(dtype=float)
        predicted = group["predicted_mean"].to_numpy(dtype=float)
        slope, intercept = (float(value) for value in np.polyfit(actual, predicted, 1))
        axes[0].scatter(actual, predicted, s=10, alpha=0.3, label=None)
        line = np.linspace(*bounds, 50)
        axes[0].plot(line, slope * line + intercept, label=f"{model} (slope {slope:.2f})")

        edges = np.quantile(actual, np.linspace(0, 1, 6))
        centres, biases = [], []
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            mask = (actual >= low) & (actual <= high)
            if mask.sum() >= 3:
                centres.append(float(np.median(actual[mask])))
                biases.append(float(group["residual_mean"].to_numpy(dtype=float)[mask].mean()))
        axes[1].plot(centres, biases, marker="o", label=model)

    axes[0].plot(list(bounds), list(bounds), color="0.3", linestyle="--", linewidth=1,
                 label="perfect (slope 1.00)")
    axes[0].set_xlabel("Actual logS")
    axes[0].set_ylabel("Predicted logS")
    axes[0].set_title("Predictions compress the target range")
    axes[0].legend(fontsize="small")
    axes[1].axhline(0.0, color="0.3", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Actual logS (bin median)")
    axes[1].set_ylabel("Mean signed error (predicted - actual)")
    axes[1].set_title("The insoluble tail is pulled toward the mean")
    axes[1].legend(fontsize="small")
    figure.suptitle(title or f"Regression toward the training mean ({regime})")
    figure.tight_layout()
    return figure


def plot_scaffold_errors(
    scaffold_errors: pd.DataFrame,
    *,
    regime: str = OUT_OF_SCAFFOLD_REGIME,
    model: str = "GIN",
    top_n: int = 12,
    title: str | None = None,
) -> Figure:
    """Per-scaffold normalised error, worst scaffolds first."""

    panel = scaffold_errors[
        (scaffold_errors["regime"] == regime) & (scaffold_errors["model"] == model)
    ].head(top_n)
    figure, axis = plt.subplots(figsize=(10, max(4.0, 0.45 * len(panel) + 2)))
    labels = [
        f"{_shorten(row.scaffold) or '(acyclic)'} (n={int(row.molecules)})"
        for row in panel.itertuples()
    ]
    positions = range(len(panel))
    colors = [
        "tab:red" if value > 1.0 else "tab:blue"
        for value in panel["normalized_mean_absolute_error"]
    ]
    axis.barh(list(positions), panel["normalized_mean_absolute_error"], color=colors)
    axis.axvline(1.0, color="0.3", linestyle="--", linewidth=1)
    axis.set_yticks(list(positions), labels, fontsize="small")
    axis.invert_yaxis()
    axis.set_xlabel("MAE / Dummy MAE (red = worse than predicting the mean)")
    axis.set_title(title or f"{model} error by Murcko scaffold ({regime})")
    figure.tight_layout()
    return figure


def plot_case_study_structures(
    cases: pd.DataFrame, *, top_n: int = 4, title: str | None = None
) -> Figure:
    """Structure grid of the worst and best predicted molecules.

    Uses ``Draw.MolToImage`` composed into matplotlib axes rather than
    ``Draw.MolsToGridImage``, which returns a PIL image and would break the module's
    "every ``plot_*`` returns a ``Figure``" contract.
    """

    rows = [
        cases[cases["case"] == case].sort_values("rank").head(top_n)
        for case in ("worst", "best")
    ]
    columns = max(1, max(len(frame) for frame in rows))
    figure, axes = plt.subplots(2, columns, figsize=(3.2 * columns, 7.2), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for row_index, (case, frame) in enumerate(zip(("worst", "best"), rows, strict=True)):
        for column_index, record in enumerate(frame.itertuples()):
            axis = axes[row_index][column_index]
            molecule = Chem.MolFromSmiles(record.smiles)
            if molecule is not None:
                axis.imshow(Draw.MolToImage(molecule, size=(320, 320)))
            axis.set_title(
                f"{case} #{int(record.rank)}\n"
                f"true {record.actual:+.2f} / pred {record.predicted_mean:+.2f}\n"
                f"|e| {record.absolute_error:.2f}, sim {record.nearest_train_similarity:.2f}",
                fontsize=8,
            )
    figure.suptitle(title or "Worst and best predicted molecules")
    figure.tight_layout()
    return figure


def _shared_worst_fraction(
    first: pd.Series, second: pd.Series, fraction: float
) -> float:
    count = max(1, int(round(len(first) * fraction)))
    worst_first = set(first.nlargest(count).index)
    worst_second = set(second.nlargest(count).index)
    return len(worst_first & worst_second) / count


def _least_squares_residual(design: np.ndarray, values: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def _finite_pairs(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    return left[mask], right[mask]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing required columns: {missing}")


def _order_by_regime(frame: pd.DataFrame) -> pd.DataFrame:
    if "regime" not in frame.columns:
        return frame
    ordering = {name: position for position, name in enumerate(REGIME_ORDER)}
    key = frame["regime"].map(lambda name: ordering.get(name, len(ordering)))
    return (
        frame.assign(_regime_order=key)
        .sort_values("_regime_order", kind="stable")
        .drop(columns="_regime_order")
        .reset_index(drop=True)
    )


def _read_analysis_csv(path: Path, numeric_columns: Sequence[str]) -> pd.DataFrame:
    frame = _read_csv(path, numeric_columns)
    for column in BOOLEAN_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype(str).isin(("True", "true", "1"))
    return frame


def _config_to_dict(config: AnalysisConfig) -> dict[str, object]:
    payload = asdict(config)
    for name, value in payload.items():
        if isinstance(value, tuple):
            payload[name] = [list(item) if isinstance(item, tuple) else item for item in value]
    return payload


def _config_from_dict(payload: Mapping[str, object]) -> AnalysisConfig:
    return AnalysisConfig(
        regimes=tuple(payload["regimes"]),
        models=tuple(payload["models"]),
        properties=tuple(payload["properties"]),
        headline_properties=tuple(payload["headline_properties"]),
        diagnostic_models=tuple(payload["diagnostic_models"]),
        control_features=tuple(payload["control_features"]),
        regime_map=tuple((str(key), str(value)) for key, value in payload["regime_map"]),
        min_baseline_error=float(payload["min_baseline_error"]),
        property_quantiles=tuple(float(value) for value in payload["property_quantiles"]),
        minimum_scaffold_molecules=int(payload["minimum_scaffold_molecules"]),
        bootstrap_samples=int(payload["bootstrap_samples"]),
        confidence=float(payload["confidence"]),
        importance_folds=int(payload["importance_folds"]),
        importance_repeats=int(payload["importance_repeats"]),
        importance_minimum_molecules=int(payload["importance_minimum_molecules"]),
        case_count=int(payload["case_count"]),
        case_model=str(payload["case_model"]),
        seed=int(payload["seed"]),
    )


def build_figures(analysis: ErrorAnalysis) -> dict[str, Figure]:
    """Build every week 4 figure, keyed by the filename ``save_figures`` will use."""

    return {
        "property_collinearity": plot_property_collinearity(analysis.properties),
        "property_error_correlation": plot_property_error_correlation(
            analysis.correlations
        ),
        # Both regimes are kept: the size trend lives in ``in_domain`` and reverses once
        # the scaffold is unfamiliar, so a single panel would tell half the story.
        **{
            f"property_error_bins_{regime}": plot_property_error_bins(
                analysis.property_bins,
                properties=analysis.config.headline_properties,
                regime=regime,
            )
            for regime in analysis.config.regimes
        },
        "property_importance": plot_error_property_importance(
            analysis.importance, models=analysis.config.diagnostic_models
        ),
        "model_error_agreement": plot_model_error_agreement(analysis.profile),
        "residual_shrinkage": plot_residual_shrinkage(
            analysis.profile, models=analysis.config.diagnostic_models
        ),
        "scaffold_errors": plot_scaffold_errors(
            analysis.scaffold_errors, model=analysis.config.case_model
        ),
        "case_study_structures": plot_case_study_structures(analysis.case_studies),
    }


def summarise(analysis: ErrorAnalysis, *, prediction_rows: int | None = None) -> dict[str, object]:
    """Assemble the JSON summary that ``reports/week4.md`` quotes from."""

    regime = OUT_OF_SCAFFOLD_REGIME
    strongest = (
        analysis.correlations[
            (analysis.correlations["regime"] == regime)
            & (analysis.correlations["model"] == analysis.config.case_model)
        ]
        .assign(magnitude=lambda frame: frame["spearman"].abs())
        .sort_values("magnitude", ascending=False)
    )
    headline: dict[str, object] = {"regime": regime}
    if not strongest.empty:
        best = strongest.iloc[0]
        headline.update(
            {
                "strongest_property": best["property"],
                "strongest_spearman": float(best["spearman"]),
                "strongest_ci": [float(best["ci_low"]), float(best["ci_high"])],
                "strongest_partial_spearman": float(best["partial_spearman"]),
                "strongest_baseline_spearman": float(best["baseline_spearman"]),
            }
        )
    importance = analysis.importance[analysis.importance["regime"] == regime]
    if not importance.empty:
        for included, label in ((True, "with_controls"), (False, "properties_only")):
            panel = importance[importance["controls_included"] == included]
            headline[f"cross_validated_r2_{label}"] = {
                model: float(group["cross_validated_r2"].iloc[0])
                for model, group in panel.groupby("model", sort=False)
            }
        properties_only = importance[~importance["controls_included"]]
        headline["properties_only_trustworthy"] = bool(
            properties_only["trustworthy"].all()
        )
        top = (
            importance[importance["controls_included"]]
            .sort_values("importance_mean", ascending=False)
            .drop_duplicates("model")
        )
        headline["top_feature"] = {
            row.model: {"feature": row.feature, "is_control": bool(row.is_control)}
            for row in top.itertuples()
        }
    agreement = analysis.agreement[analysis.agreement["regime"] == regime]
    if not agreement.empty:
        headline["within_family_spearman"] = float(
            agreement[agreement["same_family"]]["spearman"].mean()
        )
        headline["across_family_spearman"] = float(
            agreement[~agreement["same_family"]]["spearman"].mean()
        )
    shrinkage = analysis.shrinkage[analysis.shrinkage["regime"] == regime]
    headline["shrinkage_slope"] = {
        row.model: float(row.slope) for row in shrinkage.itertuples()
    }

    # The report only quotes the headline properties' bins and the case molecules
    # without their percentile block; the exhaustive versions stay in the CSVs.
    headline_bins = analysis.property_bins[
        analysis.property_bins["property"].isin(list(analysis.config.headline_properties))
    ]
    cases = analysis.case_studies[
        [name for name in analysis.case_studies.columns if not name.endswith("_percentile")]
    ]

    return {
        "headline": headline,
        "effective_sample_sizes": analysis.sample_sizes.to_dict("records"),
        "property_collinearity": analysis.collinearity.to_dict("records"),
        "property_reference": analysis.reference.to_dict("records"),
        "property_error_correlations": analysis.correlations.to_dict("records"),
        "property_bins": headline_bins.to_dict("records"),
        "property_importance": analysis.importance.to_dict("records"),
        "model_agreement": analysis.agreement.to_dict("records"),
        "model_disagreement": analysis.disagreement.to_dict("records"),
        "shrinkage": analysis.shrinkage.to_dict("records"),
        "scaffold_errors": analysis.scaffold_errors.to_dict("records"),
        "case_studies": cases.to_dict("records"),
        "source": {
            "week3_directory": str(WEEK3_RESULTS_DIRECTORY),
            "prediction_rows": prediction_rows,
        },
        # No training happens here, so recording torch or CUDA versions would imply a
        # dependency this analysis does not have.
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
    }


def main() -> None:
    """Analyse week 3's predictions and write every artifact under results/week4."""

    from .week1 import load_esol
    from .week3 import load_or_run_sweep

    dataset = load_esol()
    smiles = [str(graph.smiles) for graph in dataset]
    print("week 3 predictions")
    sweep = load_or_run_sweep(dataset, directory=WEEK3_RESULTS_DIRECTORY)
    print(f"  rows: {len(sweep.predictions)}")

    print("week 4 analysis")
    analysis = load_or_run_error_analysis(
        sweep.predictions,
        smiles_list=smiles,
        directory=DEFAULT_RESULTS_DIRECTORY,
        refresh=True,
    )

    directory = DEFAULT_RESULTS_DIRECTORY
    (directory / "summary.json").write_text(
        json.dumps(
            summarise(analysis, prediction_rows=len(sweep.predictions)),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    figures = build_figures(analysis)
    for name, path in save_figures(figures, directory / "figures").items():
        print(f"  {name}: {path}")
    for figure in figures.values():
        plt.close(figure)

    print("\neffective sample sizes")
    print(analysis.sample_sizes.to_string(index=False))
    print("\nerror agreement between models")
    print(analysis.agreement.to_string(index=False))
    print("\nshrinkage")
    print(analysis.shrinkage.to_string(index=False))
    print("\nmultivariate error model")
    gate = analysis.importance.drop_duplicates(["regime", "model"])
    print(
        gate[
            ["regime", "model", "cross_validated_r2", "out_of_fold_spearman", "trustworthy"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
