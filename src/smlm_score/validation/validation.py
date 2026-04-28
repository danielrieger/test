"""
Validation helpers for SMLM-IMP scoring workflows.

The EMAN2 workflow is expected to provide NPC-enriched picked particles, so
cluster-vs-noise separation is a fallback QC check. The thesis-facing
validation path compares an aligned NPC model against realistic null controls
on held-out localizations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ValidationResult:
    """Structured result from a single validation check."""

    test_name: str
    passed: bool
    details: str
    skipped: bool = False
    valid_scores: Optional[Dict[str, float]] = None
    control_scores: Optional[Dict[str, float]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


def _normalize_score(score: float, n_points, scoring_type: str = "") -> float:
    """Normalize a total log-likelihood to mean score per localization."""
    if n_points > 0:
        return float(score) / float(n_points)
    return float(score)


def validate_scoring_separation(
    cluster_scores: Dict[int, Dict[str, float]],
    scoring_types: Optional[List[str]] = None,
) -> List[ValidationResult]:
    """
    Fallback QC: valid NPC clusters should score above available noise controls.

    In EMAN2-picked workflows, noise controls may not exist. In that case the
    check is skipped rather than treated as a failure.
    """
    if scoring_types is None:
        scoring_types = ["Tree", "GMM", "Distance"]

    results = []
    for stype in scoring_types:
        valid_norm_scores = []
        noise_norm_scores = []

        for scores in cluster_scores.values():
            if stype not in scores:
                continue

            raw_score = scores[stype]
            n_points = scores.get("n_points", 1)
            norm_score = _normalize_score(raw_score, n_points, scoring_type=stype)

            if scores.get("type") == "Valid":
                valid_norm_scores.append(norm_score)
            elif scores.get("type") == "Noise":
                noise_norm_scores.append(norm_score)

        if len(valid_norm_scores) == 0 or len(noise_norm_scores) == 0:
            results.append(
                ValidationResult(
                    test_name=f"Separation_{stype}",
                    passed=False,
                    skipped=True,
                    details=(
                        "Separation skipped: no valid/noise contrast is available. "
                        "This is expected when EMAN2 particle picking yields "
                        "NPC-enriched clusters."
                    ),
                    metrics={
                        "n_valid": len(valid_norm_scores),
                        "n_noise": len(noise_norm_scores),
                        "skipped_reason": "missing_valid_or_noise_controls",
                    },
                )
            )
            continue

        mean_valid = float(np.mean(valid_norm_scores))
        mean_noise = float(np.mean(noise_norm_scores))
        std_noise = (
            float(np.std(noise_norm_scores))
            if len(noise_norm_scores) > 1
            else abs(mean_noise) * 0.1
        )
        separation = (mean_valid - mean_noise) / std_noise if std_noise > 0 else float("inf")
        passed = bool(mean_valid > mean_noise)

        results.append(
            ValidationResult(
                test_name=f"Separation_{stype}",
                passed=passed,
                details=(
                    f"Valid mean/pt: {mean_valid:.4f}, Noise mean/pt: {mean_noise:.4f}, "
                    f"Separation: {separation:.2f} sigma. "
                    f"{'PASS' if passed else 'FAIL'}: Valid NPCs "
                    f"{'outscore' if passed else 'do not outscore'} noise."
                ),
                valid_scores={f"valid_{i}": s for i, s in enumerate(valid_norm_scores)},
                control_scores={f"noise_{i}": s for i, s in enumerate(noise_norm_scores)},
                metrics={
                    "mean_valid_per_point": mean_valid,
                    "mean_noise_per_point": mean_noise,
                    "mean_valid_per_pt": mean_valid,
                    "mean_noise_per_pt": mean_noise,
                    "separation_sigma": separation,
                    "n_valid": len(valid_norm_scores),
                    "n_noise": len(noise_norm_scores),
                },
            )
        )

    return results


def validate_with_held_out_data(
    valid_cluster_score: float,
    valid_n_points: int,
    held_out_scores: List[float],
    held_out_n_points: List[int],
    scoring_type: str,
) -> ValidationResult:
    """Compare a valid NPC score against held-out/off-region controls."""
    if len(held_out_scores) == 0:
        return ValidationResult(
            test_name=f"HeldOut_{scoring_type}",
            passed=False,
            details="No held-out samples available for comparison.",
            skipped=True,
            metrics={"skipped_reason": "no_held_out_samples"},
        )

    valid_norm = _normalize_score(valid_cluster_score, valid_n_points, scoring_type=scoring_type)
    held_out_norm = [
        _normalize_score(s, n, scoring_type=scoring_type)
        for s, n in zip(held_out_scores, held_out_n_points)
    ]

    mean_held_out = float(np.mean(held_out_norm))
    std_held_out = (
        float(np.std(held_out_norm))
        if len(held_out_norm) > 1
        else abs(mean_held_out) * 0.1
    )
    separation = (
        (valid_norm - mean_held_out) / std_held_out
        if std_held_out > 0
        else float("inf")
    )
    passed = bool(valid_norm > mean_held_out)
    held_out_all_zero = bool(all(abs(s) < 1e-12 for s in held_out_scores))

    warning = (
        " Held-out controls are all zero; this is treated as degenerate evidence, "
        "not an automatic Tree pass."
        if held_out_all_zero
        else ""
    )

    return ValidationResult(
        test_name=f"HeldOut_{scoring_type}",
        passed=passed,
        details=(
            f"Valid score/pt: {valid_norm:.4f} (n={valid_n_points}), "
            f"Held-out mean/pt: {mean_held_out:.4f} (+/-{std_held_out:.4f}), "
            f"Separation: {separation:.2f} sigma. "
            f"{'PASS' if passed else 'FAIL'}: Valid NPC "
            f"{'outscores' if passed else 'does not outscore'} held-out data."
            f"{warning}"
        ),
        valid_scores={"valid_per_point": valid_norm, "valid_per_pt": valid_norm},
        control_scores={f"held_out_{i}_per_point": s for i, s in enumerate(held_out_norm)},
        metrics={
            "valid_score_per_point": valid_norm,
            "valid_score_per_pt": valid_norm,
            "held_out_mean_per_point": mean_held_out,
            "held_out_mean_per_pt": mean_held_out,
            "held_out_std_per_point": std_held_out,
            "separation_sigma": separation,
            "held_out_all_zero": held_out_all_zero,
            "n_held_out_samples": len(held_out_scores),
        },
    )


def _scramble_data(data: np.ndarray, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Destroy coordinate coupling while preserving marginal coordinate values."""
    rng = rng or np.random
    scrambled = np.asarray(data).copy()
    for i in range(scrambled.shape[1]):
        scrambled[:, i] = rng.permutation(scrambled[:, i])
    return scrambled


def _radial_null(data: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Preserve radii and z values while randomizing angular structure."""
    pts = np.asarray(data, dtype=np.float64)
    null_pts = pts.copy()
    radii = np.linalg.norm(pts[:, :2], axis=1)
    angles = rng.uniform(-np.pi, np.pi, size=len(pts))
    null_pts[:, 0] = radii * np.cos(angles)
    null_pts[:, 1] = radii * np.sin(angles)
    return null_pts


def _rotate_model_z(model_coords: np.ndarray, angle_rad: float) -> np.ndarray:
    rot = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
            [np.sin(angle_rad), np.cos(angle_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return np.asarray(model_coords, dtype=np.float64) @ rot.T


def _mirror_model_x(model_coords: np.ndarray) -> np.ndarray:
    mirrored = np.asarray(model_coords, dtype=np.float64).copy()
    mirrored[:, 0] *= -1.0
    return mirrored


def _translate_model_xy(model_coords: np.ndarray, scale: float = 0.75) -> np.ndarray:
    translated = np.asarray(model_coords, dtype=np.float64).copy()
    xy_radius = np.linalg.norm(translated[:, :2], axis=1)
    shift = max(float(np.nanmedian(xy_radius)) * scale, 25.0)
    translated[:, 0] += shift
    return translated


def _is_2d_points(points: np.ndarray, z_tol: float = 1e-6) -> bool:
    pts = np.asarray(points)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array.")
    if pts.shape[1] < 3:
        return True
    if len(pts) == 0:
        return True
    return bool(np.nanmax(pts[:, 2]) - np.nanmin(pts[:, 2]) <= z_tol)


def _fit_alignment(points: np.ndarray, data_dim: str = "auto") -> Dict[str, Any]:
    """Fit train-only centering/PCA transform."""
    pts = np.asarray(points, dtype=np.float64)
    if data_dim not in {"auto", "2d", "3d"}:
        raise ValueError("data_dim must be 'auto', '2d', or '3d'.")

    use_2d = data_dim == "2d" or (data_dim == "auto" and _is_2d_points(pts))
    centroid = pts.mean(axis=0)

    if use_2d:
        translation = np.zeros(3, dtype=np.float64)
        translation[:2] = -centroid[:2]
        return {
            "translation": translation,
            "rotation": np.eye(3, dtype=np.float64),
            "data_dim": "2d",
            "used_pca": False,
        }

    centered = pts - centroid
    cov_matrix = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sort_indices = np.argsort(eigenvalues)[::-1]
    rotation = eigenvectors[:, sort_indices].T
    if np.linalg.det(rotation) < 0:
        rotation[2, :] *= -1

    return {
        "translation": -centroid,
        "rotation": rotation,
        "data_dim": "3d",
        "used_pca": True,
    }


def _apply_alignment(points: np.ndarray, transform: Dict[str, Any]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).copy()
    if transform["data_dim"] == "2d":
        pts[:, :2] = pts[:, :2] + transform["translation"][:2]
        return pts
    return (pts + transform["translation"]) @ transform["rotation"].T


def _align_model(model_coords: np.ndarray, transform: Dict[str, Any]) -> np.ndarray:
    model = np.asarray(model_coords, dtype=np.float64).copy()
    if transform["data_dim"] == "2d":
        model[:, 2] = 0.0
        return model
    return model @ transform["rotation"].T


def _covariances_from_variances(variances, n_points: int):
    if variances is None:
        return None
    arr = np.asarray(variances)
    if arr.ndim == 1:
        return np.array([np.eye(3) * max(float(v), 1e-9) for v in arr[:n_points]])
    return arr[:n_points]


def _score_model_against_points(
    scoring_type: str,
    model: Any,
    avs: List[Any],
    model_coords: np.ndarray,
    points: np.ndarray,
    variances=None,
) -> Optional[float]:
    from sklearn.neighbors import KDTree as SKKDTree
    from smlm_score.imp_modeling.restraint.scoring_restraint import ScoringRestraintWrapper
    from smlm_score.imp_modeling.scoring.gmm_score import test_gmm_components

    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return None

    if scoring_type == "Tree":
        sr = ScoringRestraintWrapper(
            model,
            avs,
            kdtree_obj=SKKDTree(pts),
            dataxyz=pts,
            var=variances,
            searchradius=50.0,
            model_coords_override=model_coords,
            type=scoring_type,
        )
    elif scoring_type == "GMM":
        if len(pts) < 3:
            return None
        _, gmm_obj, gmm_mean, gmm_cov, gmm_w = test_gmm_components(pts)
        sr = ScoringRestraintWrapper(
            model,
            avs,
            gmm_sel_components=gmm_obj.n_components,
            gmm_sel_mean=gmm_mean,
            gmm_sel_cov=gmm_cov,
            gmm_sel_weight=gmm_w,
            model_coords_override=model_coords,
            type=scoring_type,
        )
    elif scoring_type == "Distance":
        sr = ScoringRestraintWrapper(
            model,
            avs,
            dataxyz=pts,
            var=_covariances_from_variances(variances, len(pts)),
            model_coords_override=model_coords,
            type=scoring_type,
        )
    else:
        return None

    return float(sr.evaluate())


def validate_model_vs_nulls(
    cluster_points: np.ndarray,
    model_coords: np.ndarray,
    scoring_type: str,
    model: Any,
    avs: List[Any],
    variances=None,
    data_dim: str = "auto",
    n_repeats: int = 5,
    test_fraction: float = 0.35,
    n_nulls: int = 8,
    random_seed: int = 42,
    min_split_points: int = 20,
) -> ValidationResult:
    """Validate held-out model fit against scrambled, rotated, mirrored, and radial nulls."""
    pts = np.asarray(cluster_points, dtype=np.float64)
    if len(pts) < 2 * min_split_points:
        return ValidationResult(
            test_name=f"ModelVsNull_{scoring_type}",
            passed=False,
            skipped=True,
            details=f"Insufficient points for model-vs-null validation ({len(pts)}).",
            metrics={"skipped_reason": "insufficient_points", "n_points": len(pts)},
        )

    rng = np.random.RandomState(random_seed)
    real_scores = []
    null_scores = []
    deltas = []
    wins = []
    null_labels = []

    variances_arr = None if variances is None else np.asarray(variances)
    n_test = max(min_split_points, int(round(len(pts) * test_fraction)))
    n_test = min(n_test, len(pts) - min_split_points)

    for repeat_idx in range(n_repeats):
        perm = rng.permutation(len(pts))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        train_pts = pts[train_idx]
        test_pts = pts[test_idx]
        test_vars = None if variances_arr is None else variances_arr[test_idx]

        transform = _fit_alignment(train_pts, data_dim=data_dim)
        test_aligned = _apply_alignment(test_pts, transform)
        model_aligned = _align_model(model_coords, transform)

        raw_real = _score_model_against_points(
            scoring_type, model, avs, model_aligned, test_aligned, variances=test_vars
        )
        if raw_real is None:
            continue
        real = _normalize_score(raw_real, len(test_aligned), scoring_type)

        repeat_nulls = []
        for null_idx in range(n_nulls):
            if transform["data_dim"] == "2d":
                mode = null_idx % 4
                if mode == 0:
                    null_label = "scrambled"
                    null_pts = _scramble_data(test_aligned, rng=rng)
                    null_model = model_aligned
                elif mode == 1:
                    null_label = "translated_model"
                    null_pts = test_aligned
                    null_model = _translate_model_xy(model_aligned)
                elif mode == 2:
                    null_label = "radial"
                    null_pts = _radial_null(test_aligned, rng)
                    null_model = model_aligned
                else:
                    null_label = "radial_translated_model"
                    null_pts = _radial_null(test_aligned, rng)
                    null_model = _translate_model_xy(model_aligned)
            else:
                mode = null_idx % 4
                if mode == 0:
                    null_label = "scrambled"
                    null_pts = _scramble_data(test_aligned, rng=rng)
                    null_model = model_aligned
                elif mode == 1:
                    null_label = "rotated_model"
                    null_pts = test_aligned
                    null_model = _rotate_model_z(model_aligned, np.deg2rad(22.5))
                elif mode == 2:
                    null_label = "mirrored_model"
                    null_pts = test_aligned
                    null_model = _mirror_model_x(model_aligned)
                else:
                    null_label = "radial"
                    null_pts = _radial_null(test_aligned, rng)
                    null_model = model_aligned

            raw_null = _score_model_against_points(
                scoring_type, model, avs, null_model, null_pts, variances=test_vars
            )
            if raw_null is None:
                continue
            null = _normalize_score(raw_null, len(null_pts), scoring_type)
            repeat_nulls.append(null)
            null_scores.append(null)
            deltas.append(real - null)
            wins.append(real > null)
            null_labels.append(null_label)

        if repeat_nulls:
            real_scores.append(real)

    if not real_scores or not null_scores:
        return ValidationResult(
            test_name=f"ModelVsNull_{scoring_type}",
            passed=False,
            skipped=True,
            details="Model-vs-null validation could not evaluate any complete folds.",
            metrics={"skipped_reason": "no_complete_folds"},
        )

    mean_real = float(np.mean(real_scores))
    mean_null = float(np.mean(null_scores))
    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas))
    effect_size = mean_delta / std_delta if std_delta > 0 else float("inf")
    win_rate = float(np.mean(wins))
    label_metrics = {}
    for label in sorted(set(null_labels)):
        idx = [i for i, x in enumerate(null_labels) if x == label]
        label_deltas = [deltas[i] for i in idx]
        label_wins = [wins[i] for i in idx]
        label_metrics[f"{label}_mean_delta"] = float(np.mean(label_deltas))
        label_metrics[f"{label}_win_rate"] = float(np.mean(label_wins))

    if _is_2d_points(pts) and scoring_type in {"Tree", "Distance"}:
        passed = bool(mean_delta > 0.0 and effect_size > 0.25)
    else:
        passed = bool(mean_delta > 0.0 and win_rate >= 0.75)

    return ValidationResult(
        test_name=f"ModelVsNull_{scoring_type}",
        passed=passed,
        details=(
            f"Real mean/pt: {mean_real:.4f}, Null mean/pt: {mean_null:.4f}, "
            f"Delta: {mean_delta:.4f}, Win rate: {win_rate:.2f}, "
            f"Effect size: {effect_size:.2f}. "
            f"{'PASS' if passed else 'FAIL'}: model "
            f"{'beats' if passed else 'does not consistently beat'} null controls."
        ),
        valid_scores={f"real_{i}": s for i, s in enumerate(real_scores)},
        control_scores={f"null_{i}": s for i, s in enumerate(null_scores)},
        metrics={
            "mean_real_per_point": mean_real,
            "mean_null_per_point": mean_null,
            "mean_delta": mean_delta,
            "effect_size": effect_size,
            "win_rate": win_rate,
            "n_repeats": n_repeats,
            "n_nulls": n_nulls,
            "n_real_scores": len(real_scores),
            "n_null_scores": len(null_scores),
            **label_metrics,
        },
    )


def validate_cross_validated_npc(
    cluster_points: np.ndarray,
    model_coords: np.ndarray,
    scoring_type: str,
    model: Any,
    avs: List[Any],
    n_splits: int = 2,
    debug: bool = False,
    **kwargs,
) -> ValidationResult:
    """Backward-compatible wrapper for the model-vs-null structural validator."""
    return validate_model_vs_nulls(
        cluster_points=cluster_points,
        model_coords=model_coords,
        scoring_type=scoring_type,
        model=model,
        avs=avs,
        n_repeats=kwargs.get("n_repeats", 5),
        n_nulls=kwargs.get("n_nulls", 8),
        data_dim=kwargs.get("data_dim", "auto"),
    )


def _validation_not_run() -> ValidationResult:
    return ValidationResult(
        test_name="Validation_NotRun",
        passed=False,
        skipped=True,
        details="No validation inputs were available.",
        metrics={"skipped_reason": "no_validation_inputs"},
    )


def run_full_validation(
    cluster_scores: Dict[int, Dict[str, float]],
    held_out_results: Optional[Dict[str, Dict]] = None,
    scoring_types: Optional[List[str]] = None,
    cross_val_data: Optional[Dict[str, Any]] = None,
    cluster_id: Optional[int] = None,
) -> List[ValidationResult]:
    """Run all applicable validation checks and print a compact summary."""
    if scoring_types is None:
        scoring_types = ["Tree", "GMM"]

    all_results: List[ValidationResult] = []

    if cluster_scores:
        all_results.extend(validate_scoring_separation(cluster_scores, scoring_types=scoring_types))

    if held_out_results:
        for stype in scoring_types:
            if stype not in held_out_results:
                continue
            held = held_out_results[stype]
            all_results.append(
                validate_with_held_out_data(
                    valid_cluster_score=held["valid_score"],
                    valid_n_points=held["valid_n_points"],
                    held_out_scores=held["held_out_scores"],
                    held_out_n_points=held["held_out_n_points"],
                    scoring_type=stype,
                )
            )

    if cross_val_data:
        for stype in scoring_types:
            all_results.append(
                validate_model_vs_nulls(
                    cluster_points=cross_val_data["cluster_points"],
                    model_coords=cross_val_data["model_coords"],
                    scoring_type=stype,
                    model=cross_val_data["model"],
                    avs=cross_val_data["avs"],
                    variances=cross_val_data.get("variances"),
                    data_dim=cross_val_data.get("data_dim", "auto"),
                )
            )

    if not all_results:
        all_results.append(_validation_not_run())

    cid_str = f": Cluster {cluster_id}" if cluster_id is not None else ""
    print(f"\nVALIDATION SUMMARY{cid_str}")
    print("-" * 58)
    print(f"{'Test Name':<22} | {'Metric':<13} | {'Status':<9}")
    print("-" * 58)

    n_passed = 0
    n_total = len([r for r in all_results if not r.skipped])

    for r in all_results:
        if r.skipped:
            status = "[SKIP]"
        else:
            status = "[PASS]" if r.passed else "[FAIL]"
            if r.passed:
                n_passed += 1

        if "win_rate" in r.metrics:
            metric_str = f"win {r.metrics['win_rate']:.2f}"
        elif "separation_sigma" in r.metrics:
            metric_str = f"{r.metrics['separation_sigma']:6.2f} sigma"
        else:
            metric_str = "n/a"

        print(f"{r.test_name:<22} | {metric_str:<13} | {status:<9}")

    print("-" * 58)
    print(f"OVERALL: {n_passed}/{n_total} NON-SKIPPED PASSED")
    print("-" * 58 + "\n")

    return all_results
