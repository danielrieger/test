import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

# Ensure imports work from both examples/ and repo root execution.
_examples_dir = os.path.dirname(__file__)
_project_dir = os.path.abspath(os.path.join(_examples_dir, ".."))
_workspace_dir = os.path.abspath(os.path.join(_project_dir, ".."))
for _p in (_workspace_dir, _project_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from smlm_score.imp_modeling.restraint.scoring_restraint import ScoringRestraintWrapper
from smlm_score.imp_modeling.scoring.gmm_score import test_gmm_components
from smlm_score.utility.data_handling import (
    align_npc_cluster_pca,
    compute_av,
    flexible_filter_smlm_data,
    isolate_individual_npcs,
)
from smlm_score.utility.input import read_experimental_data, read_parameters_from_json


def _load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_minmax(values):
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return arr
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.full_like(arr, 0.5, dtype=float)
    vmin = np.nanmin(arr[finite])
    vmax = np.nanmax(arr[finite])
    if abs(vmax - vmin) < 1e-12:
        out = np.full_like(arr, 0.5, dtype=float)
        out[~finite] = 0.0
        return out
    out = (arr - vmin) / (vmax - vmin)
    out[~finite] = 0.0
    return out


def _geometry_quality(aligned_points, expected_diameter_nm=120.0, diameter_sigma_nm=45.0):
    xy = aligned_points[:, :2]
    z = aligned_points[:, 2] if aligned_points.shape[1] > 2 else np.zeros(len(aligned_points))

    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    bbox_xy = maxs - mins
    width = float(max(bbox_xy[0], 1e-9))
    height = float(max(bbox_xy[1], 1e-9))
    mean_diameter = 0.5 * (width + height)
    circularity = min(width, height) / max(width, height)
    z_std = float(np.std(z))

    diameter_score = np.exp(-((mean_diameter - expected_diameter_nm) ** 2) / (2.0 * diameter_sigma_nm**2))
    z_planarity_score = np.exp(-(z_std**2) / (2.0 * 20.0**2))
    quality = 0.5 * diameter_score + 0.3 * circularity + 0.2 * z_planarity_score

    return {
        "bbox_x_nm": width,
        "bbox_y_nm": height,
        "mean_diameter_nm": mean_diameter,
        "circularity": circularity,
        "z_std_nm": z_std,
        "diameter_score": diameter_score,
        "z_planarity_score": z_planarity_score,
        "geometry_quality": quality,
    }


def _score_candidate(model, avs, model_centered_baseline, cluster_points, cluster_variances):
    align = align_npc_cluster_pca(cluster_points, debug=False)
    aligned = align["aligned_data"]
    rotation = align["rotation"]
    model_aligned = np.dot(model_centered_baseline, rotation.T)
    n_points = len(aligned)

    # Tree score
    tree_wrapper = ScoringRestraintWrapper(
        model,
        avs,
        kdtree_obj=KDTree(aligned),
        dataxyz=aligned,
        var=cluster_variances,
        searchradius=50.0,
        model_coords_override=model_aligned,
        type="Tree",
    )
    tree_score = float(tree_wrapper.evaluate())
    tree_norm = tree_score / float(max(n_points, 1) ** 2)

    # Distance score (same likelihood form as Tree in current implementation)
    cov_list = [np.eye(3) * max(v, 1e-9) for v in cluster_variances]
    dist_wrapper = ScoringRestraintWrapper(
        model,
        avs,
        dataxyz=aligned,
        var=cov_list,
        model_coords_override=model_aligned,
        type="Distance",
    )
    dist_score = float(dist_wrapper.evaluate())
    dist_norm = dist_score / float(max(n_points, 1) ** 2)

    # GMM score (can fail for degenerate small/noisy clusters)
    gmm_score = np.nan
    gmm_norm = np.nan
    gmm_ok = False
    if n_points > 2:
        try:
            _, gmm_obj, gmm_mean, gmm_cov, gmm_w = test_gmm_components(
                aligned.astype(np.float64), reg_covar=1e-4
            )
            gmm_wrapper = ScoringRestraintWrapper(
                model,
                avs,
                gmm_sel_components=gmm_obj.n_components,
                gmm_sel_mean=gmm_mean,
                gmm_sel_cov=gmm_cov,
                gmm_sel_weight=gmm_w,
                model_coords_override=model_aligned,
                type="GMM",
            )
            gmm_score = float(gmm_wrapper.evaluate())
            gmm_norm = gmm_score / float(max(n_points, 1))
            gmm_ok = True
        except Exception:
            pass

    return {
        "n_points": n_points,
        "tree_score": tree_score,
        "tree_norm": tree_norm,
        "distance_score": dist_score,
        "distance_norm": dist_norm,
        "gmm_score": gmm_score,
        "gmm_norm": gmm_norm,
        "gmm_ok": gmm_ok,
        "aligned_points": aligned,
    }


def _plot_overview(points_xy, labels, top_df, best_cluster_id, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    unique = sorted([cid for cid in set(labels.tolist()) if cid != -1])
    cmap = plt.get_cmap("tab20", max(len(unique), 1))
    color_lookup = {cid: cmap(i) for i, cid in enumerate(unique)}
    colors = [(0.85, 0.85, 0.85, 0.45) if cid == -1 else color_lookup[cid] for cid in labels]

    ax.scatter(points_xy[:, 0], points_xy[:, 1], c=colors, s=2.5, linewidths=0, rasterized=True)
    ax.set_title("Candidate Clusters and Recommended Target")
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("y [nm]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, linewidth=0.4)

    # Annotate top-ranked candidates
    for _, row in top_df.head(12).iterrows():
        cid = int(row["cluster_id"])
        mask = labels == cid
        if not np.any(mask):
            continue
        c = points_xy[mask].mean(axis=0)
        ax.text(c[0], c[1], str(cid), fontsize=8, ha="center", va="center")

    best_mask = labels == int(best_cluster_id)
    if np.any(best_mask):
        best_center = points_xy[best_mask].mean(axis=0)
        ax.scatter(
            [best_center[0]],
            [best_center[1]],
            s=240,
            facecolors="none",
            edgecolors="black",
            linewidths=2.0,
            marker="o",
            label=f"Recommended target: {best_cluster_id}",
        )
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def _plot_ranking(top_df, out_path):
    show_df = top_df.head(20).copy()
    labels = [str(int(x)) for x in show_df["cluster_id"]]
    y = np.arange(len(show_df))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(y, show_df["overall_quality"], color="#4c78a8", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Overall Quality Score")
    ax.set_ylabel("Cluster ID")
    ax.set_title("Top Candidate Clusters (Higher is Better)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Rank NPC target cluster candidates by scoring + geometry (non-productive analysis)."
    )
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "pipeline_config.json"),
        help="Path to pipeline_config.json",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "figures", "target_cluster_analysis"),
        help="Directory for analysis outputs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible random filter window.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=80,
        help="Evaluate at most this many largest candidate clusters for speed.",
    )
    parser.add_argument(
        "--expected-diameter-nm",
        type=float,
        default=120.0,
        help="Expected NPC XY diameter used in geometric quality term.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    paths = cfg["paths"]
    filtering_cfg = cfg.get("filtering", {})
    clustering_cfg = cfg.get("clustering", {})

    filter_type = filtering_cfg.get("type", "none")
    cut_cfg = filtering_cfg.get("filter", {})
    x_range = tuple(cut_cfg.get("x_range")) if cut_cfg.get("x_range") else None
    y_range = tuple(cut_cfg.get("y_range")) if cut_cfg.get("y_range") else None
    z_range = tuple(cut_cfg.get("z_range")) if cut_cfg.get("z_range") else None
    percentage = filtering_cfg.get("random", {}).get("size_percentage")

    data_path = os.path.join(os.path.dirname(args.config), paths["smlm_data"])
    pdb_path = os.path.join(os.path.dirname(args.config), paths["pdb_data"])
    av_param_path = os.path.join(os.path.dirname(args.config), paths["av_parameters"])

    raw_df = read_experimental_data(data_path)
    if raw_df is None:
        raise RuntimeError(f"Failed to load SMLM data: {data_path}")

    coords, variances, _, _, applied_cuts = flexible_filter_smlm_data(
        raw_df,
        filter_type=filter_type,
        x_cut=x_range,
        y_cut=y_range,
        z_cut=z_range,
        percentage=percentage,
        fill_z_value=0.0,
        random_seed=args.seed,
        return_tree=False,
    )
    if len(coords) == 0:
        raise RuntimeError("No points after filtering. Adjust filter settings.")

    npc_results = isolate_individual_npcs(
        coords,
        min_cluster_size=int(clustering_cfg.get("min_cluster_size", 15)),
        min_npc_points=int(clustering_cfg.get("min_npc_points", 100)),
        perform_geometric_merging=bool(clustering_cfg.get("perform_geometric_merging", True)),
        debug=True,
    )
    labels = npc_results["labels"]
    all_info = npc_results["all_cluster_info"]
    candidates = [c for c in all_info if c["n_points"] >= int(clustering_cfg.get("min_npc_points", 100))]
    if not candidates:
        raise RuntimeError("No candidate clusters found with current min_npc_points.")

    # Prioritize large clusters first for compute budget.
    candidates = sorted(candidates, key=lambda c: c["n_points"], reverse=True)[: args.max_candidates]

    print(f"Preparing IMP model once for scoring {len(candidates)} candidates ...")
    params = read_parameters_from_json(av_param_path)
    avs, model, _ = compute_av(pdb_path, params)
    model_coords_nm = np.array([np.array(p.get_coordinates()) * 0.1 for p in avs], dtype=np.float64)
    model_centered_baseline = model_coords_nm - model_coords_nm.mean(axis=0)

    rows = []
    for i, candidate in enumerate(candidates, start=1):
        cid = int(candidate["cluster_id"])
        mask = labels == cid
        c_xyz = coords[mask]
        c_var = variances[mask] if variances is not None else np.ones(len(c_xyz), dtype=np.float32)
        print(f"[{i:3d}/{len(candidates):3d}] scoring cluster {cid} ({len(c_xyz)} pts)")

        score_data = _score_candidate(model, avs, model_centered_baseline, c_xyz, c_var)
        geom = _geometry_quality(
            score_data["aligned_points"],
            expected_diameter_nm=float(args.expected_diameter_nm),
            diameter_sigma_nm=45.0,
        )

        row = {
            "cluster_id": cid,
            "n_points": int(len(c_xyz)),
            "tree_score": score_data["tree_score"],
            "tree_norm": score_data["tree_norm"],
            "distance_score": score_data["distance_score"],
            "distance_norm": score_data["distance_norm"],
            "gmm_score": score_data["gmm_score"],
            "gmm_norm": score_data["gmm_norm"],
            "gmm_ok": bool(score_data["gmm_ok"]),
        }
        row.update(geom)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("n_points", ascending=False).reset_index(drop=True)
    df["tree_scaled"] = _safe_minmax(df["tree_norm"].values)
    df["distance_scaled"] = _safe_minmax(df["distance_norm"].values)
    df["gmm_scaled"] = _safe_minmax(df["gmm_norm"].values)
    df["score_quality"] = np.nanmean(
        np.vstack([df["tree_scaled"].values, df["distance_scaled"].values, df["gmm_scaled"].values]), axis=0
    )
    df["overall_quality"] = 0.6 * df["score_quality"] + 0.4 * df["geometry_quality"]
    df = df.sort_values("overall_quality", ascending=False).reset_index(drop=True)

    recommended = int(df.iloc[0]["cluster_id"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"candidate_ranking_{timestamp}.csv")
    txt_path = os.path.join(out_dir, f"candidate_summary_{timestamp}.txt")
    fig_map_path = os.path.join(out_dir, f"candidate_map_{timestamp}.png")
    fig_rank_path = os.path.join(out_dir, f"candidate_ranking_{timestamp}.png")

    df.to_csv(csv_path, index=False)
    _plot_overview(coords[:, :2], labels, df, recommended, fig_map_path)
    _plot_ranking(df, fig_rank_path)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Target cluster candidate analysis\n")
        f.write(f"Recommended target_cluster_id: {recommended}\n")
        f.write(f"Applied filter window: {applied_cuts}\n")
        f.write(f"Candidates scored: {len(df)}\n\n")
        f.write("Top 10 candidates:\n")
        for _, row in df.head(10).iterrows():
            f.write(
                f"  cluster={int(row['cluster_id'])}, n={int(row['n_points'])}, "
                f"overall={row['overall_quality']:.4f}, score={row['score_quality']:.4f}, "
                f"geom={row['geometry_quality']:.4f}\n"
            )

    print("\nAnalysis complete.")
    print(f"Recommended target_cluster_id: {recommended}")
    print(f"CSV ranking: {csv_path}")
    print(f"Summary:     {txt_path}")
    print(f"Map figure:  {fig_map_path}")
    print(f"Rank figure: {fig_rank_path}")


if __name__ == "__main__":
    main()

