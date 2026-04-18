# Contents of data_handling.py
import pathlib
import IMP
import IMP.core
import typing
import numpy as np
import pandas as pd
import sys
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN, HDBSCAN
import tqdm
import IMP.atom # Ensure IMP.atom is imported
import IMP.bff  # Ensure IMP.bff is imported


############


def _amplitude_to_variance(amplitudes):
    """Convert amplitude-derived precision proxy to scalar variance.

    Parameters
    ----------
    amplitudes : array-like
        Raw amplitude values from SMLM localization table.
        Non-positive values are clamped to 1e-9.

    Returns
    -------
    np.ndarray, dtype float32
        Variance array: sigma^2 = (1 / sqrt(amplitude))^2.
    """
    valid_amp = np.asarray(amplitudes, dtype=np.float64).copy()
    valid_amp[valid_amp <= 0] = 1e-9
    sigma = 1.0 / np.sqrt(valid_amp)
    return (sigma ** 2.0).astype(np.float32)


def check_offset_and_clustering(model_avs: typing.List, smlm_data: np.ndarray,
                                debug: bool = True) -> dict:
    """
    Comprehensive offset and clustering analysis for NPC modeling.

    This function analyzes coordinate system mismatches between IMP model and SMLM data,
    detects multiple NPC clusters, and provides recommendations for offset correction.

    Parameters:
    -----------
    model_avs : list of IMP.bff.AV
        List of Accessible Volume decorators from the IMP model
    smlm_data : np.ndarray
        SMLM coordinates array with shape (N, 3) [x, y, z]
    debug : bool
        If True, prints detailed diagnostic information

    Returns:
    --------
    dict : Analysis results containing offset and clustering information
    """

    # Extract model coordinates
    model_coords = []
    for av in model_avs:
        model_xyz = IMP.core.XYZ(av)
        coords = np.array(model_xyz.get_coordinates(), dtype=np.float64)
        model_coords.append(coords)
    model_coords = np.array(model_coords)

    if debug:
        print(f"Model coordinates shape: {model_coords.shape}")
        print(f"Model coordinate range: X[{model_coords[:, 0].min():.1f}, {model_coords[:, 0].max():.1f}], "
              f"Y[{model_coords[:, 1].min():.1f}, {model_coords[:, 1].max():.1f}], "
              f"Z[{model_coords[:, 2].min():.1f}, {model_coords[:, 2].max():.1f}]")
        print(f"SMLM data shape: {smlm_data.shape}")
        print(f"SMLM coordinate range: X[{smlm_data[:, 0].min():.1f}, {smlm_data[:, 0].max():.1f}], "
              f"Y[{smlm_data[:, 1].min():.1f}, {smlm_data[:, 1].max():.1f}], "
              f"Z[{smlm_data[:, 2].min():.1f}, {smlm_data[:, 2].max():.1f}]")

    # Unit detection and conversion
    max_coord = np.max(np.abs(smlm_data))
    if max_coord > 1000:  # Likely nanometers
        units_detected = 'nm'
        data_converted = smlm_data * 10.0  # Convert nm to Angstrom
    else:  # Likely already in Angstroms
        units_detected = 'angstrom'
        data_converted = smlm_data.copy()

    # Calculate centroids
    model_centroid = np.mean(model_coords, axis=0)
    data_centroid = np.mean(data_converted, axis=0)
    suggested_offset = model_centroid - data_centroid

    if debug:
        print(f"Model centroid: [{model_centroid[0]:.1f}, {model_centroid[1]:.1f}, {model_centroid[2]:.1f}]")
        print(f"Data centroid: [{data_centroid[0]:.1f}, {data_centroid[1]:.1f}, {data_centroid[2]:.1f}]")
        print(f"Suggested offset: [{suggested_offset[0]:.1f}, {suggested_offset[1]:.1f}, {suggested_offset[2]:.1f}]")

    # NPC clustering analysis (using 30 nm = 300 Angstrom clustering radius)
    clustering_radius_angstrom = 300.0  # 30 nm in Angstroms
    cluster_labels = np.array([])
    if len(data_converted) > 0:
        dbscan = DBSCAN(eps=clustering_radius_angstrom, min_samples=10)
        cluster_labels = dbscan.fit_predict(data_converted)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

        if debug:
            print(f"\nNPC Clustering Analysis:")
            print(f"Clustering radius: {clustering_radius_angstrom} Å ({clustering_radius_angstrom / 10:.1f} nm)")
            print(f"Number of clusters found: {n_clusters}")
            print(f"Number of noise points: {np.sum(cluster_labels == -1)}")

        # Analyze each cluster
        cluster_info = []
        optimal_cluster_idx = -1
        min_distance_to_model = float('inf')

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_points = data_converted[cluster_mask]
            cluster_centroid = np.mean(cluster_points, axis=0)
            distance_to_model = np.linalg.norm(cluster_centroid - model_centroid)

            cluster_info.append({
                'cluster_id': cluster_id,
                'n_points': np.sum(cluster_mask),
                'centroid': cluster_centroid,
                'distance_to_model': distance_to_model
            })

            if distance_to_model < min_distance_to_model:
                min_distance_to_model = distance_to_model
                optimal_cluster_idx = cluster_id

            if debug:
                print(f"Cluster {cluster_id}: {np.sum(cluster_mask)} points, "
                      f"centroid [{cluster_centroid[0]:.1f}, {cluster_centroid[1]:.1f}, {cluster_centroid[2]:.1f}], "
                      f"distance to model: {distance_to_model:.1f} Å")
    else:
        n_clusters = 0
        cluster_info = []
        optimal_cluster_idx = -1

    # Multi-radius matching analysis
    test_radii = [10, 25, 50, 100, 200, 500, 1000, 2000]  # Test radii in Angstroms
    match_analysis = []

    if len(data_converted) > 0:
        data_tree = KDTree(data_converted)

        if debug:
            print(f"\nRadius-based matching analysis:")

        for radius in test_radii:
            matches = data_tree.query_radius(model_coords, r=radius)
            total_matches = sum(len(m) for m in matches)
            avg_matches_per_av = total_matches / len(model_coords) if len(model_coords) > 0 else 0

            match_analysis.append({
                'radius_angstrom': radius,
                'radius_nm': radius / 10,
                'total_matches': total_matches,
                'avg_matches_per_av': avg_matches_per_av,
                'has_matches': total_matches > 0
            })

            if debug:
                print(f"Radius {radius:4d} Å ({radius / 10:5.1f} nm): {total_matches:6d} matches, "
                      f"{avg_matches_per_av:6.2f} avg per AV")

    # Compile results
    results = {
        'units_detected': units_detected,
        'data_converted': data_converted,
        'model_centroid': model_centroid,
        'data_centroid': data_centroid,
        'suggested_offset': suggested_offset,
        'npc_clusters': {
            'n_clusters': n_clusters,
            'cluster_info': cluster_info,
            'optimal_cluster_idx': optimal_cluster_idx,
            'cluster_labels': cluster_labels
        },
        'match_analysis': match_analysis
    }

    # Recommendations
    if debug:
        print(f"\n=== RECOMMENDATIONS ===")
        if n_clusters > 1:
            print(f"Multiple NPCs detected ({n_clusters} clusters). Consider:")
            print(f"1. Filter data to single NPC (cluster {optimal_cluster_idx})")
            print(f"2. Apply offset correction: {suggested_offset}")
            if optimal_cluster_idx >= 0:
                optimal_cluster = cluster_info[optimal_cluster_idx]
                optimal_offset = model_centroid - optimal_cluster['centroid']
                print(
                    f"3. Optimal cluster offset: [{optimal_offset[0]:.1f}, {optimal_offset[1]:.1f}, {optimal_offset[2]:.1f}]")
                results['optimal_cluster_offset'] = optimal_offset

        # Find appropriate search radius
        working_radii = [ma for ma in match_analysis if ma['has_matches']]
        if working_radii:
            recommended_radius = working_radii[0]['radius_angstrom']  # Smallest working radius
            print(f"4. Recommended search radius: {recommended_radius} Å ({recommended_radius / 10} nm)")
            results['recommended_search_radius'] = recommended_radius
        else:
            print(f"4. No matches found at any tested radius - check coordinate systems!")
            results['recommended_search_radius'] = 1000  # Default fallback

        print(f"5. Use scaling=0.1 for Tree scoring restraint")

    return results


def isolate_individual_npcs(
    smlm_data: np.ndarray,
    min_cluster_size: int = 15,
    min_npc_points: int = 50,
    use_xy_only: bool = True,
    perform_geometric_merging: bool = True,
    min_samples: int = None,
    cluster_selection_method: str = 'eom',
    debug: bool = False,
):
    """
    Isolate individual NPCs from SMLM data using HDBSCAN.
    """
    if len(smlm_data) == 0:
        return {
            'labels': np.array([], dtype=int),
            'n_npcs': 0,
            'npc_info': [],
            'probabilities': np.array([]),
            'all_cluster_info': [],
        }

    # Select clustering dimensions
    cluster_coords = smlm_data[:, :2] if use_xy_only else smlm_data

    # Guard: sklearn HDBSCAN raises if min_samples (defaults to min_cluster_size)
    # exceeds n_samples. For tiny inputs, return a deterministic all-noise result.
    n_samples = len(cluster_coords)
    if min_cluster_size > n_samples or n_samples < 2:
        return {
            'labels': np.full(n_samples, -1, dtype=int),
            'n_npcs': 0,
            'npc_info': [],
            'probabilities': np.zeros(n_samples, dtype=float),
            'all_cluster_info': [],
        }

    # Run HDBSCAN
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=0.0,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=False,
        copy=False,
    )
    labels = hdb.fit_predict(cluster_coords)
    probabilities = hdb.probabilities_

    # --- Optional MACRO-CLUSTERING: Hierarchical Geometric Merging ---
    if perform_geometric_merging:
        clean_mask = labels != -1
        if np.any(clean_mask):
            from sklearn.cluster import AgglomerativeClustering
            clean_pts = cluster_coords[clean_mask]

            # Point-wise complete-linkage agglomeration has O(n^2) memory use.
            # Keep the original behavior for small point clouds and switch to a
            # cluster-level merge strategy for large datasets.
            max_pointwise_merge_points = 5000
            if len(clean_pts) <= max_pointwise_merge_points:
                agg = AgglomerativeClustering(
                    n_clusters=None, distance_threshold=140, linkage='complete'
                )
                macro_labels = agg.fit_predict(clean_pts)

                new_labels = np.full(len(cluster_coords), -1)
                new_labels[clean_mask] = macro_labels
                labels = new_labels
            else:
                clean_cluster_ids = np.array(sorted(set(labels[clean_mask])), dtype=int)

                # Fallback safety: if there is only one clean cluster, nothing to merge.
                if len(clean_cluster_ids) > 1:
                    centroids = np.array(
                        [cluster_coords[labels == cid].mean(axis=0) for cid in clean_cluster_ids],
                        dtype=np.float64,
                    )

                    agg = AgglomerativeClustering(
                        n_clusters=None, distance_threshold=140, linkage='complete'
                    )
                    macro_cluster_labels = agg.fit_predict(centroids)

                    old_to_macro = {
                        int(old): int(new)
                        for old, new in zip(clean_cluster_ids, macro_cluster_labels)
                    }
                    new_labels = np.full(len(cluster_coords), -1, dtype=int)
                    for old_cid, new_cid in old_to_macro.items():
                        new_labels[labels == old_cid] = new_cid
                    labels = new_labels

                    if debug:
                        print(
                            "  Geometric merge switched to cluster-level mode "
                            f"({len(clean_pts)} clean points, {len(clean_cluster_ids)} clusters)."
                        )

    n_total = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))

    if debug:
        print(f"HDBSCAN: {n_total} clusters, {n_noise} noise points "
              f"(min_cluster_size={min_cluster_size})")

    # Collect info for all clusters
    all_cluster_info = []
    npc_info = []

    for cid in range(n_total):
        mask = labels == cid
        n_pts = int(np.sum(mask))
        pts = smlm_data[mask]
        centroid = pts.mean(axis=0)
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        bbox_size = bbox_max - bbox_min

        info = {
            'cluster_id': cid,
            'n_points': n_pts,
            'centroid': centroid,
            'bbox_min': bbox_min,
            'bbox_max': bbox_max,
            'bbox_size': bbox_size,
        }
        all_cluster_info.append(info)

        if n_pts >= min_npc_points:
            npc_info.append(info)

    if debug:
        print(f"  NPC-sized clusters (>={min_npc_points} pts): {len(npc_info)}")
        for npc in npc_info[:10]:
            print(f"    Cluster {npc['cluster_id']}: {npc['n_points']} pts, "
                  f"size {npc['bbox_size'][:2].astype(int)} nm")

    return {
        'labels': labels,
        'n_npcs': len(npc_info),
        'npc_info': npc_info,
        'probabilities': probabilities,
        'all_cluster_info': all_cluster_info,
    }


def filter_to_single_npc_cluster(smlm_data: np.ndarray, cluster_idx: int,
                                 cluster_radius: float = 1500.0) -> np.ndarray:
    """
    Filter SMLM data to a single NPC cluster.
    """
    if len(smlm_data) == 0:
        return smlm_data

    dbscan = DBSCAN(eps=cluster_radius, min_samples=10)
    cluster_labels = dbscan.fit_predict(smlm_data)

    if cluster_idx >= 0 and cluster_idx < len(set(cluster_labels)):
        cluster_mask = cluster_labels == cluster_idx
        return smlm_data[cluster_mask]
    else:
        print(f"Warning: Cluster {cluster_idx} not found. Returning original data.")
        return smlm_data


def align_npc_cluster_pca(smlm_cluster: np.ndarray, debug: bool = False) -> dict:
    """
    Aligns an SMLM point cloud of an NPC cluster to the XY plane (z=0) centered at the origin.
    """
    if len(smlm_cluster) < 3:
        if debug: print("Not enough points to perform PCA.")
        return {
            'aligned_data': smlm_cluster.copy(),
            'translation': np.zeros(3),
            'rotation': np.eye(3)
        }

    # Center the data
    centroid = np.mean(smlm_cluster, axis=0)
    centered_data = smlm_cluster - centroid

    # PCA for Rotation
    cov_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors_sorted = eigenvectors[:, sort_indices]
    rotation_matrix = eigenvectors_sorted.T
    
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[2, :] *= -1

    aligned_data = np.dot(centered_data, rotation_matrix.T)

    if debug:
        print("\nPCA Alignment Results:")
        print(f"Centroid Translation applied: {-centroid}")
        print(f"Eigenvalues: {eigenvalues[sort_indices]}")
        
    return {
        'aligned_data': aligned_data,
        'translation': -centroid,
        'rotation': rotation_matrix
    }


def flexible_filter_smlm_data(df,
                              filter_type='cut',
                              x_cut=None,
                              y_cut=None,
                              z_cut=None,
                              percentage=None,
                              fill_z_value=None,
                              random_seed=None,
                              return_tree=False):
    """
    Filters SMLM data and optionally creates a KD-tree.
    
    Supported filter_types:
    - 'cut': Spatial window defined by x_cut, y_cut, z_cut tuples.
    - 'percentage': Random scattering of points (subsampling).
    - 'random': A random CONTIGUOUS spatial cut of the full image.
    - 'none' or 'full': Use all data.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    # Work on a copy to avoid modifying the original DataFrame
    processed_df = df.copy()
    applied_cuts = {'x': x_cut, 'y': y_cut, 'z': z_cut}

    if filter_type == 'cut' or filter_type == 'filter':
        conditions = []
        if x_cut is not None:
            if not (isinstance(x_cut, tuple) and len(x_cut) == 2): raise ValueError("x_cut must be a tuple (min, max)")
            conditions.append((processed_df['x [nm]'] >= x_cut[0]) & (processed_df['x [nm]'] <= x_cut[1]))
        if y_cut is not None:
            if not (isinstance(y_cut, tuple) and len(y_cut) == 2): raise ValueError("y_cut must be a tuple (min, max)")
            conditions.append((processed_df['y [nm]'] >= y_cut[0]) & (processed_df['y [nm]'] <= y_cut[1]))
        if z_cut is not None and 'z [nm]' in processed_df.columns:
            if not (isinstance(z_cut, tuple) and len(z_cut) == 2): raise ValueError("z_cut must be a tuple (min, max)")
            conditions.append((processed_df['z [nm]'] >= z_cut[0]) & (processed_df['z [nm]'] <= z_cut[1]))
        elif z_cut is not None:
            print("Warning: z_cut provided, but 'z [nm]' column not found. z_cut ignored.")

        if conditions:
            final_condition = pd.Series(True, index=processed_df.index)
            for cond in conditions:
                final_condition &= cond
            processed_df = processed_df[final_condition]

    elif filter_type == 'percentage':
        if percentage is None or not (0 <= percentage <= 100):
            raise ValueError("Percentage must be between 0 and 100 for 'percentage' filter type.")
        n_samples = int(round(len(processed_df) * (percentage / 100.0)))
        if 0 < n_samples <= len(processed_df):
            processed_df = processed_df.sample(n=n_samples, random_state=random_seed, replace=False)
        elif n_samples == 0:
            processed_df = processed_df.iloc[0:0]

    elif filter_type == 'random':
        if percentage is None or not (0 < percentage <= 100):
            raise ValueError("Percentage must be between 1 and 100 for 'random' (spatial cut) filter type.")
        
        rng = np.random.RandomState(random_seed)
        clean_df = processed_df.dropna(subset=['x [nm]', 'y [nm]'])
        if clean_df.empty:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                None,
                None,
                applied_cuts,
            )

        min_x, max_x = clean_df['x [nm]'].min(), clean_df['x [nm]'].max()
        min_y, max_y = clean_df['y [nm]'].min(), clean_df['y [nm]'].max()
        
        W = max_x - min_x
        H = max_y - min_y
        
        if W <= 0 or H <= 0:
            processed_df = processed_df.sample(frac=percentage/100.0, random_state=random_seed)
        else:
            scale = np.sqrt(percentage / 100.0)
            target_w = W * scale
            target_h = H * scale
            
            x0 = rng.uniform(min_x, max_x - target_w)
            y0 = rng.uniform(min_y, max_y - target_h)
            applied_cuts = {
                'x': (float(x0), float(x0 + target_w)),
                'y': (float(y0), float(y0 + target_h)),
                'z': z_cut,
            }
            
            processed_df = processed_df[
                (processed_df['x [nm]'] >= x0) & (processed_df['x [nm]'] <= x0 + target_w) &
                (processed_df['y [nm]'] >= y0) & (processed_df['y [nm]'] <= y0 + target_h)
            ]

    elif filter_type == 'none' or filter_type == 'full':
        pass
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}.")

    # Handle z-axis filling
    if fill_z_value is not None:
        processed_df.loc[:, 'z [nm]'] = float(fill_z_value)
    elif 'z [nm]' not in processed_df.columns:
        processed_df.loc[:, 'z [nm]'] = 0.0

    # Prepare outputs
    if processed_df.empty:
        data_xyz = np.empty((0, 3), dtype=np.float32)
        variance_array = np.empty((0,), dtype=np.float32)
        return data_xyz, variance_array, None, None, applied_cuts

    data_xyz = processed_df[['x [nm]', 'y [nm]', 'z [nm]']].values.astype(np.float32)

    variance_array = np.ones(len(processed_df), dtype=np.float32)
    if 'Amplitude_0_0' in processed_df.columns:
        variance_array = _amplitude_to_variance(processed_df['Amplitude_0_0'].values)

    kdtree = None
    data_for_tree = None
    if return_tree:
        data_for_tree = data_xyz.copy()
        kdtree = KDTree(data_for_tree)

    return data_xyz, variance_array, data_for_tree, kdtree, applied_cuts


def get_held_out_complement(
    df,
    x_cut=None,
    y_cut=None,
    z_cut=None,
    fill_z_value=None,
    n_samples=500,
    random_seed=42
):
    """Returns sample of data OUTSIDE the spatial filter region."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    complement_df = df.copy()
    conds = []
    if x_cut is not None:
        conds.append((complement_df['x [nm]'] < x_cut[0]) | (complement_df['x [nm]'] > x_cut[1]))
    if y_cut is not None:
        conds.append((complement_df['y [nm]'] < y_cut[0]) | (complement_df['y [nm]'] > y_cut[1]))
    if z_cut is not None and 'z [nm]' in complement_df.columns:
        conds.append((complement_df['z [nm]'] < z_cut[0]) | (complement_df['z [nm]'] > z_cut[1]))

    if conds:
        mask = pd.Series(False, index=complement_df.index)
        for c in conds: mask |= c
        complement_df = complement_df[mask]

    if fill_z_value is not None:
        complement_df.loc[:, 'z [nm]'] = float(fill_z_value)
    elif 'z [nm]' not in complement_df.columns:
        complement_df.loc[:, 'z [nm]'] = 0.0

    if complement_df.empty:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    actual_n = min(n_samples, len(complement_df))
    sampled_df = complement_df.sample(n=actual_n, random_state=random_seed)
    
    comp_xyz = sampled_df[['x [nm]', 'y [nm]', 'z [nm]']].values.astype(np.float32)
    comp_vars = np.ones(len(sampled_df), dtype=np.float32)
    if 'Amplitude_0_0' in sampled_df.columns:
        comp_vars = _amplitude_to_variance(sampled_df['Amplitude_0_0'].values)

    return comp_xyz, comp_vars


def compute_av(pdb_datapath, parameter):
    """Compute Accessible Volumes for specified chain/residue/atom selections.

    Parameters
    ----------
    pdb_datapath : str
        Path to PDB or CIF structure file.
    parameter : dict
        Dictionary with keys 'chains', 'residue_index', 'atom_name',
        'av_parameter' (passed as kwargs to IMP.bff.AV.do_setup_particle).

    Returns
    -------
    avs_list : list of IMP.bff.AV
        Computed AV decorators.
    m : IMP.Model
        The IMP model instance.
    hier : IMP.atom.Hierarchy
        The loaded PDB hierarchy.
    """
    chains = parameter['chains']
    residue_index = parameter['residue_index']
    atom_name = parameter['atom_name']
    av_setup_params = parameter['av_parameter']

    m = IMP.Model()
    ext = str(pdb_datapath).lower()
    if ext.endswith('.cif') or ext.endswith('.mmcif'):
        hier = IMP.atom.read_mmcif(str(pathlib.Path(pdb_datapath).absolute()), m)
    else:
        hier = IMP.atom.read_pdb(str(pathlib.Path(pdb_datapath).absolute()), m, IMP.atom.CAlphaPDBSelector())

    for p_pdb in IMP.atom.get_by_type(hier, IMP.atom.ATOM_TYPE):
        IMP.core.XYZ(p_pdb).set_coordinates_are_optimized(False)

    avs_list = []
    for chain_id in tqdm.tqdm(chains, desc="AV Computation"):
        av_p = IMP.Particle(m)
        sel = IMP.atom.Selection(hier)
        sel.set_chain_id(str(chain_id))
        sel.set_atom_type(IMP.atom.AtomType(atom_name))
        sel.set_residue_index(residue_index)
        
        selected = sel.get_selected_particles()
        if not selected: continue
        
        IMP.bff.AV.do_setup_particle(m, av_p, selected[0], **av_setup_params)
        av_decorator = IMP.bff.AV(av_p)
        av_decorator.resample()
        avs_list.append(av_decorator)

    downsample_factor = parameter.get('downsample_residues_per_bead', None)
    if downsample_factor is not None and downsample_factor > 0:
        print(f"Downsampling hierarchy to {downsample_factor} AAs per bead...")
        hier = IMP.atom.create_simplified_along_backbone(hier, downsample_factor, False)

    return avs_list, m, hier


def scalar_variances_to_covariances(scalar_vars):
    """Expand scalar variances to diagonal 3x3 covariance matrices.

    Parameters
    ----------
    scalar_vars : array-like, shape (N,)
        Scalar variance values.

    Returns
    -------
    np.ndarray, shape (N, 3, 3)
        Diagonal covariance matrices.
    """
    return np.array([np.eye(3)*v for v in scalar_vars])



def isolate_npcs_from_eman2_boxes(
    smlm_data: np.ndarray,
    boxes_path: str,
    pixel_map_path: str,
    min_npc_points: int = 50,
    debug: bool = False,
):
    """
    Isolate individual NPCs from SMLM data using coordinates from EMAN2 pick results.
    """
    import json
    import os

    if not os.path.exists(boxes_path):
        raise FileNotFoundError(f"EMAN2 box file not found: {boxes_path}")
    if not os.path.exists(pixel_map_path):
        raise FileNotFoundError(f"Pixel map file not found: {pixel_map_path}")

    # Load Box Data
    with open(boxes_path, 'r') as f:
        info_data = json.load(f)
    
    # EMAN2 format handling (same logic as reverse translation script)
    boxes = info_data.get('boxes', [])
    if not boxes and isinstance(info_data, list):
        boxes = info_data
    elif not boxes:
        for k in info_data.keys():
            if 'boxes' in k.lower():
                boxes = info_data[k]
                break
    
    if not boxes:
        print("Warning: No boxes found in EMAN2 JSON.")
        return {
            'labels': np.full(len(smlm_data), -1, dtype=int),
            'n_npcs': 0,
            'npc_info': [],
            'probabilities': np.zeros(len(smlm_data), dtype=float),
            'all_cluster_info': [],
        }

    # Load Pixel Map
    with open(pixel_map_path, 'r') as f:
        pixel_map = json.load(f)
    
    pixel_size_nm = pixel_map['pixel_size_nm']
    
    # Prepare results
    labels = np.full(len(smlm_data), -1, dtype=int)
    npc_info = []
    all_cluster_info = []

    # EMAN2 usually stores a global box size in the project.json or top level of info
    project_box_size = info_data.get('global.boxsize', 32)

    for i, box in enumerate(boxes):
        px_x, px_y = box[0], box[1]
        box_size_px = project_box_size
        
        half_size_nm = (box_size_px / 2.0) * pixel_size_nm
        x_min, x_max = (px_x * pixel_size_nm) - half_size_nm, (px_x * pixel_size_nm) + half_size_nm
        y_min, y_max = (px_y * pixel_size_nm) - half_size_nm, (px_y * pixel_size_nm) + half_size_nm
        
        # Filter points (faster vectorization)
        mask = (smlm_data[:, 0] >= x_min) & (smlm_data[:, 0] <= x_max) & \
               (smlm_data[:, 1] >= y_min) & (smlm_data[:, 1] <= y_max)
        
        labels[mask] = i
        n_pts = int(np.sum(mask))
        pts = smlm_data[mask]
        
        if n_pts > 0:
            centroid = pts.mean(axis=0)
            bbox_min, bbox_max = pts.min(axis=0), pts.max(axis=0)
            
            info = {
                'cluster_id': i,
                'n_points': n_pts,
                'centroid': centroid,
                'bbox_min': bbox_min,
                'bbox_max': bbox_max,
                'bbox_size': bbox_max - bbox_min,
            }
            all_cluster_info.append(info)
            if n_pts >= min_npc_points:
                npc_info.append(info)

    if debug:
        print(f"  EMAN2 Picking: Found {len(boxes)} boxes. {len(npc_info)} clusters match min_points filter.")

    return {
        'labels': labels,
        'n_npcs': len(npc_info),
        'npc_info': npc_info,
        'probabilities': np.ones(len(smlm_data), dtype=float), # Confidence is 1 for assigned points
        'all_cluster_info': all_cluster_info,
    }
