import os
import typing

import IMP
import IMP.algebra
import IMP.bff
import IMP.core
import IMP.pmi.restraints
import IMP.pmi.tools
import numpy as np

from smlm_score.imp_modeling.scoring.distance_score import (
    _compute_distance_score_and_grad_cpu,
    computescoresimple,
)
from smlm_score.imp_modeling.scoring.gmm_score import compute_nb_gmm
from smlm_score.imp_modeling.scoring.tree_score import (
    computescoretree,
    computescoretree_with_grad,
)


def _extract_live_model_coords(avs_decorators, scaling, offsetxyz):
    """Extract current model AV coordinates, scaled and offset to data units."""
    modelxyzs = np.array(
        [
            np.array(IMP.core.XYZ(av).get_coordinates(), dtype=np.float64)
            for av in avs_decorators
        ]
    )
    modelxyzs = modelxyzs * scaling
    if offsetxyz is not None:
        modelxyzs = modelxyzs + offsetxyz
    return modelxyzs


def _resolve_model_coords(
    avs_decorators,
    scaling,
    offsetxyz,
    model_coords_override,
    reference_model_coords_nm,
):
    """Resolve model coordinates, tracking BD/CG-induced deltas from reference."""
    live_model_coords = _extract_live_model_coords(avs_decorators, scaling, offsetxyz)
    if model_coords_override is None:
        return live_model_coords

    override_coords = np.array(model_coords_override, dtype=np.float64)
    if reference_model_coords_nm is None:
        return override_coords

    delta = live_model_coords - reference_model_coords_nm
    return override_coords + delta


class _ScoringRestraintBase(IMP.Restraint):
    """Abstract base class for all SMLM scoring restraints.

    Manages coordinate extraction, override tracking, and sign flipping
    between raw score mode and IMP objective mode.
    """
    def __init__(
        self,
        m,
        name,
        avs,
        scaling=0.1,
        offsetxyz=None,
        model_coords_override=None,
    ):
        super().__init__(m, name)
        self.avs_decorators = avs
        self.scaling = scaling
        self.offsetxyz = offsetxyz
        self.model_coords_override = model_coords_override
        self.reference_model_coords_nm = (
            _extract_live_model_coords(avs, scaling, offsetxyz)
            if model_coords_override is not None
            else None
        )
        self.return_objective = False

    def _current_model_coords(self):
        return _resolve_model_coords(
            self.avs_decorators,
            self.scaling,
            self.offsetxyz,
            self.model_coords_override,
            self.reference_model_coords_nm,
        )

    def _score_sign(self):
        return -1.0 if self.return_objective else 1.0

    def _imp_derivative_sign(self):
        # IMP optimizers interpret stored derivatives as the direction that
        # decreases the returned score, so we store the mathematical
        # gradient of whatever value unprotected_evaluate() returns.
        return self._score_sign()

    def set_return_objective(self, enabled: bool):
        self.return_objective = enabled

    def do_get_inputs(self):
        return [av.get_particle() for av in self.avs_decorators]

    def get_particle_decorators(self):
        return self.avs_decorators


class ScoringRestraintGMM(_ScoringRestraintBase):
    """IMP Restraint backed by Gaussian Mixture Model log-likelihood."""
    def __init__(
        self,
        m,
        avs: typing.List[IMP.bff.AV],
        gmm_sel_components,
        gmm_sel_mean,
        gmm_sel_cov,
        gmm_sel_weight,
        scaling=0.1,
        offsetxyz=None,
        model_coords_override=None,
    ):
        super().__init__(
            m,
            "GMMScoringRestraint",
            avs,
            scaling=scaling,
            offsetxyz=offsetxyz,
            model_coords_override=model_coords_override,
        )
        self.gmm_sel_mean = gmm_sel_mean
        self.gmm_sel_cov = gmm_sel_cov
        self.gmm_sel_weight = gmm_sel_weight

    def unprotected_evaluate(self, da):
        del da
        model_xyzs = self._current_model_coords()
        score = compute_nb_gmm(
            model_xyzs,
            self.gmm_sel_mean,
            self.gmm_sel_cov,
            self.gmm_sel_weight,
        )
        return self._score_sign() * score

    def get_output(self):
        return {"GMMScoringRestraint_Score": str(self.unprotected_evaluate(None))}


class ScoringRestraintTree(_ScoringRestraintBase):
    """IMP Restraint backed by KD-tree log-likelihood with gradient support."""
    def __init__(
        self,
        m,
        avs: typing.List[IMP.bff.AV],
        kdtree_obj,
        dataxyz_for_tree,
        var_for_tree,
        scaling,
        searchradius,
        offsetxyz,
        model_coords_override=None,
    ):
        super().__init__(
            m,
            "TreeScoringRestraint",
            avs,
            scaling=scaling,
            offsetxyz=offsetxyz,
            model_coords_override=model_coords_override,
        )
        self.kdtree_obj = kdtree_obj
        self.dataxyz_for_tree = dataxyz_for_tree
        self.var_for_tree = var_for_tree
        self.searchradius = searchradius

    def unprotected_evaluate(self, da):
        current_model = self._current_model_coords()
        sign = self._score_sign()

        if da is not None:
            score, grad = computescoretree_with_grad(
                self.kdtree_obj,
                self.avs_decorators,
                self.dataxyz_for_tree,
                self.var_for_tree,
                self.scaling,
                self.searchradius,
                self.offsetxyz,
                model_coords_override=current_model,
            )
            derivative_scale = self.scaling
            derivative_sign = self._imp_derivative_sign()
            for i, av in enumerate(self.avs_decorators):
                xyz = IMP.core.XYZ(av)
                grad_vec = IMP.algebra.Vector3D(
                    derivative_sign * grad[i][0] * derivative_scale,
                    derivative_sign * grad[i][1] * derivative_scale,
                    derivative_sign * grad[i][2] * derivative_scale,
                )
                xyz.add_to_derivatives(grad_vec, da)
            return sign * score

        score = computescoretree(
            self.kdtree_obj,
            self.avs_decorators,
            self.dataxyz_for_tree,
            self.var_for_tree,
            self.scaling,
            self.searchradius,
            self.offsetxyz,
            model_coords_override=current_model,
        )
        return sign * score

    def get_output(self):
        return {"TreeScoringRestraint_Score": str(self.unprotected_evaluate(None))}


class ScoringRestraintDistance(_ScoringRestraintBase):
    """IMP Restraint backed by pairwise distance log-likelihood with gradients."""
    def __init__(
        self,
        m,
        avs: typing.List[IMP.bff.AV],
        dataxyz_for_distance,
        var_for_distance,
        weights_for_distance=None,
        scaling=0.1,
        offsetxyz=None,
        model_coords_override=None,
    ):
        super().__init__(
            m,
            "DistanceScoringRestraint",
            avs,
            scaling=scaling,
            offsetxyz=offsetxyz,
            model_coords_override=model_coords_override,
        )
        self.dataxyz_for_distance = dataxyz_for_distance
        self.var_for_distance = var_for_distance
        self.weights_for_distance = weights_for_distance

    def unprotected_evaluate(self, da):
        modelxyzs = self._current_model_coords()
        sign = self._score_sign()

        if da is not None:
            datacov_arr = np.asarray(self.var_for_distance, dtype=np.float64)
            weights = (
                self.weights_for_distance
                if self.weights_for_distance is not None
                else np.ones(len(self.dataxyz_for_distance))
            )
            sigmaav = 8.0

            score, grad = _compute_distance_score_and_grad_cpu(
                self.dataxyz_for_distance,
                datacov_arr,
                weights,
                modelxyzs,
                sigmaav,
            )
            derivative_scale = self.scaling
            derivative_sign = self._imp_derivative_sign()
            for i, av in enumerate(self.avs_decorators):
                xyz = IMP.core.XYZ(av)
                grad_vec = IMP.algebra.Vector3D(
                    derivative_sign * grad[i][0] * derivative_scale,
                    derivative_sign * grad[i][1] * derivative_scale,
                    derivative_sign * grad[i][2] * derivative_scale,
                )
                xyz.add_to_derivatives(grad_vec, da)
            return sign * score

        score = computescoresimple(
            self.avs_decorators,
            self.dataxyz_for_distance,
            self.var_for_distance,
            self.weights_for_distance,
            scaling=self.scaling,
            offsetxyz=self.offsetxyz,
            model_coords_override=modelxyzs,
        )
        return sign * score

    def get_output(self):
        return {"DistanceScoringRestraint_Score": str(self.unprotected_evaluate(None))}


class ScoringRestraintWrapper(IMP.pmi.restraints.RestraintBase):
    """PMI-compatible factory that dispatches to Tree, GMM, or Distance restraints.

    Provides a unified interface for the pipeline to create, evaluate, and
    register scoring restraints without knowing the underlying type.
    """
    def __init__(
        self,
        m: IMP.Model,
        avs: typing.List[IMP.bff.AV],
        gmm_sel_components=None,
        gmm_sel_mean=None,
        gmm_sel_cov=None,
        gmm_sel_weight=None,
        dataxyz=None,
        var=None,
        kdtree_obj=None,
        weights=None,
        scaling: float = 0.1,
        searchradius: float = 10.0,
        offsetxyz: np.ndarray = None,
        model_coords_override: np.ndarray = None,
        type: str = "GMM",
        label: str = "Scoring Restraint",
    ):
        super().__init__(m, label=label)
        self.type = type
        self.avs_decorators = avs
        self.scoring_restraint_instance = None

        if type == "Distance":
            self.scoring_restraint_instance = ScoringRestraintDistance(
                m,
                avs,
                dataxyz,
                var,
                weights,
                scaling=scaling,
                offsetxyz=offsetxyz,
                model_coords_override=model_coords_override,
            )
        elif type == "Tree":
            self.scoring_restraint_instance = ScoringRestraintTree(
                m,
                avs,
                kdtree_obj,
                dataxyz,
                var,
                scaling,
                searchradius,
                offsetxyz,
                model_coords_override=model_coords_override,
            )
        elif type == "GMM":
            self.scoring_restraint_instance = ScoringRestraintGMM(
                m,
                avs,
                gmm_sel_components,
                gmm_sel_mean,
                gmm_sel_cov,
                gmm_sel_weight,
                scaling=scaling,
                offsetxyz=offsetxyz,
                model_coords_override=model_coords_override,
            )
        else:
            raise ValueError(
                f"Unknown restraint type: {type}. Must be 'Distance', 'Tree', or 'GMM'."
            )

        if self.scoring_restraint_instance is not None:
            self.rs.add_restraint(self.scoring_restraint_instance)

    def evaluate(self):
        return self.rs.unprotected_evaluate(None)

    def set_return_objective(self, enabled: bool):
        if self.scoring_restraint_instance is not None:
            self.scoring_restraint_instance.set_return_objective(enabled)

    def _log_trajectory_frame(self, frame_idx, score):
        """Side-channel log of raw and aligned coordinates for debugging."""
        if not hasattr(self, "_trajectory_csv_path"):
            return

        import csv
        file_exists = os.path.exists(self._trajectory_csv_path)
        
        # 1. Extract Current Coordinates
        # Raw IMP space (Angstroms)
        raw_coords = [np.array(IMP.core.XYZ(av).get_coordinates()) for av in self.avs_decorators]
        # Aligned Data space (nm)
        data_coords = self.scoring_restraint_instance._current_model_coords()
        
        # 2. Metadata: Centroid and RMSD
        centroid_raw = np.mean(raw_coords, axis=0)
        centroid_data = np.mean(data_coords, axis=0)
        
        if self._initial_raw_coords is None:
            self._initial_raw_coords = raw_coords

        rmsd = np.sqrt(np.mean(np.sum((np.array(raw_coords) - np.array(self._initial_raw_coords))**2, axis=1)))

        with open(self._trajectory_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write Header if new file
            if not file_exists:
                header = ["frame", "score", "rmsd_raw", "centroid_raw_x", "centroid_raw_y", "centroid_raw_z", 
                          "centroid_data_x", "centroid_data_y", "centroid_data_z"]
                for i in range(len(raw_coords)):
                    header += [f"av_{i}_raw_x", f"av_{i}_raw_y", f"av_{i}_raw_z"]
                for i in range(len(data_coords)):
                    header += [f"av_{i}_data_x", f"av_{i}_data_y", f"av_{i}_data_z"]
                writer.writerow(header)

            # Write Data Row
            row = [frame_idx, score, rmsd, *centroid_raw, *centroid_data]
            for c in raw_coords: row.extend(c)
            for c in data_coords: row.extend(c)
            writer.writerow(row)

    def enable_trajectory_logging(self, output_dir):
        """Initialize the trajectory logging state."""
        self._trajectory_csv_path = os.path.join(output_dir, "trajectory_trace.csv")
        self._trajectory_frame_counter = 0
        self._initial_raw_coords = None
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Clear existing file if any
        if os.path.exists(self._trajectory_csv_path):
            os.remove(self._trajectory_csv_path)

    def get_output(self):
        if self.scoring_restraint_instance:
            score = self.scoring_restraint_instance.unprotected_evaluate(None)
            # Inject trajectory logging hook
            if hasattr(self, "_trajectory_csv_path"):
                self._log_trajectory_frame(self._trajectory_frame_counter, score)
                self._trajectory_frame_counter += 1
            
            return {f"{self.type}ScoringRestraint_Score": str(score)}
        return {}
