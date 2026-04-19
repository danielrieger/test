import json
import types
from pathlib import Path
import uuid
import shutil

import numpy as np
import pandas as pd
import pytest

from smlm_score.utility import input as input_mod
from smlm_score.utility import data_handling
from smlm_score.imp_modeling.model_setup import model as model_mod
from smlm_score.imp_modeling.scoring.gmm_score import test_gmm_components as run_test_gmm_components
from smlm_score.imp_modeling.simulation import mcmc_sampler
from smlm_score.validation.validation import run_full_validation


@pytest.mark.integration
def test_stage1_input_integration_reads_csv_and_json():
    tmp_dir = Path("smlm_score") / "tests" / f".tmp_stage1_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        csv_path = tmp_dir / "data.csv"
        pd.DataFrame({"Amplitude_0_0": [1.0, 4.0], "x [nm]": [0.0, 1.0]}).to_csv(csv_path, index=False)
        df = input_mod.read_experimental_data(str(csv_path))
        assert "precision" in df.columns
        assert "variance" in df.columns
        assert np.allclose(df["precision"].to_numpy(), np.array([1.0, 0.5]))
        assert np.allclose(df["variance"].to_numpy(), np.array([1.0, 0.25]))

        json_path = tmp_dir / "params.json"
        json_path.write_text(
            json.dumps(
                {
                    "chains": ["A"],
                    "residue_index": 1,
                    "atom_name": "CA",
                    "av_parameter": {"radii": [1.0, 2.0, 3.0]},
                }
            ),
            encoding="utf-8",
        )
        p = input_mod.read_parameters_from_json(str(json_path))
        assert isinstance(p["av_parameter"]["radii"], tuple)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.integration
def test_stage2_model_construction_integration_initialize_without_paths():
    m = model_mod.Model()
    m.initialize()
    assert m.model is not None
    assert m.p_root is not None
    assert m.h_root is not None


@pytest.mark.integration
def test_stage3_experimental_data_integration_held_out_complement():
    df = pd.DataFrame(
        {
            "x [nm]": [0.0, 10.0, 20.0, 30.0],
            "y [nm]": [0.0, 10.0, 20.0, 30.0],
            "z [nm]": [0.0, 0.0, 0.0, 0.0],
            "Amplitude_0_0": [1.0, 4.0, 9.0, 16.0],
        }
    )
    xyz, var = data_handling.get_held_out_complement(df, x_cut=(5, 25), y_cut=(5, 25), n_samples=10)
    assert xyz.shape[0] == 2
    assert var.shape[0] == 2
    assert np.allclose(np.sort(var), np.array([1.0 / 16.0, 1.0], dtype=np.float32))


@pytest.mark.integration
def test_stage3_experimental_data_integration_held_out_complement_fills_missing_z():
    df = pd.DataFrame(
        {
            "x [nm]": [0.0, 10.0, 20.0, 30.0],
            "y [nm]": [0.0, 10.0, 20.0, 30.0],
            "Amplitude_0_0": [1.0, 4.0, 9.0, 16.0],
        }
    )
    xyz, var = data_handling.get_held_out_complement(
        df, x_cut=(5, 25), y_cut=(5, 25), fill_z_value=0.0, n_samples=10
    )
    assert xyz.shape == (2, 3)
    assert np.allclose(xyz[:, 2], 0.0)
    assert var.shape[0] == 2


@pytest.mark.integration
def test_stage6_scoring_integration_test_gmm_components_runs():
    rng = np.random.RandomState(0)
    data = rng.normal(size=(32, 3))
    result, gmm_sel, mean, cov, weight = run_test_gmm_components(data, component_min=1, component_max=8)
    assert len(result["n_components"]) >= 1
    assert gmm_sel.n_components in result["n_components"]
    assert mean.shape[1] == 3
    assert cov.shape[1:] == (3, 3)
    assert np.isclose(np.sum(weight), 1.0)


@pytest.mark.integration
def test_stage7_sampling_integration_bayesian_sampler_flow(monkeypatch):
    base_tmp = Path("smlm_score") / "tests" / f".tmp_stage7_{uuid.uuid4().hex}"
    output_dir = base_tmp / "bayes_out"
    base_tmp.mkdir(parents=True, exist_ok=True)
    calls = {"mmcif": None, "added": False, "objective": None}

    class FakeDOF:
        def __init__(self, model):
            self.model = model

        def create_rigid_body(self, particles, name, **kwargs):
            self.particles = particles
            self.name = name
            self.kwargs = kwargs

        def get_movers(self):
            return ["mover"]

    class FakeReplicaExchange:
        def __init__(self, *args, **kwargs):
            self.replica_exchange_object_cif = False
            self.output_dir = Path(kwargs["global_output_directory"])
            self.stat_path = self.output_dir / "stat.0.out"
            calls["mmcif"] = kwargs.get("mmcif")

        def execute_macro(self):
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.stat_path.write_text("frame0\nframe1\n", encoding="utf-8")

    class FakeMass:
        @staticmethod
        def get_is_setup(_p):
            return True
        @staticmethod
        def setup_particle(_p, _m):
            return None

    class FakeChainDecorator:
        def __init__(self, p):
            self.p = p
        @staticmethod
        def get_is_setup(_p):
            return True
        def set_id(self, _new):
            return None

    class FakeXYZ:
        @staticmethod
        def get_is_setup(_p):
            return True
        def __init__(self, _p):
            pass
        def get_coordinates(self):
            return [0, 0, 0]
        def get_x(self): return 0
        def get_y(self): return 0
        def get_z(self): return 0

    class FakeXYZR:
        @staticmethod
        def get_is_setup(_p):
            return False
        @staticmethod
        def setup_particle(_p, _r):
            return None

    class FakeRMF:
        @staticmethod
        def create_rmf_file(_path):
            return FakeRMFFile()
        @staticmethod
        def ParticleFactory(_rmf):
            return FakeFactory()
        @staticmethod
        def ColoredFactory(_rmf):
            return FakeFactory()
        class Vector3:
            def __init__(self, x, y, z):
                pass
        FRAME = "frame"
        REPRESENTATION = "repr"

    class FakeRMFFile:
        def set_description(self, _d): pass
        def get_root_node(self): return FakeNode()
        def add_frame(self, _n, _t): pass

    class FakeNode:
        def add_child(self, _n, _t): return FakeNode()
        def get_name(self): return "leaf"

    class FakeFactory:
        def get(self, _node): return FakeParticle()

    class FakeParticle:
        def set_mass(self, _m): pass
        def set_radius(self, _r): pass
        def set_rgb_color(self, _c): pass
        def set_coordinates(self, _c): pass

    fake_imp = types.SimpleNamespace(
        Model=object,
        pmi=types.SimpleNamespace(
            dof=types.SimpleNamespace(DegreesOfFreedom=FakeDOF),
            macros=types.SimpleNamespace(ReplicaExchange=FakeReplicaExchange),
        ),
        atom=types.SimpleNamespace(
            Mass=FakeMass,
            get_by_type=lambda _h, _t: [object()],
            CHAIN_TYPE=1,
            Chain=FakeChainDecorator,
            get_leaves=lambda _h: [FakeNode()],
        ),
        core=types.SimpleNamespace(
            XYZ=FakeXYZ,
            XYZR=FakeXYZR,
        ),
    )

    monkeypatch.setattr(mcmc_sampler, "IMP", fake_imp)
    monkeypatch.setattr(mcmc_sampler, "RMF", FakeRMF)

    class FakeAV:
        def __init__(self):
            self.p = object()

        def get_particle(self):
            return self.p

    class FakeScoringWrapper:
        def set_return_objective(self, enabled):
            calls["objective"] = enabled

        def add_to_model(self):
            calls["added"] = True

    try:
        mcmc_sampler.run_bayesian_sampling(
            model=object(),
            pdb_hierarchy=object(),
            avs=[FakeAV(), FakeAV()],
            scoring_restraint_wrapper=FakeScoringWrapper(),
            output_dir=str(output_dir),
            number_of_frames=2,
            monte_carlo_steps=1,
        )

        assert output_dir.exists()
        assert (output_dir / "stat.0.out").exists()
        assert calls["added"] is True
        assert calls["objective"] is True
        assert calls["mmcif"] is True
    finally:
        shutil.rmtree(base_tmp, ignore_errors=True)


@pytest.mark.integration
def test_stage8_validation_integration_run_full_validation():
    cluster_scores = {
        0: {"type": "Valid", "n_points": 100, "Tree": -100.0, "GMM": -50.0, "Distance": -70.0},
        1: {"type": "Noise", "n_points": 100, "Tree": -500.0, "GMM": -200.0, "Distance": -300.0},
    }
    held_out = {
        "Tree": {
            "valid_score": -100.0,
            "valid_n_points": 100,
            "held_out_scores": [0.0, 0.0],
            "held_out_n_points": [50, 50],
        }
    }
    results = run_full_validation(
        cluster_scores=cluster_scores,
        held_out_results=held_out,
        scoring_types=["Tree", "GMM", "Distance"]
    )
    assert len(results) >= 4
    assert any(r.test_name == "HeldOut_Tree" for r in results)
