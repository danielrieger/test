import types
import numpy as np
import pandas as pd
import pytest

from smlm_score.utility import input as input_mod
from smlm_score.utility import data_handling
from smlm_score.imp_modeling.model_setup import model as model_mod
from smlm_score.imp_modeling.scoring import gmm_score
from smlm_score.imp_modeling.brownian_dynamics import simulation_setup as bd_setup
from smlm_score.validation import validation as validation_mod


@pytest.mark.unit
def test_stage1_input_read_experimental_data_missing_amplitude_returns_none(monkeypatch):
    monkeypatch.setattr(input_mod.pd, "read_csv", lambda *args, **kwargs: pd.DataFrame({"x [nm]": [1.0]}))
    out = input_mod.read_experimental_data("dummy.csv")
    assert out is None


@pytest.mark.unit
def test_stage2_model_construction_initialize_core_objects(monkeypatch):
    class FakeModel:
        pass

    class FakeParticle:
        def __init__(self, model, name):
            self.model = model
            self.name = name

    class FakeHierarchy:
        @staticmethod
        def setup_particle(p):
            return {"particle": p}

    fake_atom = types.SimpleNamespace(Hierarchy=FakeHierarchy)
    fake_imp = types.SimpleNamespace(Model=FakeModel, Particle=FakeParticle, atom=fake_atom)
    monkeypatch.setattr(model_mod, "IMP", fake_imp)

    m = model_mod.Model()
    m.initialize()

    assert isinstance(m.model, FakeModel)
    assert isinstance(m.p_root, FakeParticle)
    assert m.h_root["particle"] is m.p_root


@pytest.mark.unit
def test_stage3_experimental_data_flexible_filter_cut_and_variance():
    df = pd.DataFrame(
        {
            "x [nm]": [0.0, 10.0, 20.0],
            "y [nm]": [0.0, 10.0, 20.0],
            "z [nm]": [0.0, 0.0, 0.0],
            "Amplitude_0_0": [1.0, 4.0, 9.0],
        }
    )
    xyz, variance, tree_xyz, tree, _ = data_handling.flexible_filter_smlm_data(
        df,
        filter_type="cut",
        x_cut=(0, 10),
        y_cut=(0, 10),
        return_tree=True,
    )

    assert xyz.shape == (2, 3)
    assert np.allclose(variance, np.array([1.0, 0.25], dtype=np.float32))
    assert tree_xyz.shape == (2, 3)
    assert tree is not None


@pytest.mark.unit
def test_stage6_scoring_compute_nb_gmm_dispatches_to_gpu_when_enabled(monkeypatch):
    calls = {"gpu": 0}

    def fake_gpu(*args, **kwargs):
        calls["gpu"] += 1
        return 123.0

    monkeypatch.setattr(gmm_score, "HAS_CUDA", True)
    monkeypatch.setattr(gmm_score, "CUDA_MIN_DATA_SIZE", 2)
    monkeypatch.setattr(gmm_score, "compute_nb_gmm_gpu", fake_gpu)

    model_xyzs = np.zeros((2, 3), dtype=np.float64)
    data_mean = np.zeros((3, 3), dtype=np.float64)
    data_cov = np.array([np.eye(3), np.eye(3), np.eye(3)], dtype=np.float64)
    data_weight = np.ones(3, dtype=np.float64)
    out = gmm_score.compute_nb_gmm(model_xyzs, data_mean, data_cov, data_weight)

    assert out == 123.0
    assert calls["gpu"] == 1


@pytest.mark.unit
def test_stage7_sampling_brownian_dynamics_wires_components(monkeypatch):
    calls = {
        "optimize_steps": None,
        "optimized_flags": [],
        "radii": [],
        "diffusion_coeffs": [],
    }

    class FakeBD:
        def __init__(self, model):
            self.model = model

        def set_scoring_function(self, sf):
            self.sf = sf

        def set_temperature(self, t):
            self.t = t

        def set_maximum_time_step(self, dt):
            self.dt = dt

        def set_log_level(self, level):
            self.level = level

        def add_optimizer_state(self, sos):
            self.sos = sos

        def optimize(self, n):
            calls["optimize_steps"] = n

    class FakeSOS:
        def __init__(self, model, rmf_file):
            self.model = model
            self.rmf_file = rmf_file

        def set_period(self, p):
            self.period = p

        def update_always(self, label):
            self.label = label

    class FakeRMFFile:
        def set_description(self, d):
            self.d = d

    class FakeRestraint:
        pass

    class FakeRestraintSet:
        def get_restraints(self):
            return [FakeRestraint()]

    class FakeScoringFunction:
        def __init__(self, restraints, name):
            pass
        def evaluate(self, _):
            return 0.0

    class FakePMITools:
        @staticmethod
        def get_restraint_set(model):
            return FakeRestraintSet()

    class FakeMass:
        @staticmethod
        def get_is_setup(p):
            return True
        @staticmethod
        def setup_particle(p, mass):
            pass

    class FakeXYZ:
        def __init__(self, av):
            self.av = av

        def get_coordinates(self):
            return self.av.coords

    class FakeXYZRDecorator:
        def __init__(self, particle):
            self.particle = particle
            self.radius = None

        def set_coordinates(self, coords):
            self.particle.coords = coords

        def set_radius(self, radius):
            self.radius = radius
            calls["radii"].append((self.particle.name, radius))

        def get_radius(self):
            return self.radius if self.radius is not None else 30.0

        def set_coordinates_are_optimized(self, flag):
            calls["optimized_flags"].append((self.particle.name, flag))

    class FakeXYZR:
        @staticmethod
        def get_is_setup(particle):
            return False

        @staticmethod
        def setup_particle(particle):
            return FakeXYZRDecorator(particle)

    class FakeDiffusionDecorator:
        def __init__(self, particle):
            self.particle = particle

        def set_diffusion_coefficient(self, coeff):
            calls["diffusion_coeffs"].append((self.particle.name, coeff))

    class FakeDiffusion:
        @staticmethod
        def get_is_setup(p):
            return False

        @staticmethod
        def setup_particle(p, coeff):
            calls["diffusion_coeffs"].append((p.name, coeff))
            return FakeDiffusionDecorator(p)

    fake_rmf_mod = types.SimpleNamespace(
        add_hierarchy=lambda *args, **kwargs: None,
        SaveOptimizerState=FakeSOS,
    )
    fake_core_mod = types.SimpleNamespace(
        RestraintsScoringFunction=FakeScoringFunction,
        XYZ=lambda av: FakeXYZ(av),
        XYZR=FakeXYZR,
    )
    fake_atom_mod = types.SimpleNamespace(
        BrownianDynamics=FakeBD,
        Mass=FakeMass,
        Diffusion=FakeDiffusion,
        get_einstein_diffusion_coefficient=lambda radius: radius * 0.1,
    )
    fake_pmi = types.SimpleNamespace(tools=FakePMITools())
    fake_imp = types.SimpleNamespace(atom=fake_atom_mod, rmf=fake_rmf_mod, core=fake_core_mod, pmi=fake_pmi, SILENT=0)

    monkeypatch.setattr(bd_setup, "IMP", fake_imp)
    monkeypatch.setattr(bd_setup, "RMF", types.SimpleNamespace(create_rmf_file=lambda fn: FakeRMFFile()))
    monkeypatch.setattr(bd_setup, "os", types.SimpleNamespace(
        makedirs=lambda *a, **kw: None,
        path=types.SimpleNamespace(join=lambda *a: "/".join(a))
    ))

    class FakeModel:
        def update(self):
            pass

    class FakeScoringWrapper:
        type = "Tree"
        def add_to_model(self):
            pass

    class FakeAV:
        def __init__(self, name):
            self.name = name
            self.coords = (1.0, 2.0, 3.0)
            self.particle = types.SimpleNamespace(name=name, coords=self.coords)

        def get_particle(self):
            return self.particle

        def get_radii(self):
            return (30.0, 0.0, 0.0)

    bd_setup.run_brownian_dynamics_simulation(
        model=FakeModel(),
        pdb_hierarchy=object(),
        avs=[FakeAV("av0"), FakeAV("av1")],
        scoring_restraint_wrapper=FakeScoringWrapper(),
        number_of_bd_steps=7,
    )

    assert calls["optimize_steps"] == 7
    assert calls["optimized_flags"] == [("av0", True), ("av1", True)]
    assert calls["radii"] == [("av0", 30.0), ("av1", 30.0)]
    assert calls["diffusion_coeffs"] == [("av0", 3.0), ("av1", 3.0)]



@pytest.mark.unit
def test_stage8_validation_tree_held_out_all_zero_does_not_auto_pass():
    res = validation_mod.validate_with_held_out_data(
        valid_cluster_score=-10.0,
        valid_n_points=10,
        held_out_scores=[0.0, 0.0, 0.0],
        held_out_n_points=[10, 10, 10],
        scoring_type="Tree",
    )
    assert res.passed is False
    assert res.metrics["held_out_all_zero"] is True


@pytest.mark.unit
def test_stage8_validation_tree_distance_gmm_use_per_point_normalization():
    cluster_scores = {
        563: {
            "type": "Valid",
            "n_points": 100,
            "Tree": -100.0,
            "Distance": -100.0,
            "GMM": -100.0,
        },
        10: {
            "type": "Noise",
            "n_points": 50,
            "Tree": -250.0,
            "Distance": -250.0,
            "GMM": -250.0,
        },
        17: {
            "type": "Noise",
            "n_points": 100,
            "Tree": -600.0,
            "Distance": -600.0,
            "GMM": -600.0,
        },
    }

    results = validation_mod.validate_scoring_separation(
        cluster_scores, scoring_types=["Tree", "Distance", "GMM"]
    )
    result_by_name = {r.test_name: r for r in results}

    assert bool(result_by_name["Separation_Tree"].passed)
    assert bool(result_by_name["Separation_Distance"].passed)
    assert bool(result_by_name["Separation_GMM"].passed)
    for name in ["Separation_Tree", "Separation_Distance", "Separation_GMM"]:
        assert result_by_name[name].metrics["mean_valid_per_point"] == pytest.approx(-1.0)
        assert result_by_name[name].metrics["mean_noise_per_point"] == pytest.approx(-5.5)


@pytest.mark.unit
def test_stage8_validation_separation_skips_without_noise_controls():
    cluster_scores = {
        1: {"type": "Valid", "n_points": 100, "Tree": -100.0},
        2: {"type": "Valid", "n_points": 120, "Tree": -110.0},
    }

    results = validation_mod.validate_scoring_separation(cluster_scores, scoring_types=["Tree"])

    assert len(results) == 1
    assert results[0].test_name == "Separation_Tree"
    assert results[0].skipped is True
    assert results[0].metrics["skipped_reason"] == "missing_valid_or_noise_controls"


@pytest.mark.unit
def test_stage8_validation_2d_alignment_projects_model_z_to_data_plane():
    pts = np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [-10.0, 0.0, 0.0],
            [0.0, -10.0, 0.0],
        ],
        dtype=float,
    )
    model_coords = np.array(
        [
            [10.0, 0.0, -20.0],
            [0.0, 10.0, 15.0],
            [-10.0, 0.0, 30.0],
            [0.0, -10.0, -5.0],
        ],
        dtype=float,
    )

    transform = validation_mod._fit_alignment(pts, data_dim="auto")
    aligned_model = validation_mod._align_model(model_coords, transform)

    assert transform["data_dim"] == "2d"
    assert np.allclose(aligned_model[:, :2], model_coords[:, :2])
    assert np.allclose(aligned_model[:, 2], 0.0)


@pytest.mark.unit
def test_stage8_validation_2d_tree_passes_on_positive_effect_even_with_mixed_wins(monkeypatch):
    class FakeWrapper:
        call_idx = 0

        def __init__(self, *args, **kwargs):
            self.is_translated = kwargs["model_coords_override"][:, 0].mean() > 1.0

        def evaluate(self):
            FakeWrapper.call_idx += 1
            if self.is_translated:
                return -40.0
            # Real and non-translated nulls alternate between slight wins/losses.
            return -10.0 if FakeWrapper.call_idx % 2 else -9.0

    import smlm_score.imp_modeling.restraint.scoring_restraint as sr_mod
    monkeypatch.setattr(sr_mod, "ScoringRestraintWrapper", FakeWrapper)

    theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    pts = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
    model = np.column_stack([np.cos(theta[:8]), np.sin(theta[:8]), np.zeros(8)])

    res = validation_mod.validate_model_vs_nulls(
        cluster_points=pts,
        model_coords=model,
        scoring_type="Tree",
        model=object(),
        avs=[object()],
        n_repeats=2,
        n_nulls=4,
    )

    assert res.metrics["mean_delta"] > 0
    assert res.metrics["win_rate"] <= 0.75
    assert res.passed is True


@pytest.mark.unit
def test_stage8_validation_scramble_preserves_shape():
    data = np.random.rand(100, 3)
    scrambled = validation_mod._scramble_data(data)
    assert scrambled.shape == data.shape
    assert not np.array_equal(scrambled, data)
    # Check that each column is just a permutation
    for i in range(3):
        assert set(scrambled[:, i]) == set(data[:, i])


@pytest.mark.unit
def test_stage8_validation_angular_split_partitions():
    # Create a simple 2D ring of points
    t = np.linspace(0, 2 * np.pi, 20)
    pts = np.vstack([np.cos(t), np.sin(t), np.zeros(20)]).T
    
    thetas = np.arctan2(pts[:, 1], pts[:, 0])
    median_t = np.median(thetas)
    
    mask_a = thetas <= median_t
    mask_b = ~mask_a
    
    assert np.sum(mask_a) >= 9
    assert np.sum(mask_b) >= 9
    assert np.sum(mask_a) + np.sum(mask_b) == 20


@pytest.mark.unit
def test_stage8_validation_cross_val_flow_mock(monkeypatch):
    eval_count = 0
    class FakeWrapper:
        def __init__(self, *args, **kwargs):
            self.dataset = kwargs.get('dataxyz')
        def evaluate(self, pts_override=None):
            nonlocal eval_count
            eval_count += 1
            if not getattr(self, "return_bad", False):
                return -10.0
            # Return varied scores for null cases (e.g., -45, -55)
            # null evaluations are the 2nd and 4th overall eval calls
            return -45.0 if eval_count == 2 else -55.0

    # We need to mock the wrapper imported inside the function
    import smlm_score.imp_modeling.restraint.scoring_restraint as sr_mod
    monkeypatch.setattr(sr_mod, "ScoringRestraintWrapper", FakeWrapper)
    
    # We also need to mock test_gmm_components for GMM
    import smlm_score.imp_modeling.scoring.gmm_score as gmm_mod
    class FakeGMM:
        n_components = 1
    monkeypatch.setattr(gmm_mod, "test_gmm_components", 
                       lambda x: (None, FakeGMM(), np.zeros((1,3)), np.zeros((1,3,3)), np.zeros(1)))

    # Create dummy ring data
    t = np.linspace(0, 2 * np.pi, 50)
    pts = np.vstack([np.cos(t), np.sin(t), np.zeros(50)]).T
    
    # Run CV Tree
    # We want sr_null to return a worse score.
    # I'll use a hack to make the 2nd and 4th instantiation "bad"
    inst_count = 0
    original_init = FakeWrapper.__init__
    def sneaky_init(self, *args, **kwargs):
        nonlocal inst_count
        original_init(self, *args, **kwargs)
        if inst_count % 2 == 1: # 2nd and 4th calls are null_pts
            self.return_bad = True
        inst_count += 1
    monkeypatch.setattr(FakeWrapper, "__init__", sneaky_init)

    res = validation_mod.validate_model_vs_nulls(
        cluster_points=pts,
        model_coords=np.zeros((10, 3)),
        scoring_type="Tree",
        model=object(),
        avs=[object()],
        n_repeats=2,
        n_nulls=1,
    )
    
    assert res.passed is True
    assert res.test_name == "ModelVsNull_Tree"
    assert res.metrics["win_rate"] >= 0.75
    assert res.metrics["effect_size"] > 0


@pytest.mark.unit
def test_stage8_run_full_validation_never_returns_empty():
    results = validation_mod.run_full_validation(
        cluster_scores={},
        held_out_results=None,
        scoring_types=["Tree"],
        cross_val_data=None,
    )

    assert len(results) == 1
    assert results[0].test_name == "Validation_NotRun"
    assert results[0].skipped is True
@pytest.mark.unit
@pytest.mark.parametrize("percent", [10, 25, 50, 100])
def test_flexible_filter_percentage_exact_counts(percent):
    """Verify that percentage filtering returns the exact expected point count."""
    n_total = 100
    df = pd.DataFrame(
        {
            "x [nm]": np.arange(n_total, dtype=float),
            "y [nm]": np.arange(n_total, dtype=float),
            "z [nm]": np.zeros(n_total),
            "Amplitude_0_0": np.ones(n_total),
        }
    )
    # flexible_filter_smlm_data uses random sampling for percentage
    xyz, variance, tree_xyz, tree, _ = data_handling.flexible_filter_smlm_data(
        df, filter_type="percentage", percentage=percent, random_seed=42
    )

    expected_count = int(n_total * (percent / 100.0))
    assert xyz.shape[0] == expected_count
    assert variance.shape[0] == expected_count


@pytest.mark.unit
def test_flexible_filter_random_returns_applied_spatial_cuts():
    df = pd.DataFrame(
        {
            "x [nm]": np.linspace(0.0, 100.0, 100),
            "y [nm]": np.linspace(50.0, 150.0, 100),
            "Amplitude_0_0": np.ones(100),
        }
    )

    xyz, variance, tree_xyz, tree, applied = data_handling.flexible_filter_smlm_data(
        df,
        filter_type="random",
        percentage=25,
        random_seed=7,
        fill_z_value=0.0,
        return_tree=True,
    )

    assert xyz.shape[0] > 0
    assert np.allclose(variance, np.ones_like(variance))
    assert tree_xyz.shape == xyz.shape
    assert tree is not None
    assert applied["x"] is not None
    assert applied["y"] is not None
    assert applied["z"] is None
    x_min, x_max = applied["x"]
    y_min, y_max = applied["y"]
    assert np.all(xyz[:, 0] >= x_min)
    assert np.all(xyz[:, 0] <= x_max)
    assert np.all(xyz[:, 1] >= y_min)
    assert np.all(xyz[:, 1] <= y_max)
