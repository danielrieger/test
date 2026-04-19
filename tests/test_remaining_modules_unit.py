import importlib
import shutil
import types
import uuid
from pathlib import Path

import numpy as np
import pytest

from smlm_score.imp_modeling.restraint import scoring_restraint
from smlm_score.imp_modeling.restraint import imp_restraint
from smlm_score.utility import plot as plot_mod
from smlm_score.benchmarking import gmm_benchmarks


@pytest.mark.unit
def test_scoring_restraint_gmm_do_get_inputs_uses_av_particles():
    class FakeAV:
        def __init__(self, p):
            self.p = p

        def get_particle(self):
            return self.p

    obj = scoring_restraint.ScoringRestraintGMM.__new__(scoring_restraint.ScoringRestraintGMM)
    obj.avs_decorators = [FakeAV("p1"), FakeAV("p2")]
    assert obj.do_get_inputs() == ["p1", "p2"]


@pytest.mark.unit
def test_scoring_restraint_wrapper_get_output_and_evaluate():
    class FakeRS:
        def unprotected_evaluate(self, _):
            return 42.0

    class FakeInner:
        def get_output(self):
            return {"k": "v"}

    wrapper = scoring_restraint.ScoringRestraintWrapper.__new__(scoring_restraint.ScoringRestraintWrapper)
    wrapper.rs = FakeRS()
    wrapper.scoring_restraint_instance = FakeInner()

    assert wrapper.evaluate() == 42.0
    assert wrapper.get_output() == {"k": "v"}


@pytest.mark.unit
def test_imp_restraint_mean_distance_do_get_inputs():
    obj = imp_restraint.AVMeanDistanceRestraint.__new__(imp_restraint.AVMeanDistanceRestraint)
    obj.particle_list = ["a", "b"]
    assert obj.do_get_inputs() == ["a", "b"]

@pytest.mark.unit
def test_plot_make_gmm_component_plot_runs_headless(monkeypatch):
    monkeypatch.setattr(plot_mod.plt, "show", lambda: None)
    r = {
        "n_components": [1, 2, 4],
        "score": [-10.0, -9.0, -8.5],
        "aic": [100.0, 90.0, 95.0],
        "bic": [110.0, 92.0, 97.0],
        "n": 1,
    }
    assert plot_mod.make_gmm_component_plot(r) is None




@pytest.mark.unit
def test_gmm_benchmarking_timing_with_patched_globals(monkeypatch):
    fake_time = types.SimpleNamespace()
    t = {"v": 0.0}

    def now():
        t["v"] += 1.0
        return t["v"]

    fake_time.time = now
    monkeypatch.setattr(gmm_benchmarks, "time", fake_time, raising=False)
    monkeypatch.setattr(gmm_benchmarks, "np", np, raising=False)
    monkeypatch.setattr(
        gmm_benchmarks,
        "test_gmm_components",
        lambda data, component_max=1000: {"n_components": [1, 2, 4]},
        raising=False,
    )

    result, times = gmm_benchmarks.gmm_benchmarking.test_gmm_components_with_timing(
        np.zeros((4, 3)), component_max=4
    )
    assert result["n_components"] == [1, 2, 4]
    assert len(times) == 3
