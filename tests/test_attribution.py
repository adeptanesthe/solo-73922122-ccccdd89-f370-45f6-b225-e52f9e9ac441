"""突变归因核心逻辑测试：似然正确性、类别后验、有效时刻规则、拒绝行为。"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.attribution import (
    DECISION_THRESHOLD,
    InputError,
    KernelNotPSDError,
    NoValidStepTimeError,
    NonFiniteInputError,
    attribute_step,
)
from app.gp import squared_exponential_kernel

BASE = dict(length_scale=2.0, noise_std=5.0, signal_variance=1e4, step_prior_std=300.0)


def _explicit_class_ll(t, y, v, *, length_scale, noise_std, signal_variance, step_prior_std):
    """显式构造 C = K + σ²I + s²vvᵀ 的边际对数似然（与 Woodbury 实现相互独立）。"""
    k = squared_exponential_kernel(t, t, length_scale, signal_variance)
    cov = k + noise_std ** 2 * np.eye(len(t)) + step_prior_std ** 2 * np.outer(v, v)
    sign, logdet = np.linalg.slogdet(cov)
    assert sign > 0
    return -0.5 * (len(t) * math.log(2.0 * math.pi) + logdet + float(y @ np.linalg.solve(cov, y)))


def _simulate(*, probe_step=0.0, common_step=0.0, tau=7.0, n=15, seed=0, ell=2.0,
              sig_var=1e4, noise=5.0, span=14.0, regular=False):
    """两支探头共享潜在 GP + 阶跃 + 独立噪声的合成数据。

    默认用随机采样时刻；``regular=True`` 时用等间隔采样，便于让候选时刻
    恰好落在真实阶跃时刻、且两侧都有读数。
    """
    rng = np.random.default_rng(seed)
    times = np.linspace(0, span, n) if regular else np.sort(rng.uniform(0, span, n))
    k = squared_exponential_kernel(times, times, ell, sig_var)
    f = np.linalg.cholesky(k + 1e-8 * np.eye(n)) @ rng.normal(size=n)
    y_ref = f + common_step * (times >= tau) + rng.normal(0, noise, n)
    y_test = f + (common_step + probe_step) * (times >= tau) + rng.normal(0, noise, n)
    return times, y_ref, times.copy(), y_test


def _log_mean_exp(values):
    m = float(np.max(values))
    return m + math.log(float(np.mean(np.exp(values - m))))


# ---------------- 似然正确性 ----------------

@pytest.mark.parametrize("tau", [3.0, 7.0, 11.0])
def test_candidate_likelihoods_match_explicit_gaussian(tau):
    times, y_ref, test_t, y_test = _simulate(seed=5)
    res = attribute_step(times, y_ref, test_t, y_test, [tau], **BASE)
    diag = res.candidates[0]
    assert diag.valid
    t = np.concatenate([times, test_t])
    y = np.concatenate([y_ref, y_test])
    is_test = np.concatenate([np.zeros(times.size, bool), np.ones(test_t.size, bool)])
    v_common = (t >= tau).astype(float)
    v_test = np.where(is_test, (t >= tau), 0.0)
    assert diag.log_likelihood_common == pytest.approx(
        _explicit_class_ll(t, y, v_common, **BASE), rel=1e-9, abs=1e-8)
    assert diag.log_likelihood_test_probe == pytest.approx(
        _explicit_class_ll(t, y, v_test, **BASE), rel=1e-9, abs=1e-8)


def test_class_scores_use_equal_priors_and_time_averaging():
    times, y_ref, test_t, y_test = _simulate(probe_step=600.0, seed=3)
    sus = [4.0, 7.0, 10.0]
    res = attribute_step(times, y_ref, test_t, y_test, sus, **BASE)
    valid = [c for c in res.candidates if c.valid]
    assert len(valid) == 3
    for cls, key in (("common_step", "log_likelihood_common"),
                     ("test_probe_step", "log_likelihood_test_probe")):
        lls = np.array([getattr(c, key) for c in valid])
        expected = math.log(1 / 3) + _log_mean_exp(lls)
        assert res.log_class_scores[cls] == pytest.approx(expected, rel=1e-10, abs=1e-8)
    expected_h0 = math.log(1 / 3) + res.log_likelihood_no_step
    assert res.log_class_scores["no_step"] == pytest.approx(expected_h0, rel=1e-12)
    assert sum(res.posterior_probabilities.values()) == pytest.approx(1.0)


def test_time_weights_within_class_are_normalized():
    times, y_ref, test_t, y_test = _simulate(probe_step=600.0, seed=4)
    res = attribute_step(times, y_ref, test_t, y_test, [4.0, 7.0, 10.0], **BASE)
    wc = sum(c.time_weight_common for c in res.candidates if c.valid)
    wt = sum(c.time_weight_test_probe for c in res.candidates if c.valid)
    assert wc == pytest.approx(1.0)
    assert wt == pytest.approx(1.0)


# ---------------- 三类情形 ----------------

def test_no_step_scenario_favours_no_step():
    times, y_ref, test_t, y_test = _simulate(seed=0, span=19.0, n=21, regular=True)
    res = attribute_step(times, y_ref, test_t, y_test, [6.0, 8.0, 10.0], **BASE)
    assert res.decision == "no_step"
    assert res.conclusive
    assert res.posterior_probabilities["no_step"] >= DECISION_THRESHOLD
    assert res.best_step_time is None


def test_common_step_scenario_recovers_class_and_time():
    times, y_ref, test_t, y_test = _simulate(common_step=900.0, tau=8.0, seed=0, n=21,
                                             span=19.0, regular=True)
    res = attribute_step(times, y_ref, test_t, y_test, [6.0, 8.0, 10.0], **BASE)
    assert res.decision == "common_step"
    assert res.conclusive
    assert res.best_step_time == 8.0
    assert res.best_step_time_probability > 0.5


def test_test_probe_only_step_recovers_class_and_time():
    times, y_ref, test_t, y_test = _simulate(probe_step=900.0, tau=8.0, seed=0, n=21,
                                             span=19.0, regular=True)
    res = attribute_step(times, y_ref, test_t, y_test, [6.0, 8.0, 10.0], **BASE)
    assert res.decision == "test_probe_step"
    assert res.conclusive
    assert res.best_step_time == 8.0


def test_ambiguous_case_requires_review():
    # 读数少、噪声相对阶跃幅度大 -> 最高后验不到 90%，不得下结论，也不给时刻。
    ref_t = [0.0, 2.0, 4.0, 6.0]
    ref_y = [100.0, 102.0, 99.0, 101.0]
    res = attribute_step(ref_t, ref_y, ref_t, ref_y, [4.0],
                         length_scale=2.0, noise_std=5.0,
                         signal_variance=2000.0, step_prior_std=20.0)
    assert not res.conclusive
    assert res.decision == "needs_review"
    assert max(res.posterior_probabilities.values()) < DECISION_THRESHOLD
    assert res.best_step_time is None


# ---------------- 有效时刻规则 ----------------

def test_step_time_requires_readings_on_both_sides_for_both_probes():
    ref_t = [0.0, 1.0, 2.0, 3.0]
    ref_y = [0.0, 1.0, 2.0, 3.0]
    test_t = [0.0, 1.0, 2.0, 3.0]
    test_y = [0.0, 1.0, 2.0, 3.0]
    res = attribute_step(ref_t, ref_y, test_t, test_y, [0.0, 2.0, 5.0], **BASE)
    by_time = {c.time: c for c in res.candidates}
    # t=0：两支探头都没有“阶跃前”读数
    assert not by_time[0.0].valid
    assert "阶跃前无读数" in by_time[0.0].reason
    # t=5（晚于最后一条读数）：两支探头都没有“从该时刻起”的读数
    assert not by_time[5.0].valid
    assert "从该时刻起无读数" in by_time[5.0].reason
    # t=2：两侧都有
    assert by_time[2.0].valid
    assert res.n_valid_step_times == 1


def test_reading_exactly_at_tau_counts_as_after_not_before():
    ref_t = [0.0, 1.0, 5.0]
    ref_y = [0.0, 1.0, 5.0]
    test_t = [0.0, 1.0, 5.0]
    test_y = [0.0, 1.0, 5.0]
    res = attribute_step(ref_t, ref_y, test_t, test_y, [1.0], **BASE)
    c = res.candidates[0]
    assert c.valid
    assert (c.reference_before, c.reference_after) == (1, 2)
    assert (c.test_before, c.test_after) == (1, 2)


def test_no_valid_step_time_raises_with_diagnostics():
    with pytest.raises(NoValidStepTimeError) as exc:
        attribute_step([0.0], [1.0], [0.0], [1.0], [0.0, 5.0], **BASE)
    err = exc.value
    assert err.code == "NO_VALID_STEP_TIME"
    assert err.candidates is not None and len(err.candidates) == 2
    assert all(not c.valid for c in err.candidates)


# ---------------- 数值稳定性 ----------------

def test_large_magnitude_readings_and_tiny_noise_stay_finite():
    # 炉温可达 1500+，读数整体偏移不影响类别后验（模型只比较似然差）。
    times = np.linspace(0, 12, 13)
    base = 1200.0 + 20.0 * times
    rng = np.random.default_rng(8)
    y_ref = base + rng.normal(0, 0.5, times.size)
    y_test = base + 300.0 * (times >= 6.0) + rng.normal(0, 0.5, times.size)
    res = attribute_step(times, y_ref, times, y_test, [4.0, 6.0, 8.0],
                         length_scale=2.0, noise_std=0.5,
                         signal_variance=500.0, step_prior_std=200.0)
    p = res.posterior_probabilities
    assert all(math.isfinite(v) for v in p.values())
    assert abs(sum(p.values()) - 1.0) < 1e-12
    assert res.decision == "test_probe_step"
    assert res.best_step_time == 6.0


def test_invalid_step_time_does_not_consume_prior_mass():
    # 无效时刻不参与均分：有效时刻集合变化时，类得分只按有效时刻平均。
    ref_t = [0.0, 2.0, 4.0, 6.0]
    ref_y = [100.0, 101.0, 99.0, 100.0]
    res = attribute_step(ref_t, ref_y, ref_t, ref_y, [2.0, 4.0, 99.0], **BASE)
    assert res.n_valid_step_times == 2
    # 无效时刻没有似然/权重
    invalid = next(c for c in res.candidates if c.time == 99.0)
    assert invalid.log_likelihood_common is None
    assert invalid.time_weight_common is None


# ---------------- 输入校验与拒绝 ----------------

def test_suspicious_times_limits():
    t = [0.0, 1.0, 2.0]
    y = [0.0, 1.0, 2.0]
    with pytest.raises(InputError):
        attribute_step(t, y, t, y, [], **BASE)
    with pytest.raises(InputError):
        attribute_step(t, y, t, y, list(np.arange(21, dtype=float)), **BASE)
    with pytest.raises(InputError):
        attribute_step(t, y, t, y, [1.0, 1.0], **BASE)


def test_noise_must_be_strictly_positive():
    t = [0.0, 1.0, 2.0]
    y = [0.0, 1.0, 2.0]
    with pytest.raises(InputError):
        attribute_step(t, y, t, y, [1.0], length_scale=1.0, noise_std=0.0,
                       signal_variance=1.0, step_prior_std=1.0)
    with pytest.raises(InputError):
        attribute_step(t, y, t, y, [1.0], length_scale=1.0, noise_std=1.0,
                       signal_variance=1.0, step_prior_std=0.0)


def test_non_finite_and_mismatched_inputs_rejected():
    t = [0.0, 1.0, 2.0]
    y = [0.0, 1.0, 2.0]
    with pytest.raises(NonFiniteInputError):
        attribute_step(t, y, t, [0.0, float("nan"), 2.0], [1.0], **BASE)
    with pytest.raises(InputError):
        attribute_step(t, [0.0, 1.0], t, y, [1.0], **BASE)
    with pytest.raises(InputError):
        attribute_step(t, y, t, [0.0, 1.0], [1.0], **BASE)


def test_empty_probes_raise_no_valid_step_time():
    with pytest.raises(NoValidStepTimeError):
        attribute_step([], [], [], [], [1.0], **BASE)


def test_near_singular_kernel_rejected():
    # 同一时刻重复读数 + 极小噪声 -> 观测核矩阵数值奇异
    with pytest.raises(KernelNotPSDError):
        attribute_step([1.0, 1.0], [10.0, 10.0], [1.0, 1.0], [10.0, 10.0], [1.0],
                       length_scale=1.0, noise_std=1e-9,
                       signal_variance=1.0, step_prior_std=1.0)


def test_result_is_deterministic():
    times, y_ref, test_t, y_test = _simulate(probe_step=700.0, seed=9)
    args = (times, y_ref, test_t, y_test, [4.0, 7.0, 10.0])
    a = attribute_step(*args, **BASE)
    b = attribute_step(*args, **BASE)
    assert a.posterior_probabilities == b.posterior_probabilities
    assert a.best_step_time == b.best_step_time
