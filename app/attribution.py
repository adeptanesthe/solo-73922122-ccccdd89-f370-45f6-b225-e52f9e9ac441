"""炉温突变归因：两支探头共享 GP 的三类阶跃假设比较（仅依赖 NumPy）。

模型
----
参考探头与待检探头共享同一个零均值平方指数核高斯过程 f(t)：

    y_p(t) = f(t) + [class 含该探头的阶跃] * A * 1[t >= τ] + ε_p(t)
    f ~ GP(0, k_SE),   ε_p(t) ~ N(0, noise_std²) 独立同分布

三类等先验假设：

* ``no_step``        ：两支探头都没有阶跃（H0）；
* ``common_step``    ：两支探头从 τ 起同时出现同一幅度的阶跃（H1）；
* ``test_probe_step``：只有待检探头从 τ 起出现阶跃（H2）。

阶跃幅度 A ~ N(0, step_prior_std²)，从候选时刻 τ 起生效（含 τ）。
给定 τ 和类别后，A 与 f 都被积分掉，观测的边际协方差为

    C = K + noise_std² I + step_prior_std² v vᵀ

其中 v 是该类别下“从 τ 起”的 0/1 指示向量（H1 对所有读数置 1，H2 仅待检
探头读数置 1）。因此边际似然可闭式求得；利用 C0 = K + noise_std² I 的
Cholesky 分解与秩一 Woodbury 恒等式，所有候选时刻只需一次批量三角回代：

    g = vᵀC0⁻¹v,   b = vᵀC0⁻¹y,   a = step_prior_std²
    log|C| = log|C0| + log(1 + a·g)
    yᵀC⁻¹y = yᵀC0⁻¹y - a·b² / (1 + a·g)

先验：三类各占 1/3；每类阶跃的时刻先验在**有效时刻**间均分。有效时刻要求
两支探头各自在阶跃前和从该时刻起至少各有一条读数。后验类别概率由三类的
边际似然（对有效时刻取平均）归一化得到。

不给出任何部分结果的情形
------------------------
* 输入非有限 / 参数越界         -> InputError (INVALID_INPUT / NON_FINITE_INPUT)
* 观测核矩阵非正定              -> KernelNotPSDError (KERNEL_NOT_PSD)
* 没有任何有效候选时刻          -> NoValidStepTimeError (NO_VALID_STEP_TIME)
* 数值求解失败                  -> SolveError (SOLUTION_FAILED)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .gp import _SAME_TIME_ATOL, _SAME_TIME_RTOL, squared_exponential_kernel

__all__ = [
    "AttributionError",
    "InputError",
    "KernelNotPSDError",
    "NoValidStepTimeError",
    "SolveError",
    "CandidateTimeDiagnostic",
    "AttributionResult",
    "attribute_step",
]

MAX_SUSPICIOUS_TIMES = 20
DECISION_THRESHOLD = 0.90

CLASS_NAMES = ("no_step", "common_step", "test_probe_step")


class AttributionError(Exception):
    """突变归因领域错误基类：出现即整批拒绝，不返回部分结果。"""

    code = "ATTRIBUTION_ERROR"

    def __init__(self, message: str, *, candidates: list | None = None):
        super().__init__(message)
        self.candidates = candidates


class InputError(AttributionError):
    """输入非法（长度不一致、重复时刻、参数越界等）。"""

    code = "INVALID_INPUT"


class NonFiniteInputError(AttributionError):
    """输入含 NaN / Inf。"""

    code = "NON_FINITE_INPUT"


class KernelNotPSDError(AttributionError):
    """观测核矩阵非正定或数值上近奇异。"""

    code = "KERNEL_NOT_PSD"


class NoValidStepTimeError(AttributionError):
    """没有任何候选时刻同时得到两支探头阶跃前后的读数支持。"""

    code = "NO_VALID_STEP_TIME"


class SolveError(AttributionError):
    """数值求解失败。"""

    code = "SOLUTION_FAILED"


@dataclass
class CandidateTimeDiagnostic:
    """单个可疑时刻的可用性诊断。"""

    time: float
    valid: bool
    reason: str | None
    reference_before: int
    reference_after: int
    test_before: int
    test_after: int
    log_likelihood_common: float | None = None
    log_likelihood_test_probe: float | None = None
    time_weight_common: float | None = None
    time_weight_test_probe: float | None = None


@dataclass
class AttributionResult:
    """一次突变归因的全部结果。"""

    posterior_probabilities: dict
    log_class_scores: dict
    log_likelihood_no_step: float
    n_valid_step_times: int
    candidates: list = field(default_factory=list)
    # 仅当后验概率达到阈值且判为某类阶跃时给出
    best_step_time: float | None = None
    best_step_time_probability: float | None = None
    decision: str = "needs_review"
    conclusive: bool = False
    threshold: float = DECISION_THRESHOLD


def _as_finite_1d(values, name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        raise InputError(f"{name} 必须是一维数值序列")
    if not np.all(np.isfinite(arr)):
        raise NonFiniteInputError(f"{name} 含有 NaN 或无穷值，整批拒绝")
    return arr


def _positive_scalar(value, name: str) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise NonFiniteInputError(f"参数 {name} 不是有限值，整批拒绝")
    if v <= 0:
        raise InputError(f"参数 {name} 必须 > 0")
    return v


def _at_or_after(times: np.ndarray, tau: float) -> np.ndarray:
    """从 tau 起（含 tau）的掩码；浮点容差内视为“该时刻”。"""
    return (times >= tau) | np.isclose(times, tau, rtol=_SAME_TIME_RTOL, atol=_SAME_TIME_ATOL)


def _before(times: np.ndarray, tau: float) -> np.ndarray:
    """严格早于 tau 的掩码；浮点容差内与 tau 重合的读数不算“之前”。"""
    return (times < tau) & ~np.isclose(times, tau, rtol=_SAME_TIME_RTOL, atol=_SAME_TIME_ATOL)


def _softmax(log_values: np.ndarray) -> np.ndarray:
    m = float(np.max(log_values))
    if not np.isfinite(m):
        raise SolveError("后验得分全部非有限，无法归一化，求解失败")
    e = np.exp(log_values - m)
    total = float(np.sum(e))
    if not math.isfinite(total) or total <= 0:
        raise SolveError("后验归一化失败，求解失败")
    return e / total


def _cholesky(cov: np.ndarray) -> np.ndarray:
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise KernelNotPSDError(
            "观测核矩阵（含观测噪声）非正定，无法求解。请检查是否存在重复读数、"
            "过小的噪声或与数据范围不相称的长度尺度。"
        ) from exc
    pivot_scale = math.sqrt(max(float(np.max(np.diag(cov))), np.finfo(float).tiny))
    if np.any(np.abs(np.diag(chol)) <= 1e-10 * pivot_scale):
        raise KernelNotPSDError(
            "观测核矩阵（含观测噪声）数值上非正定（Cholesky 主元近零），求解不可靠。"
            "请增大观测噪声或检查长度尺度。"
        )
    return chol


def attribute_step(
    reference_times,
    reference_values,
    test_times,
    test_values,
    suspicious_times,
    *,
    length_scale: float,
    noise_std: float,
    signal_variance: float,
    step_prior_std: float,
) -> AttributionResult:
    """比较无阶跃 / 共同阶跃 / 仅待检探头阶跃三类假设的后验概率。

    参数
    ----
    reference_times / reference_values : 参考探头读数（即页面上的已测数据）
    test_times / test_values           : 待检探头读数
    suspicious_times                   : 可疑阶跃时刻，1..20 个且互不相同
    length_scale                       : 共享平方指数核长度尺度 ℓ (> 0)
    noise_std                          : 每条读数独立高斯噪声标准差 σ_n (> 0)
    signal_variance                    : 共享核信号方差 (> 0)
    step_prior_std                     : 阶跃幅度零均值高斯先验标准差 (> 0)

    任何输入异常都会抛出 AttributionError 子类，整批拒绝。
    """
    length_scale = _positive_scalar(length_scale, "length_scale")
    noise_std = _positive_scalar(noise_std, "noise_std")
    signal_variance = _positive_scalar(signal_variance, "signal_variance")
    step_prior_std = _positive_scalar(step_prior_std, "step_prior_std")

    t_ref = _as_finite_1d(reference_times, "参考探头时刻")
    y_ref = _as_finite_1d(reference_values, "参考探头温度")
    t_test = _as_finite_1d(test_times, "待检探头时刻")
    y_test = _as_finite_1d(test_values, "待检探头温度")
    if t_ref.size != y_ref.size:
        raise InputError("参考探头时刻与温度数量不一致")
    if t_test.size != y_test.size:
        raise InputError("待检探头时刻与温度数量不一致")

    t_sus = _as_finite_1d(suspicious_times, "可疑时刻")
    if t_sus.size == 0:
        raise InputError("至少需要 1 个可疑时刻")
    if t_sus.size > MAX_SUSPICIOUS_TIMES:
        raise InputError(f"可疑时刻至多 {MAX_SUSPICIOUS_TIMES} 个")
    iu0, iu1 = np.triu_indices(t_sus.size, k=1)
    if np.any(np.isclose(t_sus[iu0], t_sus[iu1], rtol=_SAME_TIME_RTOL, atol=_SAME_TIME_ATOL)):
        raise InputError("可疑时刻必须互不相同")

    t = np.concatenate([t_ref, t_test])
    y = np.concatenate([y_ref, y_test])
    n = t.size
    if n == 0:
        raise NoValidStepTimeError(
            "两支探头都没有读数，无法比较阶跃假设",
            candidates=[
                CandidateTimeDiagnostic(float(tau), False, "两支探头均无读数", 0, 0, 0, 0)
                for tau in t_sus
            ],
        )
    is_test = np.concatenate([np.zeros(t_ref.size, dtype=bool), np.ones(t_test.size, dtype=bool)])

    noise_var = noise_std * noise_std
    step_var = step_prior_std * step_prior_std

    k_tt = squared_exponential_kernel(t, t, length_scale, signal_variance)
    c0 = k_tt + noise_var * np.eye(n)
    chol = _cholesky(c0)
    logdet0 = 2.0 * float(np.sum(np.log(np.diag(chol))))

    try:
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))  # C0^{-1} y
        quad0 = float(y @ alpha)
        log_norm = n * math.log(2.0 * math.pi)
    except np.linalg.LinAlgError as exc:
        raise SolveError("观测核矩阵求解失败（三角回代不收敛），请检查输入。") from exc

    # H0：无阶跃
    log_likelihood_no_step = -0.5 * (log_norm + logdet0 + quad0)
    if not math.isfinite(log_likelihood_no_step):
        raise SolveError("无阶跃假设的边际似然非有限，求解失败")

    diags: list[CandidateTimeDiagnostic] = []
    valid_indices: list[int] = []
    v_columns: list[np.ndarray] = []

    for idx, tau in enumerate(t_sus):
        tau = float(tau)
        ref_after_m = _at_or_after(t_ref, tau)
        ref_before_m = _before(t_ref, tau)
        test_after_m = _at_or_after(t_test, tau)
        test_before_m = _before(t_test, tau)
        n_rb, n_ra = int(np.count_nonzero(ref_before_m)), int(np.count_nonzero(ref_after_m))
        n_tb, n_ta = int(np.count_nonzero(test_before_m)), int(np.count_nonzero(test_after_m))

        missing = []
        if n_rb < 1:
            missing.append("参考探头在阶跃前无读数")
        if n_ra < 1:
            missing.append("参考探头从该时刻起无读数")
        if n_tb < 1:
            missing.append("待检探头在阶跃前无读数")
        if n_ta < 1:
            missing.append("待检探头从该时刻起无读数")
        diags.append(CandidateTimeDiagnostic(
            time=tau,
            valid=not missing,
            reason="；".join(missing) if missing else None,
            reference_before=n_rb,
            reference_after=n_ra,
            test_before=n_tb,
            test_after=n_ta,
        ))
        if missing:
            continue

        valid_indices.append(idx)
        # 阶跃指示向量：H1 作用于全部读数，H2 仅作用于待检探头读数。
        after = _at_or_after(t, tau).astype(float)
        v_columns.append(after)
        v_columns.append(np.where(is_test, after, 0.0))

    n_valid = len(valid_indices)
    if n_valid == 0:
        raise NoValidStepTimeError(
            "没有有效候选时刻：每个可疑时刻都需要两支探头各自在阶跃前和从该时刻起"
            "至少各有一条读数。请调整可疑时刻或补充读数。",
            candidates=diags,
        )

    # 所有候选时刻的指示向量一次性批量回代（远快于逐个求解）。
    v_mat = np.column_stack(v_columns)                     # (n, 2 * n_valid)
    try:
        gamma = np.linalg.solve(chol.T, np.linalg.solve(chol, v_mat))  # C0^{-1} V
    except np.linalg.LinAlgError as exc:
        raise SolveError("候选时刻的秩一更新求解失败，请检查输入。") from exc
    s_all = np.einsum("ij,ij->j", v_mat, gamma)            # vᵀ C0^{-1} v
    b_all = v_mat.T @ alpha                                # vᵀ C0^{-1} y

    if not np.all(np.isfinite(s_all)) or np.any(s_all <= 0.0) or not np.all(np.isfinite(b_all)):
        bad = [
            diags[idx].time
            for k, idx in enumerate(valid_indices)
            if not (math.isfinite(float(s_all[2 * k])) and s_all[2 * k] > 0.0
                    and math.isfinite(float(s_all[2 * k + 1])) and s_all[2 * k + 1] > 0.0
                    and math.isfinite(float(b_all[2 * k])) and math.isfinite(float(b_all[2 * k + 1])))
        ]
        raise SolveError(
            "候选时刻的阶跃方向数值退化"
            + (f"（时刻 {', '.join(f'{x:g}' for x in bad)}）" if bad else "")
            + "，求解失败。请增大观测噪声或检查长度尺度。"
        )

    common_lls = np.full(t_sus.size, np.nan)
    test_lls = np.full(t_sus.size, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        amps = step_var * s_all
        logdets = logdet0 + np.log1p(amps)
        quads = quad0 - step_var * b_all * b_all / (1.0 + amps)
        lls = -0.5 * (log_norm + logdets + quads)
    if not np.all(np.isfinite(lls)):
        raise SolveError("候选时刻的边际似然出现非有限值，求解失败。")
    for k, idx in enumerate(valid_indices):
        common_lls[idx] = lls[2 * k]
        test_lls[idx] = lls[2 * k + 1]

    valid_idx_arr = np.asarray(valid_indices)

    log_prior_class = math.log(1.0 / 3.0)

    # 类内时刻先验在有效时刻间均分，故该类得分为 log(1/3) + log(mean_tau L_tau)；
    # _logmeanexp 已含 1/n_valid 的平均，不再重复扣减。
    log_score_no_step = log_prior_class + log_likelihood_no_step
    log_score_common = log_prior_class + _logmeanexp(common_lls[valid_idx_arr])
    log_score_test = log_prior_class + _logmeanexp(test_lls[valid_idx_arr])
    log_scores = np.array([log_score_no_step, log_score_common, log_score_test])

    post = _softmax(log_scores)
    probs = {name: float(p) for name, p in zip(CLASS_NAMES, post)}

    # 每类阶跃内部的时刻后验（有效时刻间先验均分，故与似然成正比）
    w_common = _softmax(common_lls[valid_idx_arr])
    w_test = _softmax(test_lls[valid_idx_arr])
    common_by_idx = {int(i): float(w) for i, w in zip(valid_indices, w_common)}
    test_by_idx = {int(i): float(w) for i, w in zip(valid_indices, w_test)}
    for idx, diag in enumerate(diags):
        if not diag.valid:
            continue
        diag.log_likelihood_common = float(common_lls[idx])
        diag.log_likelihood_test_probe = float(test_lls[idx])
        diag.time_weight_common = common_by_idx[idx]
        diag.time_weight_test_probe = test_by_idx[idx]

    order = list(CLASS_NAMES)
    best_name = order[int(np.argmax(post))]
    best_prob = probs[best_name]
    conclusive = best_prob >= DECISION_THRESHOLD

    best_time = None
    best_time_prob = None
    if conclusive and best_name in ("common_step", "test_probe_step"):
        weights = common_by_idx if best_name == "common_step" else test_by_idx
        best_idx = max(weights, key=weights.get)
        best_time = float(t_sus[best_idx])
        best_time_prob = float(weights[best_idx])

    return AttributionResult(
        posterior_probabilities=probs,
        log_class_scores={n: float(s) for n, s in zip(CLASS_NAMES, log_scores)},
        log_likelihood_no_step=float(log_likelihood_no_step),
        n_valid_step_times=n_valid,
        candidates=diags,
        best_step_time=best_time,
        best_step_time_probability=best_time_prob,
        decision=best_name if conclusive else "needs_review",
        conclusive=conclusive,
        threshold=DECISION_THRESHOLD,
    )


def _logmeanexp(values: np.ndarray) -> float:
    """log(mean(exp(values)))，数值稳定。"""
    m = float(np.max(values))
    if not math.isfinite(m):
        raise SolveError("候选时刻的边际似然出现非有限值，求解失败")
    return m + math.log(float(np.mean(np.exp(values - m))))
