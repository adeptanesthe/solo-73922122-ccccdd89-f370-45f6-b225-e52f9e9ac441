"""FastAPI 入口：高温炉温度采样选点与温度突变归因服务。"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from .attribution import AttributionError, attribute_step
from .gp import GPError, select_pair
from .schemas import AttributeRequest, SelectRequest

app = FastAPI(
    title="高温炉温度实验工具",
    version="1.1.0",
    summary="基于高斯过程的炉温采样选点与温度突变归因",
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.exception_handler(RequestValidationError)
async def _validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """把 schema 校验失败统一成结构化 422（整批拒绝，不给部分建议）。"""
    errors = [
        {
            "loc": [str(p) for p in err.get("loc", [])],
            "msg": err.get("msg", ""),
            "type": err.get("type", ""),
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "code": "INVALID_INPUT",
                "message": "输入校验失败（字段缺失、非有限值或参数越界），整批拒绝",
                "errors": errors,
            }
        },
    )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/select")
def select(req: SelectRequest) -> dict:
    """计算最优新增采样点对，返回选点前后每个候选时刻的方差。

    失败时返回 422，detail.code 取值：
    NON_FINITE_INPUT / KERNEL_NOT_PSD / NO_FEASIBLE_COMBINATION / INVALID_INPUT。
    """
    try:
        res = select_pair(
            req.measured_times,
            req.measured_values,
            req.candidate_times,
            length_scale=req.length_scale,
            noise_std=req.noise_std,
            signal_variance=req.signal_variance,
            min_interval=req.min_interval,
        )
    except GPError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc

    candidates = [
        {
            "index": i,
            "time": float(t),
            "eligible": bool(res.eligible[i]),
            "exclusion_reason": res.exclusion_reasons[i],
            "variance_before": float(res.variance_before[i]),
            "variance_after": float(res.variance_after[i]),
            "posterior_mean": float(res.posterior_mean[i]),
            "selected": i in res.selected_indices,
        }
        for i, t in enumerate(req.candidate_times)
    ]
    return {
        "selected_indices": list(res.selected_indices),
        "selected_times": list(res.selected_times),
        "total_variance_before": res.total_variance_before,
        "total_variance_after": res.total_variance_after,
        "total_reduction": res.total_reduction,
        "n_pairs_total": res.n_pairs_total,
        "n_pairs_feasible": res.n_pairs_feasible,
        "candidates": candidates,
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.post("/api/attribute")
def attribute(req: AttributeRequest) -> dict:
    """炉温突变归因：比较无阶跃 / 共同阶跃 / 仅待检探头阶跃三类假设。

    失败时返回 422，detail.code 取值：
    NON_FINITE_INPUT / INVALID_INPUT / KERNEL_NOT_PSD / NO_VALID_STEP_TIME / SOLUTION_FAILED。
    没有有效候选时刻时，detail 里附带每个可疑时刻的诊断（candidates）。
    """
    try:
        res = attribute_step(
            req.reference_times,
            req.reference_values,
            req.test_times,
            req.test_values,
            req.suspicious_times,
            length_scale=req.length_scale,
            noise_std=req.noise_std,
            signal_variance=req.signal_variance,
            step_prior_std=req.step_prior_std,
        )
    except AttributionError as exc:
        detail = {"code": exc.code, "message": str(exc)}
        if exc.candidates is not None:
            detail["candidates"] = [_candidate_diag(c) for c in exc.candidates]
        raise HTTPException(status_code=422, detail=detail) from exc

    return {
        "posterior_probabilities": res.posterior_probabilities,
        "log_class_scores": res.log_class_scores,
        "log_likelihood_no_step": res.log_likelihood_no_step,
        "n_valid_step_times": res.n_valid_step_times,
        "threshold": res.threshold,
        "conclusive": res.conclusive,
        "decision": res.decision,
        "best_step_time": res.best_step_time,
        "best_step_time_probability": res.best_step_time_probability,
        "candidates": [_candidate_diag(c) for c in res.candidates],
    }


def _candidate_diag(c) -> dict:
    """候选取证诊断的统一序列化（有效结果与错误详情共用）。"""
    item = {
        "time": c.time,
        "valid": c.valid,
        "reason": c.reason,
        "reference_before": c.reference_before,
        "reference_after": c.reference_after,
        "test_before": c.test_before,
        "test_after": c.test_after,
    }
    if c.log_likelihood_common is not None:
        item["log_likelihood_common"] = c.log_likelihood_common
        item["log_likelihood_test_probe"] = c.log_likelihood_test_probe
        item["time_weight_common"] = c.time_weight_common
        item["time_weight_test_probe"] = c.time_weight_test_probe
    return item
