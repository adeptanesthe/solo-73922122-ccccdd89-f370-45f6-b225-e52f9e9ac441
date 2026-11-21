"""API 请求模型（Pydantic v2）。

allow_inf_nan=False 使 NaN / Inf 在 schema 层即被拒绝；
app.gp 内部还有一层非有限值检查作为纵深防御。
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_OBSERVATIONS = 1000
MAX_CANDIDATES = 500
MAX_SUSPICIOUS_TIMES = 20


class SelectRequest(BaseModel):
    """选点请求体。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    measured_times: list[float] = Field(
        default_factory=list, max_length=MAX_OBSERVATIONS,
        description="已测时刻",
    )
    measured_values: list[float] = Field(
        default_factory=list, max_length=MAX_OBSERVATIONS,
        description="已测温度，与已测时刻一一对应",
    )
    candidate_times: list[float] = Field(
        min_length=2, max_length=MAX_CANDIDATES,
        description="候选采样时刻",
    )
    length_scale: float = Field(gt=0, description="平方指数核长度尺度 ℓ")
    noise_std: float = Field(ge=0, description="观测噪声标准差 σ_n")
    signal_variance: float = Field(default=1.0, gt=0, description="核信号方差")
    min_interval: float = Field(default=0.0, ge=0, description="两个新采样点之间的最小时间间隔")

    @model_validator(mode="after")
    def _check_aligned(self) -> "SelectRequest":
        if len(self.measured_times) != len(self.measured_values):
            raise ValueError("已测时刻与已测温度数量不一致")
        return self


class AttributeRequest(BaseModel):
    """突变归因请求体。

    ``reference_*`` 是已测数据（参考探头），``test_*`` 是待检探头读数；
    两支探头共享同一个平方指数核 GP，每条读数的噪声独立且标准差同为 ``noise_std``。
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    reference_times: list[float] = Field(
        default_factory=list, max_length=MAX_OBSERVATIONS,
        description="参考探头（已测数据）时刻",
    )
    reference_values: list[float] = Field(
        default_factory=list, max_length=MAX_OBSERVATIONS,
        description="参考探头温度，与 reference_times 一一对应",
    )
    test_times: list[float] = Field(
        default_factory=list, max_length=MAX_OBSERVATIONS,
        description="待检探头时刻",
    )
    test_values: list[float] = Field(
        default_factory=list, max_length=MAX_OBSERVATIONS,
        description="待检探头温度，与 test_times 一一对应",
    )
    suspicious_times: list[float] = Field(
        min_length=1, max_length=MAX_SUSPICIOUS_TIMES,
        description="可疑阶跃时刻（1..20 个，互不相同）",
    )
    length_scale: float = Field(gt=0, description="共享平方指数核长度尺度 ℓ")
    noise_std: float = Field(gt=0, description="每条读数的独立高斯噪声标准差 σ_n（> 0）")
    signal_variance: float = Field(gt=0, description="共享核信号方差")
    step_prior_std: float = Field(gt=0, description="阶跃幅度的零均值高斯先验标准差")

    @model_validator(mode="after")
    def _check_aligned(self) -> "AttributeRequest":
        if len(self.reference_times) != len(self.reference_values):
            raise ValueError("参考探头时刻与温度数量不一致")
        if len(self.test_times) != len(self.test_values):
            raise ValueError("待检探头时刻与温度数量不一致")
        return self
