"""API 层测试：正常流程、页面服务与整批拒绝行为。"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "measured_times": [0, 1, 2, 3, 5, 8, 13],
    "measured_values": [25.3, 196.1, 330.8, 433.5, 578.2, 695.1, 769.9],
    "candidate_times": [0.5, 1.5, 2.5, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5,
                        8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 14, 14.5, 15, 15.5],
    "length_scale": 2.0,
    "noise_std": 5.0,
    "signal_variance": 100000.0,
    "min_interval": 1.0,
}


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_page_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "高温炉温度实验工具" in r.text
    assert "突变归因" in r.text


def test_select_ok():
    r = client.post("/api/select", json=VALID_PAYLOAD)
    assert r.status_code == 200, r.text
    data = r.json()
    # 选中两个不同的候选时刻
    assert len(data["selected_times"]) == 2
    assert data["selected_times"][0] != data["selected_times"][1]
    assert len(data["selected_indices"]) == 2
    # 总量自洽
    assert data["total_variance_before"] > data["total_variance_after"]
    assert abs(data["total_variance_before"] - data["total_variance_after"]
               - data["total_reduction"]) < 1e-6 * data["total_variance_before"]
    # 候选明细完整且选后方差不超过选前
    cands = data["candidates"]
    assert len(cands) == len(VALID_PAYLOAD["candidate_times"])
    for c in cands:
        assert c["variance_after"] <= c["variance_before"] + 1e-6
    # 已测时刻 5.0 对应的候选点被排除
    c5 = next(c for c in cands if c["time"] == 5.0)
    assert not c5["eligible"] and c5["exclusion_reason"]
    # 选中的点必须合法
    for idx in data["selected_indices"]:
        assert cands[idx]["eligible"] and cands[idx]["selected"]
    assert 0 < data["n_pairs_feasible"] <= data["n_pairs_total"]


def test_nan_in_candidates_rejected():
    payload = dict(VALID_PAYLOAD)
    raw = json.dumps(payload).replace("0.5", "NaN", 1)  # 构造含 NaN 的非法 JSON 字面量
    r = client.post("/api/select", content=raw,
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert r.json()["detail"]["code"] in ("INVALID_INPUT", "NON_FINITE_INPUT")


def test_non_psd_kernel_rejected():
    payload = dict(VALID_PAYLOAD, measured_times=[1.0, 1.0],
                   measured_values=[100.0, 100.0], noise_std=0.0)
    r = client.post("/api/select", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "KERNEL_NOT_PSD"


def test_no_feasible_combination_rejected():
    payload = dict(VALID_PAYLOAD, min_interval=1e6)
    r = client.post("/api/select", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "NO_FEASIBLE_COMBINATION"


def test_mismatched_measured_lengths_rejected():
    payload = dict(VALID_PAYLOAD, measured_values=[1.0, 2.0])
    r = client.post("/api/select", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "INVALID_INPUT"


def test_invalid_parameters_rejected():
    for field, bad in [("length_scale", 0.0), ("noise_std", -1.0),
                       ("signal_variance", -2.0), ("min_interval", -0.5)]:
        r = client.post("/api/select", json=dict(VALID_PAYLOAD, **{field: bad}))
        assert r.status_code == 422, field
        assert r.json()["detail"]["code"] == "INVALID_INPUT"


def test_error_response_has_no_partial_result():
    payload = dict(VALID_PAYLOAD, min_interval=1e6)
    r = client.post("/api/select", json=payload)
    body = r.json()
    assert "candidates" not in body and "selected_times" not in body


# ---------------- 突变归因 API ----------------

ATTR_PAYLOAD = {
    "reference_times": [0, 1, 2, 3, 5, 8, 13],
    "reference_values": [25.3, 196.1, 330.8, 433.5, 578.2, 695.1, 769.9],
    "test_times": [0, 1, 2, 3, 5, 8, 13],
    "test_values": [22.1, 201.5, 336.0, 428.7, 972.4, 1091.0, 1168.3],
    "suspicious_times": [3, 5, 8, 10],
    "length_scale": 2.0,
    "noise_std": 5.0,
    "signal_variance": 100000.0,
    "step_prior_std": 400.0,
}


def test_attribute_ok():
    r = client.post("/api/attribute", json=ATTR_PAYLOAD)
    assert r.status_code == 200, r.text
    data = r.json()
    probs = data["posterior_probabilities"]
    assert set(probs) == {"no_step", "common_step", "test_probe_step"}
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert data["threshold"] == 0.9
    # 示例数据在 t=5 之后待检探头抬升，参考探头未抬升
    assert data["decision"] == "test_probe_step"
    assert data["conclusive"] is True
    assert data["best_step_time"] == 5.0
    assert data["n_valid_step_times"] == 4
    assert len(data["candidates"]) == 4
    for c in data["candidates"]:
        assert c["valid"]
        assert c["reference_before"] >= 1 and c["reference_after"] >= 1
        assert c["test_before"] >= 1 and c["test_after"] >= 1
        assert 0.0 <= c["time_weight_test_probe"] <= 1.0


def test_attribute_invalid_times_rejected():
    payload = dict(ATTR_PAYLOAD, suspicious_times=list(range(21)))
    r = client.post("/api/attribute", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "INVALID_INPUT"
    payload = dict(ATTR_PAYLOAD, suspicious_times=[5.0, 5.0])
    r = client.post("/api/attribute", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "INVALID_INPUT"


def test_attribute_zero_noise_rejected():
    r = client.post("/api/attribute", json=dict(ATTR_PAYLOAD, noise_std=0))
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "INVALID_INPUT"


def test_attribute_mismatched_lengths_rejected():
    r = client.post("/api/attribute", json=dict(ATTR_PAYLOAD, test_values=[1.0]))
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "INVALID_INPUT"


def test_attribute_no_valid_step_time_returns_diagnostics():
    payload = dict(ATTR_PAYLOAD, suspicious_times=[0, 20])
    r = client.post("/api/attribute", json=payload)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["code"] == "NO_VALID_STEP_TIME"
    assert len(detail["candidates"]) == 2
    assert all(not c["valid"] and c["reason"] for c in detail["candidates"])
    assert "posterior_probabilities" not in detail


def test_attribute_unknown_field_rejected():
    r = client.post("/api/attribute", json=dict(ATTR_PAYLOAD, bogus=1))
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "INVALID_INPUT"


def test_attribute_non_psd_rejected():
    payload = dict(ATTR_PAYLOAD, reference_times=[5.0, 5.0],
                   reference_values=[1.0, 1.0], test_times=[5.0, 5.0],
                   test_values=[1.0, 1.0], suspicious_times=[5.0], noise_std=1e-9)
    r = client.post("/api/attribute", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "KERNEL_NOT_PSD"
