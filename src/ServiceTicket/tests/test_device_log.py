"""device_log.py (root-level, shared by every stage) — GPU/CPU reporting.

Runs off-cluster with no torch/pynvml installed, which is exactly the "no CUDA"
branch the module has to survive: every helper must degrade to a printed note and
never raise. The GPU branches are covered with fake model objects.
"""

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

spec = importlib.util.spec_from_file_location("device_log", os.path.join(ROOT, "device_log.py"))
device_log = importlib.util.module_from_spec(spec)
spec.loader.exec_module(device_log)


@pytest.fixture(autouse=True)
def fresh_banner():
    """banner() memoizes per process — clear it so each test sees a real call."""
    device_log._BANNER_CACHE = None
    yield
    device_log._BANNER_CACHE = None


# --- banner / params ----------------------------------------------------------

def test_banner_reports_environment_and_is_memoized(capsys):
    first = device_log.banner()
    out = capsys.readouterr().out
    assert "[device]" in out
    assert "cuda_available=" in out
    assert "stage 04 GBM (sklearn)" in out          # per-stage verdict block

    second = device_log.banner()                     # cached: no second report
    assert second is first
    assert capsys.readouterr().out == ""


def test_banner_survives_missing_torch_and_onnxruntime(monkeypatch):
    monkeypatch.setattr(device_log, "_torch_info",
                        lambda: {"torch": None, "cuda_available": False})
    monkeypatch.setattr(device_log, "_onnx_info",
                        lambda: {"onnxruntime": None, "onnx_gpu": False})
    monkeypatch.setattr(device_log, "_spacy_info", lambda: {"spacy": None})
    info = device_log.banner()
    assert info["cuda_available"] is False


def test_params_are_flat_and_mlflow_safe():
    p = device_log.params()
    assert set(p) == {"device_torch", "device_cuda_available", "device_gpu_name",
                      "device_onnx_providers", "device_onnx_gpu"}
    assert isinstance(p["device_onnx_providers"], str)   # joined, never a list


def test_verdict_flips_with_the_environment():
    gpu = "\n".join(device_log._verdict({"cuda_available": True, "onnx_gpu": True}))
    cpu = "\n".join(device_log._verdict({"cuda_available": False, "onnx_gpu": False}))
    assert "stage 03 rerank + bi-encoder       -> GPU" in gpu
    assert "stage 01 embed (onnx backend)      -> GPU" in gpu
    assert "stage 03 rerank + bi-encoder       -> CPU" in cpu
    # sklearn / UMAP / spaCy have no GPU path in either environment
    assert "stage 04 GBM (sklearn)             -> CPU (no GPU path)" in gpu


# --- describe -----------------------------------------------------------------

class _FakeTorchModel:
    device = "cuda:0"


class _FakeSession:
    def get_providers(self):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]


class _FakeOnnxModel:
    """Shaped like SentenceTransformer(backend='onnx'): model[0].auto_model.model."""

    def __init__(self, providers):
        session = _FakeSession()
        session.get_providers = lambda: providers
        self._modules = [type("T", (), {"auto_model": type("O", (), {"model": session})()})()]

    def __getitem__(self, i):
        return self._modules[i]


def test_describe_reports_torch_device(capsys):
    assert device_log.describe(_FakeTorchModel(), "[ph03] cross-encoder") == "cuda:0"
    out = capsys.readouterr().out
    assert "-> GPU (torch device=cuda:0)" in out


def test_describe_reports_onnx_providers(capsys):
    model = _FakeOnnxModel(["CPUExecutionProvider"])
    assert device_log.describe(model, "[ph01] embed") == "onnx:CPUExecutionProvider"
    assert "-> CPU (onnx:CPUExecutionProvider" in capsys.readouterr().out


def test_describe_calls_a_cuda_onnx_provider_gpu(capsys):
    model = _FakeOnnxModel(["CUDAExecutionProvider", "CPUExecutionProvider"])
    device_log.describe(model, "[ph01] embed")
    assert "-> GPU (onnx:CUDAExecutionProvider" in capsys.readouterr().out


def test_describe_unknown_object_does_not_raise(capsys):
    assert device_log.describe(object(), "[phXX] mystery") is None
    assert "device unknown" in capsys.readouterr().out


def test_cpu_only_states_the_reason(capsys):
    assert device_log.cpu_only("[ph04] GBM score", "sklearn") == "cpu"
    assert "-> CPU by construction (sklearn)" in capsys.readouterr().out


# --- probe --------------------------------------------------------------------

def test_probe_times_the_block_and_reports_cpu_without_cuda(capsys):
    with device_log.probe("[ph01] encode") as p:
        pass
    assert p["label"] == "[ph01] encode"
    assert isinstance(p["seconds"], float)
    assert p.used_gpu is False
    assert "on CPU (no CUDA device in this process)" in capsys.readouterr().out


def test_probe_reports_the_verdict_even_when_the_block_raises(capsys):
    with pytest.raises(ValueError):
        with device_log.probe("[ph03] rerank"):
            raise ValueError("boom")
    assert "[ph03] rerank ran" in capsys.readouterr().out


def test_probe_result_metrics_are_namespaced_and_numeric():
    p = device_log.ProbeResult({"label": "[ph03] bi-encoder encode", "seconds": 1.5,
                                "gpu_util_max": 87, "gpu_mem_max_gb": 1.8,
                                "used": True})
    m = p.metrics()
    assert m == {"bi_encoder_encode_seconds": 1.5,
                 "bi_encoder_encode_gpu_util_max": 87,
                 "bi_encoder_encode_gpu_mem_max_gb": 1.8}      # label + bool dropped
    assert device_log.ProbeResult({"label": "x", "seconds": 2}).metrics("ph05_embed") == \
        {"ph05_embed_seconds": 2}


def test_used_gpu_is_false_when_the_gpu_sat_idle():
    idle = device_log.ProbeResult({"gpu_util_max": 0, "torch_peak_alloc_gb": 0.0})
    busy = device_log.ProbeResult({"gpu_util_max": 91, "torch_peak_alloc_gb": 1.2})
    assert idle.used_gpu is False           # CUDA present, but nothing ran on it
    assert busy.used_gpu is True


def test_probe_line_calls_out_an_idle_gpu():
    line = device_log._probe_line(
        device_log.ProbeResult({"label": "[ph01] encode", "seconds": 42.0,
                                "gpu_util_max": 0, "gpu_mem_max_gb": 0.4,
                                "torch_peak_alloc_gb": 0.0}), had_cuda=True)
    assert "GPU IDLE" in line
    assert "gpu_util max 0%" in line


def test_probe_line_confirms_real_gpu_work():
    line = device_log._probe_line(
        device_log.ProbeResult({"label": "[ph03] rerank", "seconds": 12.3,
                                "gpu_util_max": 87, "gpu_mem_max_gb": 1.8,
                                "torch_peak_alloc_gb": 0.64}), had_cuda=True)
    assert "GPU DID WORK" in line


def test_slug_strips_the_stage_tag():
    assert device_log._slug("[ph01] encode") == "encode"
    assert device_log._slug("[ph03] bi-encoder encode") == "bi_encoder_encode"
    assert device_log._slug("") == "probe"


def test_logging_can_be_silenced(monkeypatch, capsys):
    monkeypatch.setattr(device_log, "_ENABLED", False)
    device_log.banner()
    device_log.cpu_only("[ph04] GBM score", "sklearn")
    assert capsys.readouterr().out == ""
