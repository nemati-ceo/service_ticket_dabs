"""device_log.py — report where each compute step ACTUALLY ran: GPU or CPU.

The cluster is a GPU runtime (g5.xlarge, 18.0.x-gpu-ml), but nothing in this repo
pins a device, so the cluster spec proves nothing about where the work happened:

  * torch models (stage 03 cross-encoder + bi-encoder, stage 05 embedder, stage 02
    eval) land on CUDA by sentence-transformers autodetect — GPU when torch sees a
    device, silently CPU when it doesn't.
  * stage 01 runs the MiniLM with `backend: "onnx"`, so it uses whatever execution
    provider onnxruntime was built for. The plain `onnxruntime` wheel is CPU-only;
    only `onnxruntime-gpu` gives CUDAExecutionProvider.
  * stage 04 (sklearn GBM), stage 05 UMAP/HDBSCAN and stage 01b (spaCy/presidio)
    have no GPU path at all.

Three entry points, all printing `[device] ...` lines into the Databricks driver log:

    banner()                          # once per process: hardware + execution providers
    describe(model, "ph01 embed")     # where a loaded model actually sits
    with probe("ph01 encode") as p:   # did the GPU do work during this block?
        ...
    p.result                          # dict, ready for ml.log_params/log_metrics

`probe` is the one that answers "are we really using the GPU": it samples GPU
utilization while the block runs, so an idle GPU next to a busy CPU shows up as
`gpu_util max 0%` instead of being hidden behind `cuda_available=True`.

Best-effort by design, exactly like mlflow_utils: no torch, no pynvml, an exotic
model object — it prints a note and moves on, never breaks a run.

Quick check on Databricks without running the pipeline (notebook cell or web
terminal on the cluster):

    %run ./device_log            # notebook: prints the banner on import
    python device_log.py         # terminal: same banner
"""

import os
import threading
import time
from contextlib import contextmanager

# Env escape hatch: set SERVICE_TICKET_DEVICE_LOG=0 to silence every line here.
_ENABLED = os.environ.get("SERVICE_TICKET_DEVICE_LOG", "1").lower() not in ("0", "false", "no")

_BANNER_CACHE = None            # banner() is memoized: once per process, not per stage
_SAMPLE_INTERVAL_S = 0.5        # GPU utilization sampling period inside probe()


def _say(msg):
    if _ENABLED:
        print(f"[device] {msg}")


# --- environment ---------------------------------------------------------------

def _torch_info():
    """torch build + visible CUDA devices. {} when torch is absent."""
    try:
        import torch
    except Exception as e:
        return {"torch": None, "torch_error": str(e), "cuda_available": False}
    info = {
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,        # None on a CPU-only wheel
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": 0,
        "devices": [],
    }
    if info["cuda_available"]:
        try:
            info["device_count"] = torch.cuda.device_count()
            for i in range(info["device_count"]):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append(
                    {"index": i, "name": props.name,
                     "total_gb": round(props.total_memory / 1e9, 1)})
        except Exception as e:
            info["devices_error"] = str(e)
    return info


def _onnx_info():
    """onnxruntime version + execution providers. This is what decides stage 01."""
    try:
        import onnxruntime as ort
    except Exception as e:
        return {"onnxruntime": None, "onnxruntime_error": str(e), "onnx_gpu": False}
    try:
        providers = list(ort.get_available_providers())
    except Exception as e:
        return {"onnxruntime": ort.__version__, "onnxruntime_error": str(e), "onnx_gpu": False}
    return {"onnxruntime": ort.__version__,
            "onnx_providers": providers,
            "onnx_gpu": any("CUDA" in p or "Tensorrt" in p or "TensorRT" in p for p in providers)}


def _spacy_info():
    """spaCy GPU allocator — stage 01b is CPU unless this is cupy/torch."""
    try:
        import spacy
        import thinc.api
    except Exception as e:
        return {"spacy": None, "spacy_error": str(e)}
    try:
        allocator = thinc.api.get_current_ops().name
    except Exception:
        allocator = None
    return {"spacy": spacy.__version__, "spacy_ops": allocator}


def _verdict(info):
    """One-line-per-stage summary of where the work will land, given this environment."""
    torch_dev = "GPU" if info.get("cuda_available") else "CPU"
    onnx_dev = "GPU" if info.get("onnx_gpu") else "CPU"
    return [
        f"stage 01 embed (onnx backend)      -> {onnx_dev}",
        "stage 01b PII (spacy/presidio)     -> CPU (no GPU path)",
        f"stage 02 eval embed (torch)        -> {torch_dev}",
        f"stage 03 rerank + bi-encoder       -> {torch_dev}",
        "stage 04 GBM (sklearn)             -> CPU (no GPU path)",
        f"stage 05 embed (torch)             -> {torch_dev}",
        "stage 05 UMAP/HDBSCAN              -> CPU (no GPU path)",
    ]


def banner(force=False):
    """Print the hardware/execution-provider report ONCE per process. Returns the info dict.

    Called at the top of a pipeline run so the driver log opens with the answer to
    "is this run on GPU at all", before any stage has spent a minute encoding.
    """
    global _BANNER_CACHE
    if _BANNER_CACHE is not None and not force:
        return _BANNER_CACHE

    info = {}
    info.update(_torch_info())
    info.update(_onnx_info())
    info.update(_spacy_info())
    _BANNER_CACHE = info

    _say("-" * 66)
    _say(f"torch={info.get('torch')} cuda_build={info.get('torch_cuda_build')} "
         f"cuda_available={info.get('cuda_available')} devices={info.get('device_count', 0)}")
    for d in info.get("devices", []):
        _say(f"  cuda:{d['index']} {d['name']} ({d['total_gb']} GB)")
    if not info.get("cuda_available"):
        _say("  no CUDA device visible to torch — every torch stage runs on CPU")
    _say(f"onnxruntime={info.get('onnxruntime')} providers={info.get('onnx_providers')}")
    if not info.get("onnx_gpu"):
        _say("  onnxruntime has no CUDA provider (CPU wheel) — stage 01 encodes on CPU. "
             "Install onnxruntime-gpu, or set model.backend: torch in config.yml, to move it.")
    _say(f"spacy={info.get('spacy')} ops={info.get('spacy_ops')}")
    for line in _verdict(info):
        _say(f"  {line}")
    _say("-" * 66)
    return info


def params():
    """Flat, MLflow-safe view of the environment for ml.log_params()."""
    info = banner()
    return {
        "device_torch": info.get("torch"),
        "device_cuda_available": info.get("cuda_available"),
        "device_gpu_name": (info.get("devices") or [{}])[0].get("name"),
        "device_onnx_providers": ",".join(info.get("onnx_providers") or []),
        "device_onnx_gpu": info.get("onnx_gpu"),
    }


# --- where a specific loaded model sits ----------------------------------------

def _onnx_providers_of(model):
    """Providers of the onnxruntime session buried inside an ONNX-backend model, or None.

    sentence-transformers wraps optimum's ORTModel, which wraps an InferenceSession.
    The nesting has moved between versions, so probe a few known paths rather than
    hard-coding one and printing "unknown" after the next upgrade.
    """
    candidates = [model]
    try:
        candidates.append(model[0])                       # models.Transformer
        candidates.append(model[0].auto_model)            # ORTModel
        candidates.append(model[0].auto_model.model)      # InferenceSession
    except Exception:
        pass
    for obj in candidates:
        getter = getattr(obj, "get_providers", None)
        if callable(getter):
            try:
                return list(getter())
            except Exception:
                continue
    return None


def _torch_device_of(model):
    """Device string of a torch model, or None if it isn't one (or has no parameters)."""
    dev = getattr(model, "device", None)
    if dev is not None and not callable(dev):
        return str(dev)
    for attr in ("model", "_target_device"):
        inner = getattr(model, attr, None)
        inner_dev = getattr(inner, "device", None)
        if inner_dev is not None:
            return str(inner_dev)
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return None


def describe(model, label):
    """Print (and return) where `model` actually lives: 'cuda:0', 'cpu', 'onnx:CPUExecution...'.

    Call right after loading a model, so the log says where the next expensive block
    is going to run instead of leaving it to be inferred from the wall clock.
    """
    providers = _onnx_providers_of(model)
    if providers:
        gpu = any("CUDA" in p or "TensorRT" in p for p in providers)
        where = f"onnx:{providers[0]}"
        _say(f"{label} model -> {'GPU' if gpu else 'CPU'} ({where}, providers={providers})")
        return where
    dev = _torch_device_of(model)
    if dev:
        _say(f"{label} model -> {'GPU' if dev.startswith('cuda') else 'CPU'} (torch device={dev})")
        return dev
    _say(f"{label} model -> device unknown ({type(model).__name__} exposes neither "
         f"a torch device nor an onnxruntime session)")
    return None


def cpu_only(label, why):
    """State outright that a step has no GPU path (sklearn, UMAP/HDBSCAN, spaCy).

    Without this, a step that is silently CPU looks identical in the log to a GPU
    step whose device detection failed.
    """
    _say(f"{label} -> CPU by construction ({why})")
    return "cpu"


# --- did the GPU actually do work? ---------------------------------------------

class _Sampler:
    """Background poller for GPU utilization/memory while a block runs.

    torch reports what THIS process allocated; nvml reports what the DEVICE did. Both
    matter: allocated-but-idle memory (util 0%) means the model was moved to the GPU
    and then fed nothing, which is exactly the failure this module exists to catch.
    """

    def __init__(self):
        self.max_util = None
        self.max_mem_gb = None
        self._stop = threading.Event()
        self._thread = None
        self._handle = None
        self._nvml = None

    def start(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            return self                       # no pynvml: torch memory stats still apply
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / 1e9
                self.max_util = util if self.max_util is None else max(self.max_util, util)
                self.max_mem_gb = mem if self.max_mem_gb is None else max(self.max_mem_gb, mem)
            except Exception:
                return
            self._stop.wait(_SAMPLE_INTERVAL_S)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * _SAMPLE_INTERVAL_S)
        try:
            if self._nvml is not None:
                self._nvml.nvmlShutdown()
        except Exception:
            pass
        return self


def _slug(label):
    """'[ph01] encode' -> 'encode'. The stage tag is already added by mlflow_utils."""
    out = []
    for ch in label.split("]")[-1].strip().lower():
        out.append(ch if ch.isalnum() else "_")
    return "".join(out).strip("_") or "probe"


class ProbeResult(dict):
    """Result of a probe() block. A plain dict so it drops straight into log_params."""

    @property
    def used_gpu(self):
        """True only when the GPU measurably did work — not merely that CUDA exists."""
        return bool(self.get("gpu_util_max")) or bool(self.get("torch_peak_alloc_gb"))

    def metrics(self, prefix=None):
        """Numeric fields as MLflow metrics, named after the probe.

        A stage can hold several probes (stage 03 encodes candidates AND reranks), so
        the label is folded into the key — otherwise the second probe overwrites the
        first under the same stage-level metric name.
        """
        prefix = prefix or _slug(self.get("label", "probe"))
        return {f"{prefix}_{k}": v for k, v in self.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}


@contextmanager
def probe(label):
    """Time a compute block and report whether the GPU did any work during it.

    Yields a ProbeResult that is filled in on exit:
        {"label", "seconds", "gpu_util_max", "gpu_mem_max_gb", "torch_peak_alloc_gb"}
    """
    result = ProbeResult({"label": label})
    torch = None
    try:
        import torch as _t
        if _t.cuda.is_available():
            torch = _t
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        torch = None

    sampler = _Sampler().start() if torch is not None else None
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        seconds = time.perf_counter() - t0
        result["seconds"] = round(seconds, 2)
        if sampler is not None:
            sampler.stop()
            result["gpu_util_max"] = sampler.max_util
            result["gpu_mem_max_gb"] = (round(sampler.max_mem_gb, 2)
                                        if sampler.max_mem_gb is not None else None)
        if torch is not None:
            try:
                result["torch_peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
            except Exception:
                pass
        _say(_probe_line(result, torch is not None))


def _probe_line(result, had_cuda):
    """Format one probe verdict line."""
    head = f"{result['label']} ran {result['seconds']}s"
    if not had_cuda:
        return f"{head} on CPU (no CUDA device in this process)"
    bits = []
    if result.get("gpu_util_max") is not None:
        bits.append(f"gpu_util max {result['gpu_util_max']}%")
    if result.get("gpu_mem_max_gb") is not None:
        bits.append(f"gpu_mem max {result['gpu_mem_max_gb']} GB")
    if result.get("torch_peak_alloc_gb") is not None:
        bits.append(f"torch peak alloc {result['torch_peak_alloc_gb']} GB")
    detail = " | ".join(bits) if bits else "no GPU counters available (pynvml missing)"
    if result.used_gpu:
        return f"{head} | {detail} -> GPU DID WORK"
    return (f"{head} | {detail} -> GPU IDLE: a CUDA device is present but this block "
            f"did not touch it (CPU-bound step, or a CPU-only backend)")


if __name__ == "__main__":
    banner()
