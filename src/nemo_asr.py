"""Optional Japanese NeMo recognition in a short-lived, separately installed worker.

The bridge supplies only exact local audio crops. Stored ALSD/beam-4 decoding is
preserved; no prompt, hotword, ASR hypothesis, or reference enters this worker.
Raw returns remain evidence, never approval. A failed batch raises with its
partial diagnostic report and cannot become a successful partial transcript.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from src.workflow_state import file_hash, fingerprint, read_json, write_json

VERSION = "nemo-raw-window-worker-1"
ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PACKAGES = ("nemo_toolkit", "torch", "torchaudio", "transformers",
                    "numpy", "lightning", "omegaconf")


class NemoASRError(RuntimeError):
    """The complete available diagnostic survives temporary-directory cleanup."""

    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


def _setting(config: Any, name: str, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def _paths(config: Any) -> tuple[Path, Path]:
    model_value = _setting(config, "nemo_model")
    python_value = _setting(config, "nemo_python")
    if not model_value or not python_value:
        raise ValueError("NeMo requires explicit nemo_model and nemo_python paths")
    model = Path(model_value).expanduser().resolve()
    # Resolving a venv's bin/python symlink would bypass that venv at launch.
    python = Path(python_value).expanduser().absolute()
    if not model.is_file():
        raise ValueError(f"NeMo archive does not exist: {model}")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"NeMo Python is not executable: {python}")
    return model, python


def _distribution_identity() -> dict:
    """Metadata only: deliberately does not import Torch, NeMo, or initialize CUDA."""
    packages = {}
    for name in RUNTIME_PACKAGES:
        dist = importlib.metadata.distribution(name)
        metadata_text = dist.read_text("METADATA")
        if metadata_text is None:
            raise RuntimeError(f"Missing installed metadata for {name}")
        record = dist.read_text("RECORD")
        packages[name] = {"version": dist.version,
                          "metadata_sha256": hashlib.sha256(metadata_text.encode("utf-8")).hexdigest(),
                          "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest() if record else None}
    return {"python_version": sys.version, "packages": packages}


def runtime_identity(config: Any) -> dict:
    """Return a cache identity using the configured environment without GPU imports."""
    model, python = _paths(config)
    command = [str(python), "-m", "src.nemo_asr", "--runtime-identity"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                            timeout=60, check=False)
    if result.returncode:
        raise ValueError(f"NeMo environment metadata failed: {result.stderr[-2000:]}")
    try:
        runtime = json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise ValueError("NeMo environment returned invalid runtime metadata") from exc
    if set(runtime.get("packages", {})) != set(RUNTIME_PACKAGES):
        raise ValueError("NeMo environment metadata is incomplete")
    venv_config = python.parent.parent / "pyvenv.cfg"
    return {"version": VERSION, "bridge_sha256": file_hash(Path(__file__)),
            "model": str(model), "model_sha256": file_hash(model),
            "python": str(python), "python_resolved": str(python.resolve()),
            "python_sha256": file_hash(python.resolve()),
            "pyvenv_config_sha256": file_hash(venv_config) if venv_config.is_file() else None,
            "runtime": runtime}


def _clips(audio, sr: int, windows) -> tuple[dict, list[dict]]:
    import numpy as np

    if sr != 16000 or not isinstance(audio, np.ndarray) or audio.ndim != 1 or audio.dtype != np.float32:
        raise ValueError("NeMo requires original mono 16kHz float32 audio")
    arrays, jobs, seen = {}, [], set()
    for position, window in enumerate(windows):
        get = window.get if isinstance(window, dict) else lambda key: getattr(window, key)
        index, start, end = get("index"), get("start"), get("end")
        if (not isinstance(index, int) or isinstance(index, bool) or index in seen
                or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end):
            raise ValueError("Invalid or duplicate NeMo raw window")
        lo, hi = int(start * sr), int(end * sr)
        if not 0 <= lo < hi <= len(audio):
            raise ValueError("NeMo raw window is outside original audio")
        clip = audio[lo:hi].copy()
        if not np.isfinite(clip).all():
            raise ValueError("NeMo audio contains nonfinite samples")
        key = f"clip_{position:06d}"
        arrays[key] = clip
        jobs.append({"window": index, "start": start, "end": end, "array": key,
                     "sample_start": lo, "sample_end": hi, "sample_count": len(clip),
                     "float32_sha256": hashlib.sha256(clip.tobytes()).hexdigest()})
        seen.add(index)
    return arrays, jobs


def _raw_json(value):
    """Preserve raw hypotheses, including token/timestamp tensors and decoder state."""
    import numpy as np
    import torch

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite_float": str(value)}
    if isinstance(value, np.generic):
        return _raw_json(value.item())
    if isinstance(value, torch.Tensor):
        return {"type": "torch.Tensor", "dtype": str(value.dtype), "shape": list(value.shape),
                "values": _raw_json(value.detach().cpu().tolist())}
    if isinstance(value, np.ndarray):
        return {"type": "numpy.ndarray", "dtype": str(value.dtype), "shape": list(value.shape),
                "values": _raw_json(value.tolist())}
    if is_dataclass(value):
        return {"type": type(value).__module__ + "." + type(value).__qualname__,
                "fields": {field.name: _raw_json(getattr(value, field.name)) for field in fields(value)}}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            return {"type": "mapping", "items": [[_raw_json(k), _raw_json(v)] for k, v in value.items()]}
        return {key: _raw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_raw_json(item) for item in value]}
    if isinstance(value, list):
        return [_raw_json(item) for item in value]
    raise TypeError("Unsupported NeMo raw value: " + type(value).__qualname__)


def _one_text(raw) -> str:
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(getattr(raw[0], "text", None), str):
        raise ValueError("Expected exactly one stored-best NeMo hypothesis")
    return raw[0].text


def _validate_result(report: dict, spec: dict) -> list[str]:
    if not isinstance(report, dict) or report.get("request_key") != spec["request_key"]:
        raise ValueError("NeMo worker response is stale or malformed")
    if report.get("complete") is not True or report.get("error"):
        raise ValueError("NeMo worker did not complete: " + str(report.get("error")))
    if report.get("identity") != spec["identity"]:
        raise ValueError("NeMo worker runtime identity changed")
    rows = report.get("windows")
    if not isinstance(rows, list) or len(rows) != len(spec["windows"]):
        raise ValueError("NeMo worker returned only partial windows")
    values = []
    for row, job in zip(rows, spec["windows"]):
        if not isinstance(row, dict) or any(row.get(k) != v for k, v in job.items()):
            raise ValueError("NeMo worker window order, bounds or audio hash mismatch")
        if not isinstance(row.get("text"), str) or "raw_return" not in row:
            raise ValueError("NeMo worker lost its raw text or hypothesis evidence")
        raw = row["raw_return"]
        if (not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict)
                or raw[0].get("fields", {}).get("text") != row["text"]):
            raise ValueError("NeMo raw hypothesis does not support returned text")
        values.append(row["text"])
    return values


def _launch_worker(spec: dict, directory: Path) -> dict:
    request, response = directory / "request.json", directory / "response.json"
    write_json(request, spec)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", WANDB_MODE="disabled")
    started = time.monotonic()
    try:
        process = subprocess.run([spec["identity"]["python"], "-m", "src.nemo_asr", "--worker",
                                  str(request), str(response)], cwd=ROOT, env=env,
                                 text=True, capture_output=True, timeout=1800, check=False)
    except subprocess.TimeoutExpired as exc:
        report = read_json(response, {})
        report.update(complete=False, error="NeMo worker timed out", process_seconds=time.monotonic()-started)
        raise NemoASRError("NeMo worker timed out; no partial transcript accepted", report) from exc
    report = read_json(response, {})
    report.update(process_seconds=time.monotonic()-started, exit_code=process.returncode,
                  worker_stdout=process.stdout, worker_stderr=process.stderr,
                  worker_exited=True)
    if process.returncode:
        report["complete"] = False
        raise NemoASRError("NeMo worker failed: " + str(report.get("error", process.stderr[-1000:])), report)
    return report


def transcribe_windows(audio, sr: int, windows, config: Any, *,
                       expected_identity: dict | None = None) -> tuple[list[str], dict]:
    """Recognize ordered original windows; returned strings are untouched raw text.

    ``NemoASRError.report`` retains available raw evidence on any failure. The
    caller must keep that report if persisting failed attempts. Empty strings
    remain explicit outputs for the caller's unresolved source-fallback policy.
    """
    import numpy as np

    started = time.monotonic()
    arrays, jobs = _clips(audio, sr, windows)
    identity = runtime_identity(config)
    if expected_identity is not None and identity != expected_identity:
        raise NemoASRError("NeMo runtime changed after the ASR cache was prepared",
                           {"complete": False, "expected_identity": expected_identity, "identity": identity})
    with tempfile.TemporaryDirectory(prefix="autosub-nemo-") as temporary:
        directory = Path(temporary)
        clips_path = directory / "clips.npz"
        np.savez(clips_path, **arrays)
        spec = {"version": VERSION, "identity": identity, "windows": jobs,
                "clips": str(clips_path), "clips_sha256": file_hash(clips_path), "sample_rate": sr,
                "warden_admin_url": _setting(config, "warden_admin_url", "http://127.0.0.1:8089/admin"),
                "unload_warden_before_asr": _setting(config, "unload_warden_before_asr", True)}
        spec["request_key"] = fingerprint(spec)
        try:
            report = _launch_worker(spec, directory)
            values = _validate_result(report, spec)
            if file_hash(clips_path) != spec["clips_sha256"]:
                raise ValueError("NeMo original audio packet changed")
        except NemoASRError:
            raise
        except Exception as exc:
            failed_report = locals().get("report", {})
            if not isinstance(failed_report, dict):
                failed_report = {"malformed_response": failed_report}
            failed_report.update(complete=False, error=f"{type(exc).__name__}: {exc}")
            raise NemoASRError(str(exc), failed_report) from exc
    report["bridge_seconds"] = time.monotonic() - started
    return values, report


def _worker(request: Path, response: Path) -> None:
    spec = read_json(request)
    epoch = time.monotonic()
    report = {"version": VERSION, "family": "r", "request_key": spec["request_key"],
              "identity": spec["identity"], "windows": [], "complete": False, "error": None,
              "source_status": "unresolved", "text_inputs_supplied": False}
    try:
        before = time.monotonic()
        import torch
        import numpy as np
        from omegaconf import OmegaConf
        from nemo.collections.asr.models import ASRModel
        from src.warden import ensure_gpu_headroom

        report["import_seconds"] = time.monotonic() - before
        if not torch.cuda.is_available():
            raise RuntimeError("NeMo optional worker requires CUDA")
        if (_distribution_identity() != spec["identity"]["runtime"]
                or file_hash(Path(spec["identity"]["model"])) != spec["identity"]["model_sha256"]
                or file_hash(Path(__file__)) != spec["identity"]["bridge_sha256"]
                or file_hash(Path(spec["clips"])) != spec["clips_sha256"]):
            raise ValueError("NeMo prepared runtime, model or audio changed")
        torch.set_num_threads(4)
        torch.manual_seed(20260913)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        before = time.monotonic()
        free = ensure_gpu_headroom(required_gb=8., hard_floor_gb=6.,
                                   admin_url=spec["warden_admin_url"],
                                   enabled=spec["unload_warden_before_asr"], caller="NeMo ASR")
        report.update(admission_seconds=time.monotonic()-before, gpu_free_before_bytes=free,
                      admission_policy={"required_gb": 8., "hard_floor_gb": 6.,
                                        "enabled": spec["unload_warden_before_asr"],
                                        "admin_url": spec["warden_admin_url"]})
        torch.cuda.reset_peak_memory_stats()
        before = time.monotonic()
        model = ASRModel.restore_from(spec["identity"]["model"], map_location=torch.device("cuda"))
        model = model.to(dtype=torch.float32).eval()
        model.freeze()
        torch.cuda.synchronize()
        report["model_load_seconds"] = time.monotonic() - before
        cfg = OmegaConf.to_container(model.cfg, resolve=True)
        report["restored_model_config"] = cfg
        decoding = cfg["decoding"]
        if (decoding["strategy"] != "alsd" or decoding["beam"]["beam_size"] != 4
                or not decoding["beam"]["return_best_hypothesis"]):
            raise ValueError("NeMo archive must preserve original ALSD beam4 stored-best decoding")
        report["numeric_runtime"] = {"dtype": str(next(model.parameters()).dtype),
                                     "float32_matmul_precision": torch.get_float32_matmul_precision(),
                                     "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                     "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                     "batch_size": 1, "timestamps": None, "torch_threads": 4}
        with np.load(spec["clips"], allow_pickle=False) as arrays:
            if set(arrays.files) != {job["array"] for job in spec["windows"]}:
                raise ValueError("NeMo audio packet has missing or extra windows")
            for job in spec["windows"]:
                clip = arrays[job["array"]]
                if (clip.dtype != np.float32 or clip.shape != (job["sample_count"],)
                        or hashlib.sha256(clip.tobytes()).hexdigest() != job["float32_sha256"]):
                    raise ValueError("NeMo raw crop hash or format changed")
                before = time.monotonic()
                with torch.inference_mode():
                    raw = model.transcribe([clip], batch_size=1, return_hypotheses=True,
                                           num_workers=0, verbose=False, timestamps=None)
                    torch.cuda.synchronize()
                seconds = time.monotonic() - before
                # Persist every raw response before validating cardinality/text.
                row = dict(job, raw_return=_raw_json(raw), decode_seconds=seconds)
                report["windows"].append(row)
                write_json(response, report)
                row["text"] = _one_text(raw)
                write_json(response, report)
                del raw
        if (file_hash(Path(spec["identity"]["model"])) != spec["identity"]["model_sha256"]
                or file_hash(Path(spec["clips"])) != spec["clips_sha256"]
                or file_hash(Path(__file__)) != spec["identity"]["bridge_sha256"]):
            raise ValueError("NeMo model, implementation or original audio changed during inference")
        report["complete"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if "torch" in locals() and torch.cuda.is_initialized():
            report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["worker_seconds"] = time.monotonic() - epoch
        report["decode_seconds"] = sum(row["decode_seconds"] for row in report["windows"])
        write_json(response, report)
    if not report["complete"]:
        raise RuntimeError(report["error"])


if __name__ == "__main__":
    if sys.argv[1:] == ["--runtime-identity"]:
        print(json.dumps(_distribution_identity()))
    elif len(sys.argv) == 4 and sys.argv[1] == "--worker":
        _worker(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit("Internal NeMo worker: use transcribe_windows from the pipeline")
