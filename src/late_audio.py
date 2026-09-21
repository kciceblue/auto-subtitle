"""Late, blind local acoustic observations; never subtitle replacements.

``preserve_master`` and ``acquire(..., execute=False)`` perform CPU preparation.
Only explicit execution starts native ASR/separation workers. Crop transcripts
include neighbours: ownership is a request association, not word alignment.
Every artifact is append-only. A started, interrupted acquisition is not retried.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit

VERSION = 1
ASR_RATE = 16000
MAX_CROP_SECONDS = 30
MAX_DEEP_OWNERS = 20
ENGINES = ("zipformer-ja", "qwen3-asr-1.7b")
PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ASR_ROOT = Path.home() / "HF/asr-models"
VAD_OPTIONS = {"threshold": 0.5, "min_speech_duration_ms": 0,
               "min_silence_duration_ms": 300, "speech_pad_ms": 0}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def _read(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pin(path: Path | str) -> dict:
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("Expected a regular local artifact")
    return {"path": str(path), "sha256": file_hash(path)}


def _check(pin: dict) -> Path:
    path = Path(pin["path"])
    if file_hash(path) != pin["sha256"]:
        raise ValueError("Pinned artifact changed: " + str(path))
    return path


def _interpreter_path(value: Path | str) -> str:
    """Preserve the venv executable symlink: resolving it selects system Python."""
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("Python interpreter is not an executable local file")
    return str(path)


def _asr_import_preflight() -> dict:
    """Exercise direct-script production imports without loading native models."""
    sys.path.insert(0, str(PROJECT))
    import soundfile
    import numpy
    from src.adjudicate import QWEN3_DIR, ZIPFORMER_DIR
    return {"executable": sys.executable, "prefix": sys.prefix, "base_prefix": sys.base_prefix,
            "soundfile": soundfile.__version__, "numpy": numpy.__version__,
            "qwen_asr_distribution": importlib.metadata.version("qwen-asr"),
            "sherpa_onnx_distribution": importlib.metadata.version("sherpa-onnx"),
            "production_imports_ready": True, "model_loads": 0, "forwards": 0}


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    return float(value)


def _audio_info(path: Path) -> dict:
    import soundfile as sf
    info = sf.info(path)
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise ValueError("Audio must contain complete nonempty frames")
    return {"frames": int(info.frames), "sample_rate": int(info.samplerate),
            "channels": int(info.channels), "subtype": info.subtype,
            "seconds": info.frames / info.samplerate}


def _command(command: list[str], timeout: float = 600) -> None:
    subprocess.run(command, check=True, capture_output=True, timeout=timeout)


def preserve_master(media: Path | str, out: Path | str, *, audio_stream: int = 0) -> dict:
    """Decode selected audio stream once, preserving native rate/channels as FLOAT WAV.

    ``audio_stream`` is the zero-based audio-stream ordinal. Owner times use the
    decoded stream's zero-based timeline; the container start offset is retained.
    Reusing a completed directory verifies the original bytes and decoded master.
    """
    if type(audio_stream) is not int or audio_stream < 0:
        raise ValueError("audio_stream must be a nonnegative integer")
    media = Path(media).expanduser().resolve(strict=True)
    out = Path(out).expanduser().resolve()
    manifest_path = out / "master.json"
    source_pin = _pin(media)
    if manifest_path.exists():
        manifest = _read(manifest_path)
        if manifest["media"] != source_pin or manifest["audio_stream_ordinal"] != audio_stream:
            raise ValueError("Master directory belongs to different input audio")
        _check(manifest["master"])
        if _audio_info(Path(manifest["master"]["path"])) != manifest["audio"]:
            raise ValueError("Master audio geometry changed")
        return manifest
    out.mkdir(parents=True, exist_ok=True)
    if (out / "decode-started.json").exists():
        raise RuntimeError("An incomplete master decode cannot be silently rerun")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", f"a:{audio_stream}",
        "-show_entries", "stream=index,codec_name,sample_rate,channels,channel_layout,start_time,time_base",
        "-of", "json", str(media)], check=True, capture_output=True, text=True, timeout=60)
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError("Selected audio stream is missing or ambiguous")
    stream = streams[0]
    rate, channels = int(stream["sample_rate"]), int(stream["channels"])
    if rate <= 0 or channels <= 0:
        raise ValueError("Native sample rate/channel count is invalid")
    offset = float(stream.get("start_time", 0))
    if not math.isfinite(offset):
        raise ValueError("Container audio start time is not finite")
    master = out / "native-float.wav"
    command = ["ffmpeg", "-n", "-v", "error", "-i", str(media), "-map", f"0:{stream['index']}",
        "-vn", "-map_metadata", "-1", "-c:a", "pcm_f32le", "-ar", str(rate), "-ac", str(channels),
        "-rf64", "auto", str(master)]
    _write(out / "decode-started.json", {"started_utc": _now(), "media": source_pin, "command": command})
    _command(command)
    if _pin(media) != source_pin:
        raise ValueError("Media changed while decoding")
    info = _audio_info(master)
    if info["sample_rate"] != rate or info["channels"] != channels or info["subtype"] != "FLOAT":
        raise ValueError("Decoded master does not preserve the native audio contract")
    master.chmod(0o444)
    manifest = {"version": VERSION, "media": source_pin, "master": _pin(master), "audio": info,
        "audio_stream_ordinal": audio_stream, "stream": stream, "container_audio_start_seconds": offset,
        "owner_timebase": "decoded_audio_zero_based", "owner_end_rounding_tolerance_seconds": 0.001, "command": command, "finished_utc": _now()}
    _write(manifest_path, manifest)
    return manifest


def validate_owners(owners: list[dict], duration: float) -> list[dict]:
    if not isinstance(owners, list) or not owners:
        raise ValueError("owners must be a nonempty list")
    result, seen = [], set()
    for row in owners:
        if not isinstance(row, dict) or set(row) != {"id", "start", "end"}:
            raise ValueError("Pass only owner id/start/end, never subtitle text")
        if type(row["id"]) is not int or row["id"] <= 0 or row["id"] in seen:
            raise ValueError("Owner IDs must be unique positive integers")
        start, end = _number(row["start"], "start"), _number(row["end"], "end")
        if not 0 <= start < end <= duration + 0.001:
            raise ValueError("Owner extends beyond decoded audio")
        if end - start > MAX_CROP_SECONDS:
            raise ValueError("Owners longer than30s require an explicitly budgeted subdivision")
        if result and start < result[-1]["start"]:
            raise ValueError("Owners must be ordered by start time")
        seen.add(row["id"])
        result.append({"id": row["id"], "start": start, "end": min(end, duration)})
    return result


def _bounds(row: dict, frames: int, before: float, after: float, speech=None) -> tuple[int, int]:
    core_start = math.floor(row["start"] * ASR_RATE)
    core_end = min(frames, math.ceil(row["end"] * ASR_RATE))
    remaining = max(0, MAX_CROP_SECONDS * ASR_RATE - (core_end - core_start))
    requested = (before + after) * ASR_RATE
    scale = min(1.0, remaining / requested) if requested else 0
    start = max(0, core_start - round(before * ASR_RATE * scale))
    end = min(frames, core_end + round(after * ASR_RATE * scale))
    if speech is not None:
        # Gap centres are only acoustic boundary hints; never remove owner frames.
        cuts = [0, frames]
        last = 0
        for segment in speech:
            a, b = segment["start"], segment["end"]
            if a > last:
                cuts.append((last + a) // 2)
            last = max(last, b)
        if frames > last:
            cuts.append((last + frames) // 2)
        left = [x for x in cuts if x <= core_start and abs(x - start) <= ASR_RATE]
        right = [x for x in cuts if x >= core_end and abs(x - end) <= ASR_RATE]
        candidate_start = min(left, key=lambda x: (abs(x - start), x)) if left else start
        candidate_end = min(right, key=lambda x: (abs(x - end), x)) if right else end
        if candidate_end - candidate_start <= MAX_CROP_SECONDS * ASR_RATE:
            start, end = candidate_start, candidate_end
    if not 0 <= start <= core_start < core_end <= end <= frames:
        raise ValueError("Crop failed complete owner containment")
    return start, end


def plan_crops(owners: list[dict], frames: int, *, mode: str = "blind",
               flagged_ids: list[int] | None = None, masking_ids: list[int] | None = None,
               speech: list[dict] | None = None) -> list[dict]:
    """Pure geometry. Exactly one blind view or at most two deep views per owner."""
    if type(frames) is not int or frames <= 0 or mode not in {"blind", "deep"}:
        raise ValueError("Invalid mode or audio frame count")
    owners = validate_owners(owners, frames / ASR_RATE)
    def ids(values):
        if values is None:
            return set()
        if (not isinstance(values, list) or any(type(v) is not int for v in values)
                or len(values) != len(set(values))):
            raise ValueError("Owner selection must contain unique integer IDs")
        return set(values)
    flagged, masking = ids(flagged_ids), ids(masking_ids)
    all_ids = {r["id"] for r in owners}
    if mode == "blind" and (flagged or masking):
        raise ValueError("Blind acquisition cannot be selected using local diagnoses")
    if mode == "deep" and (not flagged or len(flagged) > MAX_DEEP_OWNERS or not flagged <= all_ids
                            or not masking <= flagged):
        raise ValueError("Deep acquisition requires1..20valid IDs and a masking subset")
    if speech is not None:
        last = 0
        for segment in speech:
            if (set(segment) != {"start", "end"} or any(type(segment[k]) is not int for k in segment)
                    or not last <= segment["start"] < segment["end"] <= frames):
                raise ValueError("Invalid VAD interval geometry")
            last = segment["end"]
    crops = []
    for owner in owners:
        if mode == "deep" and owner["id"] not in flagged:
            continue
        variants = [("raw", 2, 2)] if mode == "blind" else (
            [("bandit_dialogue", 2, 2), ("bandit_residual", 2, 2)] if owner["id"] in masking
            else [("raw_early", 3, 1), ("raw_late", 1, 3)])
        seen = set()
        for kind, before, after in variants:
            start, end = _bounds(owner, frames, before, after,
                speech if kind.startswith("raw_") else None)
            duplicate = (kind.startswith("raw"), start, end) in seen
            seen.add((kind.startswith("raw"), start, end))
            crops.append({"view_id": f"{mode}-o{owner['id']:04d}-{kind}", "owner_id": owner["id"],
                "owner_start": owner["start"], "owner_end": owner["end"], "kind": kind,
                "crop_start_frame": start, "crop_end_frame": end, "sample_rate": ASR_RATE,
                "duplicate_geometry": duplicate and kind.startswith("raw")})
    return crops


def _convert(source: Path, destination: Path, rate: int) -> dict:
    _command(["ffmpeg", "-n", "-v", "error", "-i", str(source), "-map", "0:a:0", "-vn",
        "-map_metadata", "-1", "-ar", str(rate), "-ac", "1", "-c:a", "pcm_f32le", str(destination)])
    destination.chmod(0o444)
    return {**_pin(destination), **_audio_info(destination),
            "downmix": "ffmpeg_default_mono", "resampling": "ffmpeg_native"}


def _runtime_identity() -> dict:
    versions, files = {}, []
    for name in ("numpy", "soundfile", "torch", "transformers", "qwen-asr", "sherpa-onnx", "faster-whisper"):
        try:
            distribution = importlib.metadata.distribution(name)
            versions[name] = distribution.version
            for entry in distribution.files or ():
                relative = str(entry)
                if (relative.startswith("qwen_asr/") and relative.endswith(".py")
                        or relative in {"faster_whisper/vad.py", "faster_whisper/assets/silero_vad_v6.onnx"}
                        or relative.startswith("sherpa_onnx/") and
                        (relative.endswith("offline_recognizer.py") or relative.endswith(".so"))):
                    files.append(_pin(distribution.locate_file(entry)))
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "prefix": sys.prefix, "base_prefix": sys.base_prefix,
            "versions": versions, "files": sorted(files, key=lambda x: x["path"])}


def _model_pins(model_dir: Path, engine: str) -> dict:
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        return {"available": False, "model_dir": str(model_dir), "reason": "model_directory_missing"}
    names = ["encoder-epoch-99-avg-1.int8.onnx", "decoder-epoch-99-avg-1.int8.onnx",
             "joiner-epoch-99-avg-1.int8.onnx", "tokens.txt"] if engine == ENGINES[0] else None
    paths = [model_dir / n for n in names] if names else sorted(p for p in model_dir.rglob("*")
        if p.is_file() and ".cache" not in p.parts and p.suffix in {".json", ".safetensors", ".txt", ".model"})
    if not paths or any(not p.is_file() for p in paths):
        return {"available": False, "model_dir": str(model_dir), "reason": "model_assets_missing"}
    return {"available": True, "model_dir": str(model_dir), "files": [_pin(p) for p in paths]}


def _provenance(crop: dict, master: dict, wav: dict) -> dict:
    native = master["audio"]["sample_rate"]
    return {"view_id": crop["view_id"], "wav_path": wav["path"], "wav_sha256": wav["sha256"],
        "master_path": master["master"]["path"], "master_sha256": master["master"]["sha256"],
        "master_start_frame": math.floor(crop["crop_start_frame"] * native / ASR_RATE),
        "master_end_frame": min(master["audio"]["frames"], math.ceil(crop["crop_end_frame"] * native / ASR_RATE)),
        "master_sample_rate": native, "transform": "raw" if crop["kind"].startswith("raw") else crop["kind"],
        "owner_start": crop["owner_start"], "owner_end": crop["owner_end"],
        "context": "", "hotwords": [], "language": "Japanese", "word_alignment": False}


def _write_crop(source: Path, crop: dict, folder: Path) -> dict:
    import soundfile as sf
    import numpy as np
    begin, end = crop["crop_start_frame"], crop["crop_end_frame"]
    data, rate = sf.read(source, start=begin, stop=end, dtype="float32", always_2d=False)
    if rate != ASR_RATE or data.ndim != 1 or len(data) != end - begin or not np.isfinite(data).all():
        raise ValueError("Crop did not preserve complete finite waveform")
    path = folder / (crop["view_id"] + ".wav")
    if path.exists():
        raise FileExistsError(path)
    sf.write(path, data, rate, subtype="FLOAT")
    path.chmod(0o444)
    return _pin(path)


def _native_asr(engine: str, model_dir: Path):
    """Construct one native recognizer. No automatic retry or contextual input."""
    if engine == ENGINES[0]:
        from src.adjudicate import ZipformerN
        native = ZipformerN(model_dir, num_threads=8, decoder_precision="int8")
        def recognize(audio):
            stream = native._rec.create_stream()
            stream.accept_waveform(ASR_RATE, audio)
            native._rec.decode_stream(stream)
            return stream.result.text, {"native_text": stream.result.text, "logical_requests": 1}
        return recognize
    import torch
    from qwen_asr import Qwen3ASRModel
    native = Qwen3ASRModel.from_pretrained(str(model_dir), dtype=torch.float16, device_map="cuda",
        local_files_only=True, max_inference_batch_size=1, max_new_tokens=512)
    native.model.eval()
    def recognize(audio):
        capture = {"generate_calls": 0, "raw_outputs": []}
        generate, infer = native.model.generate, native._infer_asr
        def captured_generate(*args, **kwargs):
            capture["generate_calls"] += 1
            if capture["generate_calls"] > 1:
                raise RuntimeError("A single crop attempted multiple native generation calls")
            output = generate(*args, **kwargs)
            count = kwargs["input_ids"].shape[1]
            ids = output.sequences[0, count:].detach().cpu().tolist()
            capture["generated_token_ids"] = ids
            eos = native.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            capture["normal_stop"] = bool(ids and ids[-1] in eos and len(ids) < 512)
            return output
        def captured_infer(*args, **kwargs):
            output = infer(*args, **kwargs)
            capture["raw_outputs"] = output
            return output
        native.model.generate, native._infer_asr = captured_generate, captured_infer
        try:
            with torch.inference_mode():
                result = native.transcribe((audio, ASR_RATE), context="", language="Japanese",
                                           return_time_stamps=False)
            if len(result) != 1 or capture["generate_calls"] != 1 or not capture.get("normal_stop"):
                raise RuntimeError("Native Qwen completion was missing, truncated, or not singular")
            capture["logical_requests"] = 1
            capture["native_language"] = result[0].language
            return result[0].text, capture
        except Exception as exc:
            exc.native_observation = capture
            raise
        finally:
            native.model.generate, native._infer_asr = generate, infer
    return recognize


def _asr_worker(spec: dict) -> None:
    sys.path.insert(0, str(PROJECT))
    import soundfile as sf
    import numpy as np
    if _runtime_identity() != spec["runtime"]:
        raise ValueError("Native ASR runtime changed after preparation")
    for pin in spec["model"]["files"]:
        _check(pin)
    recognize = _native_asr(spec["engine"], Path(spec["model"]["model_dir"]))
    for item in spec["observations"]:
        path = Path(item["receipt_path"])
        _write(path.with_suffix(".started.json"), {"started_utc": _now(), "observation_id": item["observation_id"]})
        started = time.monotonic()
        receipt = {"observation": item, "model_identity_sha256": _hash(spec["model"]),
            "worker_sha256": spec["worker_sha256"], "runtime_identity_sha256": _hash(spec["runtime"]),
            "retries": 0, "started_utc": _now()}
        try:
            provenance = item["provenance"]
            wav = _check({"path": provenance["wav_path"], "sha256": provenance["wav_sha256"]})
            data, rate = sf.read(wav, dtype="float32")
            if rate != ASR_RATE or data.ndim != 1 or not np.isfinite(data).all():
                raise ValueError("Invalid ASR waveform")
            if not 0 < len(data) <= ASR_RATE * MAX_CROP_SECONDS:
                raise ValueError("ASR crop is outside the fixed duration budget")
            text, native = recognize(data)
            if not isinstance(text, str):
                raise TypeError("Native transcript is not a string")
            receipt.update(text=text, native=native, status="ok" if text.strip() else "empty")
        except Exception as exc:
            receipt.update(text="", status="unavailable", error_type=type(exc).__name__)
            if hasattr(exc, "native_observation"):
                receipt["native"] = exc.native_observation
        receipt.update(wall_seconds=time.monotonic() - started, finished_utc=_now())
        _write(path, receipt)


def _request_json(url, payload=None):
    from src.local_backend import _request_json as request
    return request(url, payload, admin=True, timeout=180)


def _backend_identity(state: dict) -> str | None:
    """Go omitempty drops loaded_model on confirmed idle Warden responses."""
    if not isinstance(state, dict):
        raise RuntimeError("Warden status is not an object")
    idle = (state.get("state") == "idle" and type(state.get("refcount")) is int
            and state["refcount"] == 0)
    if "loaded_model" not in state or state["loaded_model"] is None:
        if idle:
            return None
        raise RuntimeError("Unloaded Warden state is not confirmed idle with zero requests")
    model = state["loaded_model"]
    if not isinstance(model, str) or not model:
        raise RuntimeError("Original Warden model identity is invalid")
    return model


@contextmanager
def _gpu_lifecycle(admin_url: str, path: Path):
    """Capture actual active state; restore only after the owned worker has stopped."""
    parsed = urlsplit(admin_url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.path != "/admin" or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Expected a loopback Warden admin endpoint")
    base = admin_url.rstrip("/")
    state = _request_json(base + "/status")
    original = _backend_identity(state)
    receipt = {"started_utc": _now(), "original_loaded_model": original, "restored": False,
               "owned_workers_stopped": True}
    try:
        _request_json(base + "/unload", {})
        if _backend_identity(_request_json(base + "/status")) is not None:
            raise RuntimeError("Warden unload was not confirmed")
        yield receipt
    except BaseException as exc:
        receipt["body_error_type"] = type(exc).__name__
        raise
    finally:
        try:
            if not receipt["owned_workers_stopped"]:
                raise RuntimeError("Owned GPU worker termination is unconfirmed; restore suppressed")
            current = _backend_identity(_request_json(base + "/status"))
            if original is not None and current != original:
                response = _request_json(base + "/load", {"model": original})
                if response.get("loaded") != original:
                    raise RuntimeError("Warden did not confirm restored identity")
            receipt["restored"] = _backend_identity(_request_json(base + "/status")) == original
            if not receipt["restored"]:
                raise RuntimeError("Original Warden state was not restored")
        except BaseException as exc:
            receipt["restore_error_type"] = type(exc).__name__
            raise
        finally:
            receipt["finished_utc"] = _now()
            _write(path, receipt)


def _launch(spec: dict, folder: Path, python: str, *, timeout: float, lifecycle: dict | None = None) -> None:
    from src.local_backend import _stop_owned_process
    folder.mkdir()
    spec_path = folder / "spec.json"
    _write(spec_path, spec)
    process = None
    try:
        with (folder / "worker.log").open("x", encoding="utf-8") as log:
            env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
            process = subprocess.Popen([_interpreter_path(python), str(Path(__file__).resolve()), "--worker", str(spec_path)],
                cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError("Native worker failed; see private worker log")
    finally:
        if process is not None:
            try:
                _stop_owned_process(process)
                if process.poll() is None:
                    raise RuntimeError("Worker remains alive")
            except BaseException:
                if lifecycle is not None:
                    lifecycle["owned_workers_stopped"] = False
                raise


def _observation(crop: dict, engine: str, provenance: dict, folder: Path) -> dict:
    oid = crop["view_id"] + ("-zipformer" if engine == ENGINES[0] else "-qwen")
    return {"observation_id": oid, "owner_id": crop["owner_id"], "crop_start_frame": crop["crop_start_frame"],
        "crop_end_frame": crop["crop_end_frame"], "sample_rate": ASR_RATE, "engine": engine,
        "provenance": provenance, "receipt_path": str(folder / "receipts" / (oid + ".json"))}


def _unavailable(item: dict, reason: str) -> None:
    path = Path(item["receipt_path"])
    if not path.exists():
        _write(path, {"observation": item, "text": "", "status": "unavailable", "reason": reason,
            "retries": 0, "finished_utc": _now()})


def _validate_bandit(manifest: dict) -> None:
    if (manifest.get("version") != 1 or manifest.get("sample_rate") != 48000
            or manifest.get("channels") != 1 or manifest.get("chunk_seconds") != 8.0
            or manifest.get("hop_seconds") != 1.0 or manifest.get("batch_size") != 1):
        raise ValueError("BandIt manifest conflicts with the fixed native audio contract")
    _check({"path": manifest["checkpoint_path"], "sha256": manifest["checkpoint_sha256"]})
    for path, digest in manifest["source_code_sha256"].items():
        _check({"path": path, "sha256": digest})
    if not manifest["source_code_sha256"]:
        raise ValueError("BandIt source code must be pinned")


def _bandit_worker(spec: dict) -> None:
    """Import the separate upstream src package only in this dedicated process."""
    import numpy as np
    import soundfile as sf
    import torch
    manifest = spec["manifest"]
    _validate_bandit(manifest)
    # This script is executed by path; no production src package has been imported.
    sys.path.insert(0, manifest["source_dir"])
    def resolve(name):
        module, attribute = name.split(":")
        return getattr(importlib.import_module(module), attribute)
    Bandit = resolve(manifest["native_model_import"])
    Handler = resolve(manifest["native_handler_import"])
    model = Bandit(**manifest["model_kwargs"])
    checkpoint = torch.load(manifest["checkpoint_path"], map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    native_state = {key[len("model."):]: value for key, value in state.items() if key.startswith("model.")}
    ignored = {key: list(value.shape) if hasattr(value, "shape") else type(value).__name__
               for key, value in state.items() if not key.startswith("model.")}
    if not native_state:
        raise ValueError("Checkpoint lacks native model state")
    model.load_state_dict(native_state, strict=True)
    model.eval().cuda()
    handler = Handler(**manifest.get("handler_kwargs", {"chunk_size_seconds": 8.0,
        "hop_size_seconds": 1.0, "inference_batch_size": 1, "fs": 48000}))
    _write(Path(spec["output_dir"]) / "native-load.json", {"nonmodel_state_keys": ignored,
        "checkpoint_sha256": manifest["checkpoint_sha256"], "strict_model_load": True})
    for crop in spec["crops"]:
        original, rate = sf.read(_check(crop["input"]), dtype="float32")
        if rate != 48000 or original.ndim != 1 or not np.isfinite(original).all():
            raise ValueError("BandIt requires finite native48k mono PCM")
        with torch.inference_mode():
            result = handler(torch.from_numpy(original)[None, None, :].cuda(), model)
        directory = Path(crop["output_dir"])
        directory.mkdir()
        stems = {}
        for name, item in result["estimates"].items():
            if not isinstance(name, str) or not name.replace("_", "").isalnum():
                raise ValueError("Unsafe native stem name")
            data = item["audio"].detach().cpu().numpy().reshape(-1)
            if len(data) != len(original) or not np.isfinite(data).all():
                raise ValueError("Separated stem changed frame count or contains nonfinite data")
            path = directory / (name + ".wav")
            sf.write(path, data, 48000, subtype="FLOAT")
            path.chmod(0o444)
            stems[name] = _pin(path)
        dialogue_key = manifest.get("dialogue_stem", "dialogue")
        if dialogue_key not in stems:
            raise ValueError("Native output lacks the configured dialogue stem")
        dialogue, _ = sf.read(stems[dialogue_key]["path"], dtype="float32")
        residual = directory / "original-minus-dialogue.wav"
        sf.write(residual, original - dialogue, 48000, subtype="FLOAT")
        residual.chmod(0o444)
        _write(directory / "stems.json", {"input": crop["input"], "stems": stems,
            "dialogue": stems[dialogue_key], "residual": _pin(residual), "sample_rate": 48000,
            "frames": len(original), "residual_definition": "original_mono_minus_estimated_dialogue",
            "estimated_not_ground_truth": True})


def acquire(media: Path | str, owners: list[dict], out_dir: Path | str, mode: str = "blind", *,
            flagged_ids: list[int] | None = None, masking_ids: list[int] | None = None,
            audio_stream: int = 0, bandit_manifest: Path | str | None = None,
            python_executable: Path | str | None = None, admin_url: str = "http://127.0.0.1:8089/admin",
            qwen_model_dir: Path | str | None = None, zipformer_model_dir: Path | str | None = None,
            execute: bool = False, worker_timeout: float = 1800) -> dict:
    """Prepare or execute an immutable pool; execution after preparation is allowed once."""
    if type(execute) is not bool or _number(worker_timeout, "worker_timeout") <= 0:
        raise ValueError("Invalid execution controls")
    folder = Path(out_dir).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    master = preserve_master(media, folder / "master", audio_stream=audio_stream)
    owners = validate_owners(owners, master["audio"]["seconds"])
    paths = {ENGINES[0]: str(Path(zipformer_model_dir or DEFAULT_ASR_ROOT /
        "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01").expanduser().resolve()),
        ENGINES[1]: str(Path(qwen_model_dir or DEFAULT_ASR_ROOT / "Qwen3-ASR-1.7B").expanduser().resolve())}
    request = {"media": master["media"], "owners": owners, "mode": mode, "flagged_ids": flagged_ids,
        "masking_ids": masking_ids, "audio_stream": audio_stream, "models": paths,
        "bandit_manifest": _pin(bandit_manifest) if bandit_manifest is not None else None,
        "python_executable": _interpreter_path(python_executable or sys.executable),
        "admin_url": admin_url, "worker_timeout": worker_timeout}
    preparation_path = folder / "preparation.json"
    if preparation_path.exists():
        preparation = _read(preparation_path)
        if request != preparation["request"]:
            raise ValueError("Prepared acquisition inputs or policy changed")
        for pin in preparation["code_pins"]:
            _check(pin)
        _check(preparation["mono"])
    else:
        mono = _convert(Path(master["master"]["path"]), folder / "original-16k-mono.wav", ASR_RATE)
        crops = plan_crops(owners, mono["frames"], mode=mode, flagged_ids=flagged_ids, masking_ids=masking_ids)
        preparation = {"version": VERSION, "request": request, "master": master, "mono": mono,
            "crops": crops, "models": {engine: _model_pins(Path(path), engine) for engine, path in paths.items()},
            "code_pins": [_pin(__file__), _pin(PROJECT / "src/adjudicate.py"), _pin(PROJECT / "src/local_backend.py")],
            "max_asr_requests": len(crops) * 2, "runtime": _runtime_identity(),
            "vad_options": VAD_OPTIONS, "created_utc": _now()}
        _write(preparation_path, preparation)
    if not execute:
        return {"version": VERSION, "status": "prepared", "preparation": _pin(preparation_path),
            "media_frames": master["audio"]["frames"], "sample_rate": master["audio"]["sample_rate"],
            "maximum_asr_requests": preparation["max_asr_requests"], "observations": []}
    if _runtime_identity() != preparation["runtime"]:
        raise ValueError("Runtime changed after acoustic preparation")
    evidence_path = folder / "evidence.json"
    if evidence_path.exists():
        return validate_evidence(evidence_path)
    if (folder / "execution-started.json").exists():
        raise RuntimeError("Started acquisition cannot be rerun or silently repaired")
    _write(folder / "execution-started.json", {"preparation": _pin(preparation_path), "started_utc": _now()})
    return _execute(preparation, preparation_path, folder)


def _execute(preparation: dict, preparation_path: Path, folder: Path) -> dict:
    import soundfile as sf
    request, master = preparation["request"], preparation["master"]
    mono = preparation["mono"]
    crops = preparation["crops"]
    vad = {"status": "not_requested"}
    if request["mode"] == "deep" and any(c["kind"].startswith("raw_") for c in crops):
        try:
            from faster_whisper.vad import VadOptions, get_speech_timestamps
            audio, _ = sf.read(_check(mono), dtype="float32")
            speech = get_speech_timestamps(audio, VadOptions(**VAD_OPTIONS), sampling_rate=ASR_RATE)
            crops = plan_crops(request["owners"], mono["frames"], mode="deep",
                flagged_ids=request["flagged_ids"], masking_ids=request["masking_ids"], speech=speech)
            vad = {"status": "ok", "speech_intervals": speech, "options": VAD_OPTIONS,
                   "intervals_are_estimates": True, "frames_dropped": 0}
        except Exception as exc:
            vad = {"status": "unavailable", "error_type": type(exc).__name__, "fallback": "fixed_shifted_raw_crops"}
    views = folder / "views"
    views.mkdir()
    items, bandit_jobs, bandit_inputs = [], [], {}
    full48 = None
    if any(c["kind"].startswith("bandit") for c in crops):
        full48 = _convert(Path(master["master"]["path"]), folder / "original-48k-mono.wav", 48000)
    for crop in crops:
        if crop["kind"].startswith("raw"):
            wav = _write_crop(Path(mono["path"]), crop, views)
        else:
            # Prepare original input once per owner; outputs remain estimated views.
            key = crop["owner_id"]
            if key not in bandit_inputs:
                raw_crop = {**crop, "view_id": f"deep-o{key:04d}-bandit-input"}
                original = _write_crop(Path(mono["path"]), raw_crop, views)
                # Resample from native master directly, not the already16k ASR view.
                begin = crop["crop_start_frame"] * 3
                end = min(crop["crop_end_frame"] * 3, full48["frames"])
                data, rate = sf.read(full48["path"], start=begin, stop=end, dtype="float32")
                native_path = views / f"deep-o{key:04d}-48k.wav"
                sf.write(native_path, data, rate, subtype="FLOAT")
                native_path.chmod(0o444)
                converted = {**_pin(native_path), **_audio_info(native_path)}
                destination = folder / "stems" / f"owner-{key:04d}"
                bandit_inputs[key] = {"input": converted, "output_dir": str(destination),
                    "original_16k": original, "full48": full48,
                    "full48_start_frame": begin, "full48_end_frame": end}
                bandit_jobs.append(bandit_inputs[key])
            wav = {"path": str(views / (crop["view_id"] + ".wav")), "sha256": None}
        provenance = _provenance(crop, master, wav)
        for engine in ENGINES:
            item = _observation(crop, engine, provenance, folder)
            item["duplicate_geometry"] = crop["duplicate_geometry"]
            items.append(item)
    bandit_error = None
    if bandit_jobs:
        try:
            if request["bandit_manifest"] is None:
                raise FileNotFoundError("BandIt is not configured")
            manifest = _read(_check(request["bandit_manifest"]))
            _validate_bandit(manifest)
            (folder / "stems").mkdir()
            spec = {"worker": "bandit", "worker_sha256": file_hash(__file__), "manifest": manifest,
                    "crops": bandit_jobs, "output_dir": str(folder / "bandit-worker")}
            with _gpu_lifecycle(request["admin_url"], folder / "bandit-lifecycle.json") as life:
                _launch(spec, folder / "bandit-worker", manifest["python_executable"],
                        timeout=request["worker_timeout"], lifecycle=life)
            for crop in crops:
                if crop["kind"].startswith("bandit"):
                    stems = _read(Path(bandit_inputs[crop["owner_id"]]["output_dir"]) / "stems.json")
                    source = _check(stems["dialogue" if crop["kind"] == "bandit_dialogue" else "residual"])
                    converted = _convert(source, views / (crop["view_id"] + ".wav"), ASR_RATE)
                    expected = crop["crop_end_frame"] - crop["crop_start_frame"]
                    if converted["frames"] != expected:
                        raise ValueError("BandIt resampling changed crop geometry")
                    for item in items:
                        if item["provenance"]["view_id"] == crop["view_id"]:
                            item["provenance"].update(wav_path=converted["path"], wav_sha256=converted["sha256"],
                                output_frames=converted["frames"], native_stems_receipt=_pin(source.parent / "stems.json"))
                            item["provenance"]["separator_input"] = bandit_inputs[crop["owner_id"]]
        except Exception as exc:
            bandit_error = type(exc).__name__
            lifecycle_path = folder / "bandit-lifecycle.json"
            if lifecycle_path.exists() and _read(lifecycle_path).get("restored") is not True:
                raise
    finalized = {"version": VERSION, "preparation": _pin(preparation_path), "crops": crops,
        "vad": vad, "observations": items, "bandit_error_type": bandit_error}
    plan_path = folder / "acquisition-plan.json"
    _write(plan_path, finalized)
    for engine in ENGINES:
        selected = []
        for item in items:
            if item["engine"] != engine:
                continue
            reason = ("duplicate_crop_geometry" if item["duplicate_geometry"] else
                "separator_unavailable" if item["provenance"]["wav_sha256"] is None else
                "model_unavailable" if not preparation["models"][engine]["available"] else None)
            if reason:
                _unavailable(item, reason)
            else:
                selected.append(item)
        if not selected:
            continue
        spec = {"worker": "asr", "worker_sha256": file_hash(__file__), "engine": engine,
            "model": preparation["models"][engine], "runtime": preparation["runtime"], "observations": selected}
        try:
            if engine == ENGINES[1]:
                with _gpu_lifecycle(request["admin_url"], folder / "qwen-lifecycle.json") as life:
                    _launch(spec, folder / "qwen-worker", request["python_executable"],
                            timeout=request["worker_timeout"], lifecycle=life)
            else:
                _launch(spec, folder / "zipformer-worker", request["python_executable"], timeout=request["worker_timeout"])
        except Exception as exc:
            for item in selected:
                _unavailable(item, "worker_failed_" + type(exc).__name__)
            lifecycle_path = folder / "qwen-lifecycle.json"
            if engine == ENGINES[1] and lifecycle_path.exists() and _read(lifecycle_path).get("restored") is not True:
                raise
    output = []
    for item in items:
        receipt = _read(item["receipt_path"])
        output.append({**item, "text": receipt["text"], "status": receipt["status"],
                       "receipt_sha256": file_hash(item["receipt_path"])})
    plan_sha = file_hash(plan_path)
    receipts = {item["observation_id"]: item["receipt_sha256"] for item in output}
    result = {"version": VERSION, "mode": request["mode"], "pool_version": _hash({"plan_sha256": plan_sha, "receipts": receipts}),
        "media_frames": master["audio"]["frames"], "sample_rate": master["audio"]["sample_rate"],
        "media_sha256": master["media"]["sha256"], "plan_path": str(plan_path), "plan_sha256": plan_sha,
        "observations": output, "complete": all(x["status"] in {"ok", "empty"} for x in output),
        "counts": {status: sum(x["status"] == status for x in output) for status in ("ok", "empty", "unavailable")},
        "maximum_asr_requests": preparation["max_asr_requests"], "finished_utc": _now(),
        "lifecycle": [_read(p) for p in sorted(folder.glob("*-lifecycle.json"))]}
    _write(folder / "evidence.json", result)
    return validate_evidence(result)


def validate_evidence(path_or_dict: Path | str | dict) -> dict:
    """Validate immutable native receipts; do not infer accuracy from agreement."""
    evidence = _read(path_or_dict) if not isinstance(path_or_dict, dict) else path_or_dict
    if evidence.get("version") != VERSION:
        raise ValueError("Unknown late evidence format")
    plan = _read(_check({"path": evidence["plan_path"], "sha256": evidence["plan_sha256"]}))
    preparation = _read(_check(plan["preparation"]))
    master = preparation["master"]
    _check(master["media"])
    _check(master["master"])
    if (evidence["media_frames"] != master["audio"]["frames"] or evidence["sample_rate"] != master["audio"]["sample_rate"]
            or evidence["media_sha256"] != master["media"]["sha256"] or evidence["mode"] != preparation["request"]["mode"]):
        raise ValueError("Evidence media geometry or mode differs from preparation")
    expected = {x["observation_id"]: x for x in plan["observations"]}
    actual = evidence["observations"]
    if len(actual) != len(expected) or {x["observation_id"] for x in actual} != set(expected):
        raise ValueError("Evidence pool coverage differs from its frozen plan")
    checked, receipts = set(), {}
    for item in actual:
        baseline = expected[item["observation_id"]]
        if {k: item[k] for k in baseline} != baseline:
            raise ValueError("Evidence observation metadata changed")
        receipt = _read(_check({"path": item["receipt_path"], "sha256": item["receipt_sha256"]}))
        if (receipt["observation"] != baseline or receipt["text"] != item["text"]
                or receipt["status"] != item["status"] or receipt.get("retries") != 0):
            raise ValueError("Observation does not match its immutable native receipt")
        if item["status"] not in {"ok", "empty", "unavailable"} or not isinstance(item["text"], str):
            raise ValueError("Invalid observation status/text")
        if item["status"] == "unavailable" and item["text"] != "":
            raise ValueError("Unavailable recognition cannot supply text")
        if item["status"] in {"ok", "empty"} and ((item["status"] == "empty") != (not item["text"].strip())):
            raise ValueError("Native status and exact text disagree")
        provenance = item["provenance"]
        if provenance["context"] != "" or provenance["hotwords"] != [] or provenance["word_alignment"] is not False:
            raise ValueError("Acoustic observation was not context blind")
        if item["status"] in {"ok", "empty"}:
            if receipt.get("model_identity_sha256") != _hash(preparation["models"][item["engine"]]):
                raise ValueError("Native model identity differs from preparation")
            if receipt.get("worker_sha256") != preparation["code_pins"][0]["sha256"]:
                raise ValueError("Native worker identity differs from preparation")
            if receipt.get("runtime_identity_sha256") != _hash(preparation["runtime"]):
                raise ValueError("Native runtime identity differs from preparation")
            if receipt.get("native", {}).get("logical_requests") != 1:
                raise ValueError("Observation lacks its single native request receipt")
            key = (provenance["wav_path"], provenance["wav_sha256"])
            if key not in checked:
                _check({"path": key[0], "sha256": key[1]})
                checked.add(key)
            if item["engine"] == ENGINES[1] and receipt.get("native", {}).get("normal_stop") is not True:
                raise ValueError("Qwen observation lacks a normal native completion")
        receipts[item["observation_id"]] = item["receipt_sha256"]
    if evidence["pool_version"] != _hash({"plan_sha256": evidence["plan_sha256"], "receipts": receipts}):
        raise ValueError("Evidence pool version is not its exact receipt identity")
    counts = {status: sum(x["status"] == status for x in actual) for status in ("ok", "empty", "unavailable")}
    if evidence["counts"] != counts or evidence["complete"] is not all(x["status"] in {"ok", "empty"} for x in actual):
        raise ValueError("Evidence summary differs from its observations")
    if len(actual) > preparation["max_asr_requests"] or (evidence["mode"] == "deep" and len(actual) > 80):
        raise ValueError("Evidence exceeds its acquisition budget")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--worker", type=Path)
    actions.add_argument("--preflight-asr-imports", action="store_true")
    args = parser.parse_args()
    if args.preflight_asr_imports:
        print(json.dumps(_asr_import_preflight()))
        return 0
    spec = _read(args.worker)
    if spec["worker_sha256"] != file_hash(__file__):
        raise ValueError("Worker code changed after preparation")
    if spec["worker"] == "bandit":
        _bandit_worker(spec)
    elif spec["worker"] == "asr":
        _asr_worker(spec)
    else:
        raise ValueError("Unknown worker type")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
