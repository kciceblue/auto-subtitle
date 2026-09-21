"""Temporarily serve one local GGUF writer and restore the configured model."""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from typing import Any, Iterator
import urllib.request
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)
LOCAL_PORT = 18101
READY_TIMEOUT = 180.0
RESTORE_TIMEOUT = 360.0
_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")


class LocalBackendError(RuntimeError):
    """The temporary writer could not be started or safely restored."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()



def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=32)
def _verified_model_hash(path: Path, identity: tuple[int, int, int, int, int]) -> str:
    """Reuse SHA256 only for the same unchanged file within this process."""
    digest = _sha256(path)
    if _file_identity(path) != identity:
        raise LocalBackendError("Local writer weights changed during verification")
    return digest


def _weight_bundle(model: Path, expected_sha256: str | None,
                   expected_shards: dict[str, str] | None) -> tuple[list[dict], dict[Path, tuple]]:
    match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", model.name)
    if match:
        if int(match[2]) != 1 or not 2 <= int(match[3]) <= 256:
            raise ValueError("Use the first shard of a complete local GGUF bundle")
        members = [model.with_name(f"{match[1]}-{index:05d}-of-{int(match[3]):05d}.gguf")
                   for index in range(1, int(match[3]) + 1)]
        if not isinstance(expected_shards, dict) or set(expected_shards) != {p.name for p in members}:
            raise ValueError("Pin every local GGUF shard by filename and SHA256")
    else:
        members = [model]
        if expected_shards:
            raise ValueError("Shard pins require a split GGUF model")
    manifest, stats = [], {}
    for index, path in enumerate(members, 1):
        logger.info("Verifying local model weights %d/%d: %s", index, len(members), path.name)
        path = path.resolve(strict=True)
        if not path.is_file():
            raise ValueError("Each model shard must be a file")
        pin = expected_shards[path.name] if match else expected_sha256
        if pin is not None and (not isinstance(pin, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", pin)):
            raise ValueError("Invalid model shard SHA256")
        identity = _file_identity(path)
        stats[path] = identity
        digest = _verified_model_hash(path, identity)
        if pin is not None and digest != pin.lower():
            raise ValueError("Local writer model SHA256 mismatch")
        manifest.append({"path": str(path), "sha256": digest, "bytes": identity[2]})
    if expected_sha256 is not None and manifest[0]["sha256"] != expected_sha256.lower():
        raise ValueError("First shard SHA256 differs from model pin")
    return manifest, stats


def _projector_file(projector_path: Path | str | None,
                    projector_sha256: str | None) -> tuple[dict | None, dict[Path, tuple]]:
    if (projector_path is None) != (projector_sha256 is None):
        raise ValueError("projector_path and projector_sha256 must be provided together")
    if projector_path is None:
        return None, {}
    if (not isinstance(projector_sha256, str)
            or not re.fullmatch(r"[A-Fa-f0-9]{64}", projector_sha256)):
        raise ValueError("projector_sha256 must contain exactly 64 hexadecimal characters")
    projector = Path(projector_path).expanduser().resolve(strict=True)
    if not projector.is_file() or projector.suffix.lower() != ".gguf":
        raise ValueError("Local audio projector requires a GGUF file")
    previous = _file_identity(projector)
    digest = _verified_model_hash(projector, previous)
    if digest != projector_sha256.lower():
        raise ValueError("Local audio projector SHA256 mismatch")
    return {"path": str(projector), "sha256": digest, "bytes": previous[2]}, {projector: previous}


def _chat_template_file(chat_template_path: Path | str | None,
                        chat_template_sha256: str | None) -> tuple[dict | None, dict[Path, tuple]]:
    if (chat_template_path is None) != (chat_template_sha256 is None):
        raise ValueError("chat_template_path and chat_template_sha256 must be provided together")
    if chat_template_path is None:
        return None, {}
    if (not isinstance(chat_template_sha256, str)
            or not re.fullmatch(r"[A-Fa-f0-9]{64}", chat_template_sha256)):
        raise ValueError("chat_template_sha256 must contain exactly 64 hexadecimal characters")
    template = Path(chat_template_path).expanduser().resolve(strict=True)
    if not template.is_file():
        raise ValueError("Local chat template requires a file")
    previous = _file_identity(template)
    digest = _verified_model_hash(template, previous)
    if digest != chat_template_sha256.lower():
        raise ValueError("Local chat template SHA256 mismatch")
    return {"path": str(template), "sha256": digest, "bytes": previous[2]}, {template: previous}


def _request_json(url: str, payload: dict | None = None, *, admin: bool = False,
                  timeout: float = 10.0) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("WARDEN_AUTH_TOKEN") if admin else None
    if token:
        headers["Authorization"] = "Bearer " + token
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise LocalBackendError("Expected an object from local model service")
    return result


def _reserve_port() -> socket.socket:
    reserved = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        reserved.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        reserved.bind(("127.0.0.1", LOCAL_PORT))
        reserved.listen(1)
    except OSError as exc:
        reserved.close()
        raise LocalBackendError(f"Local writer port {LOCAL_PORT} is occupied or unavailable") from exc
    return reserved


def _stop_owned_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        if process.poll() is None:
            raise
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _ready(process: subprocess.Popen, model_path: Path, alias: str) -> dict[str, Any]:
    deadline = time.monotonic() + READY_TIMEOUT
    while True:
        if process.poll() is not None:
            raise LocalBackendError(f"Local writer exited during startup: {process.returncode}")
        try:
            props = _request_json(f"http://127.0.0.1:{LOCAL_PORT}/props", timeout=2)
        except (OSError, ValueError):
            props = None
        if props is not None:
            observed = props.get("model_path")
            if not isinstance(observed, str) or Path(observed).resolve() != model_path:
                raise LocalBackendError("Local writer readiness returned a different model path")
            observed_alias = props.get("model_alias")
            if observed_alias is not None and observed_alias != alias and observed_alias != [alias]:
                raise LocalBackendError("Local writer readiness returned a different model alias")
            if process.poll() is not None:
                raise LocalBackendError("Owned local writer exited while readiness was checked")
            return props
        if time.monotonic() >= deadline:
            raise LocalBackendError("Local writer readiness timed out")
        time.sleep(0.25)


@contextmanager
def temporary_local_writer(
    model_path: Path | str,
    server_binary: Path | str,
    *,
    alias: str,
    expected_sha256: str | None = None,
    expected_shards: dict[str, str] | None = None,
    projector_path: Path | str | None = None,
    projector_sha256: str | None = None,
    chat_template_path: Path | str | None = None,
    chat_template_sha256: str | None = None,
    context_size: int = 16384,
    full_swa: bool = False,
    cpu_moe_layers: int = 0,
    gpu_layers: int = 99,
    threads: int = 4,
    admin_url: str = "http://127.0.0.1:8089/admin",
    restore_model: str = "qwen3.8-27b-dflash",
    restore_previous: bool = False,
    log_path: Path | str,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(chat_endpoint, identity)``; send no inference requests itself.

    All preflight checks precede Warden unloading. Cleanup touches only this
    context's child process. If cleanup fails during a caller exception, the
    original exception survives with a note describing the cleanup failure.
    ``restore_model`` is the caller's configured profile, never a registry edit.
    Audio input is opt-in: pin ``projector_path`` and ``projector_sha256`` together.
    A template override likewise requires a paired path/SHA256 and server identity match.
    """
    if not isinstance(alias, str) or not _ALIAS.fullmatch(alias):
        raise ValueError("Invalid local writer alias")
    if not isinstance(restore_model, str) or not _ALIAS.fullmatch(restore_model):
        raise ValueError("Invalid restore model alias")
    if isinstance(context_size, bool) or not isinstance(context_size, int) or context_size < 512:
        raise ValueError("Local writer context_size must be an integer of at least 512")
    if type(gpu_layers) is not int or not 0 <= gpu_layers <= 512:
        raise ValueError("gpu_layers must be an integer from 0 through 512")
    if type(cpu_moe_layers) is not int or not 0 <= cpu_moe_layers <= 512:
        raise ValueError("cpu_moe_layers must be an integer from 0 through 512")
    if type(threads) is not int or not 1 <= threads <= 512:
        raise ValueError("threads must be an integer from 1 through 512")
    if type(full_swa) is not bool:
        raise ValueError("full_swa must be a boolean")
    parsed = urlsplit(admin_url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Warden admin_url must be a plain loopback HTTP URL")
    if type(restore_previous) is not bool:
        raise ValueError("restore_previous must be a boolean")
    captured_model = restore_model
    if restore_previous:
        original = _request_json(admin_url.rstrip("/") + "/status", admin=True, timeout=10)
        captured_model = original.get("loaded_model")
        if captured_model is None and original.get("state") != "idle":
            raise LocalBackendError("Cannot capture an unstable Warden state")
        if captured_model is not None and (not isinstance(captured_model, str) or not _ALIAS.fullmatch(captured_model)):
            raise LocalBackendError("Invalid active Warden identity")
    model = Path(model_path).expanduser().resolve(strict=True)
    binary = Path(server_binary).expanduser().resolve(strict=True)
    if not model.is_file() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("Local writer requires a model file and executable server binary")
    log = Path(log_path).expanduser().resolve()
    if log in {model, binary}:
        raise ValueError("Local writer log must not overwrite model or server files")
    if expected_sha256 is not None and not re.fullmatch(r"[A-Fa-f0-9]{64}", expected_sha256):
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
    manifest, model_stats = _weight_bundle(model, expected_sha256, expected_shards)
    projector, projector_stats = _projector_file(projector_path, projector_sha256)
    model_stats.update(projector_stats)
    template, template_stats = _chat_template_file(chat_template_path, chat_template_sha256)
    model_stats.update(template_stats)
    if str(log) in {row["path"] for row in manifest}:
        raise ValueError("Local writer log must not overwrite any model shard")
    protected_files = {binary, *model_stats}
    if log in protected_files or (log.exists() and any(log.samefile(path) for path in protected_files)):
        raise ValueError("Local writer log must not overwrite model, projector, template, or server files")
    digest = manifest[0]["sha256"]
    binary_digest = _sha256(binary)
    reserved = _reserve_port()
    process = None
    unload_attempted = False
    primary_error = None
    command = [str(binary), "--model", str(model), "--host", "127.0.0.1",
               "--port", str(LOCAL_PORT), "--alias", alias, "--ctx-size", str(context_size),
               "--parallel", "1", "--n-gpu-layers", str(gpu_layers), "--threads", str(threads),
               "--batch-size", "512", "--ubatch-size", "128", "--flash-attn", "on",
               "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--fit", "off", "--jinja"]
    if full_swa:
        command.append("--swa-full")
    if cpu_moe_layers:
        command += ["--n-cpu-moe", str(cpu_moe_layers)]
    if projector is not None:
        command += ["--mmproj", projector["path"]]
    if template is not None:
        command += ["--chat-template-file", template["path"]]
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as output:
            unload_attempted = True
            _request_json(admin_url.rstrip("/") + "/unload", {}, admin=True, timeout=45)
            reserved.close()
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            props = _ready(process, model, alias)
            if template is not None:
                actual_template = props.get("chat_template")
                if (not isinstance(actual_template, str)
                        or hashlib.sha256(actual_template.encode("utf-8")).hexdigest() != template["sha256"]):
                    raise LocalBackendError("Local writer readiness returned a different chat template")
            for path, previous in model_stats.items():
                if _file_identity(path) != previous:
                    raise LocalBackendError("Local writer weights changed during model startup")
            identity = {"model_path": str(model), "model_alias": alias,
                        "model_sha256": digest, "model_files": manifest,
                        "model_bundle_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                        "gpu_layers": gpu_layers, "cpu_moe_layers": cpu_moe_layers, "threads": threads, "server_binary": str(binary),
                        "server_binary_sha256": binary_digest, "pid": process.pid,
                        "context_size": context_size, "full_swa": full_swa, "command": command,
                        "build_info": props.get("build_info"), "model_ftype": props.get("model_ftype"),
                        "reported_model_alias": props.get("model_alias"),
                        "chat_template_sha256": hashlib.sha256(str(props.get("chat_template", "")).encode()).hexdigest(),
                        "default_generation_settings": props.get("default_generation_settings")}
            if projector is not None:
                identity.update(projector_path=projector["path"], projector_sha256=projector["sha256"],
                                projector_file=projector)
            if template is not None:
                identity.update(chat_template_path=template["path"],
                                chat_template_file_sha256=template["sha256"], chat_template_file=template)
            yield f"http://127.0.0.1:{LOCAL_PORT}/v1/chat/completions", identity
            for path, previous in model_stats.items():
                if _file_identity(path) != previous:
                    raise LocalBackendError("Local writer weights changed during inference")
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        reserved.close()
        cleanup_errors = []
        stopped = True
        if process is not None:
            try:
                _stop_owned_process(process)
            except BaseException as exc:
                stopped = False
                cleanup_errors.append(f"Owned local writer shutdown failed: {exc}")
        if unload_attempted and stopped:
            try:
                if captured_model is None:
                    _request_json(admin_url.rstrip("/") + "/unload", {}, admin=True, timeout=45)
                    restored = _request_json(admin_url.rstrip("/") + "/status", admin=True, timeout=10)
                    if restored.get("loaded_model") is not None or restored.get("state") != "idle":
                        raise LocalBackendError("Warden did not restore its original unloaded state")
                else:
                    restored = _request_json(admin_url.rstrip("/") + "/load", {"model": captured_model},
                                             admin=True, timeout=RESTORE_TIMEOUT)
                    if restored.get("loaded") != captured_model:
                        raise LocalBackendError("Warden did not confirm the requested restore model")
            except BaseException as exc:
                cleanup_errors.append(f"Restore model {captured_model} failed: {exc}")
        elif unload_attempted:
            cleanup_errors.append("Restore skipped because owned local writer shutdown was not confirmed")
        if cleanup_errors:
            message = "; ".join(cleanup_errors)
            logger.error(message)
            if primary_error is not None:
                primary_error.add_note(message)
            else:
                raise LocalBackendError(message)
