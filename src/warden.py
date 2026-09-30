"""GPU headroom coordination with model-warden: prevent VRAM OOM.

The Warden-resident 27B LLM (qwen3.8-27b-dflash, ~28 GB) fills most of the
32 GB RTX 5090, so a CUDA model that loads in-process must first make room.
In the evidence-first workflow this is the Qwen3-ForcedAligner worker of the
align stage (src/aligned_display.py); the ASR workers and the Gemma draft use
the stricter unload/restore lifecycle in src/late_audio.py and
src/local_backend.py instead.

model-warden is a lazy-loading gateway: it keeps the GPU empty until a chat
request arrives (state "idle", refcount 0), loads on demand, and evicts the
model on `POST /admin/unload` (200 = GPU clear, 409 = requests in flight).
``ensure_gpu_headroom`` evicts the Warden LLM only when free VRAM is short;
Warden reloads it lazily on the next chat request.
"""

from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)

DEFAULT_ADMIN_URL = "http://127.0.0.1:8089/admin"

# Default thresholds, sized for a ~4 GB speech model (measured 2026-08-30).
# The eviction threshold is deliberately conservative and leaves room for the
# CUDA context and the desktop compositor.
ASR_MIN_FREE_GB = 8.0
# Below this much free VRAM after eviction, loading will very likely OOM —
# fail with a clear message instead of crashing mid-run.
ASR_HARD_FLOOR_GB = 6.0
# qwen3-asr-1.7B (float16) needs ~3.7 GB of weights+buffers on this rig
# (measured 2026-08-30 via audio arbitration). Same conservative policy as
# ASR: evict the warden LLM before loading, and fail loudly if even after
# eviction there is no room.
ADJUDICATE_Q_MIN_FREE_GB = 6.0
ADJUDICATE_Q_HARD_FLOOR_GB = 4.5


def unload_warden(admin_url: str = DEFAULT_ADMIN_URL, timeout: int = 10) -> bool:
    """Evict the warden-resident LLM from the GPU.

    Returns True when the GPU is clear (200 — the response body says whether
    this call did the eviction or the GPU was already idle). Returns False
    on 409 (requests in flight / load in progress — do not hammer), on
    transport errors, and on unexpected status codes.
    """
    try:
        resp = requests.post(f"{admin_url}/unload", timeout=timeout)
    except requests.RequestException as e:
        logger.warning("warden /admin/unload unreachable (%s) — continuing", e)
        return False
    if resp.status_code == 200:
        logger.info("warden LLM evicted from GPU (or already idle)")
        return True
    if resp.status_code == 409:
        logger.warning(
            "warden busy (requests in flight), LLM stays resident — the next model "
            "may not fit. body: %s",
            resp.text[:120],
        )
        return False
    logger.warning(
        "warden /admin/unload returned %d — %s",
        resp.status_code, resp.text[:120],
    )
    return False


def ensure_gpu_headroom(
    required_gb: float = ASR_MIN_FREE_GB,
    admin_url: str = DEFAULT_ADMIN_URL,
    hard_floor_gb: float = ASR_HARD_FLOOR_GB,
    enabled: bool = True,
    caller: str = "ASR",
) -> float:
    """Before loading a CUDA model: make sure at least ``required_gb`` is free.

    If the GPU is too full, evict the warden LLM and re-check (polling for
    up to a few seconds, since an eviction may finish asynchronously).

    Shared by every GPU consumer that cannot coexist with the warden-resident
    27B model: call it right before loading, with ``caller`` naming the
    consumer so the error message stays accurate.

    Returns the free VRAM in bytes after the check. Raises RuntimeError when
    even after eviction the free memory sits below ``hard_floor_gb`` —
    loading there would OOM, and failing loudly beats crashing hours into a
    job. Returns infinity when CUDA is unavailable.
    """
    import torch

    if not torch.cuda.is_available():
        return float("inf")

    free = torch.cuda.mem_get_info()[0]
    free_gb = free / 1e9
    if free_gb >= required_gb:
        logger.info("GPU headroom OK for %s: %.1f GB free (need >= %.1f)",
                    caller, free_gb, required_gb)
        return free

    if not enabled:
        logger.warning(
            "GPU headroom LOW for %s (%.1f GB free < %.1f GB) and warden "
            "eviction disabled — this model may OOM",
            caller, free_gb, required_gb,
        )
        return free

    logger.warning(
        "GPU headroom low for %s: %.1f GB free < %.1f GB — evicting the "
        "warden LLM",
        caller, free_gb, required_gb,
    )
    unload_warden(admin_url)

    # Poll for the eviction to land; warden's 200 means "GPU clear", but a
    # few seconds of tolerance costs nothing and covers an async teardown.
    for _ in range(3):
        free = torch.cuda.mem_get_info()[0]
        if free / 1e9 >= required_gb:
            break
        time.sleep(2)

    free_gb = free / 1e9
    if free_gb < hard_floor_gb:
        raise RuntimeError(
            f"Not enough free VRAM for {caller} even after evicting the "
            f"warden LLM: {free_gb:.1f} GB free (need >= {hard_floor_gb:.1f} "
            f"GB). Stop the other GPU processes (python -m src.preflight "
            f"lists them) and retry."
        )
    logger.info("After eviction: %.1f GB free — OK for %s", free_gb, caller)
    return free
