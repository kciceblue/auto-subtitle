"""Preflight: is this machine ready to run the evidence-first workflow?

    python -m src.preflight [MEDIA ...]
    ./run.sh --check [MEDIA ...]        (./run.sh also runs it before every run; SKIP_PREFLIGHT=1 skips)

Takes a few seconds and loads no model. Checks ffmpeg, both Python runtimes, the model files,
the Warden LLM gateway, the Gemma llama-server, GPU memory held by other processes, other
active runs and free disk. Prints one line per check and a fix for each failure; exits 1
when any check fails.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
from typing import NamedTuple
import urllib.request

from src.evidence_first import ADMIN, ASR_MODELS, QWEN_MODEL, ROOT, WRITER_PROFILE
from src.local_backend import LOCAL_PORT

WARDEN_PORT = 8089
WARDEN_START = 'systemctl --user start model-warden'
RUNTIMES = {
    '.venv': ('torch', 'transformers', 'qwen_asr', 'sherpa_onnx', 'faster_whisper', 'soundfile', 'jsonschema',
              'pykakasi', 'janome'),
    '.venv-voxtral': ('torch', 'transformers', 'mistral_common', 'soxr', 'soundfile'),
}
_CU128 = '--extra-index-url https://download.pytorch.org/whl/cu128'
RUNTIME_FIX = {
    '.venv': f'python3 -m venv .venv && .venv/bin/pip install -r requirements.txt {_CU128}',
    '.venv-voxtral': f'python3 -m venv .venv-voxtral && .venv-voxtral/bin/pip install -r requirements-voxtral.txt {_CU128}',
}
MODELS = {'anime-whisper': ROOT / 'models/anime-whisper/model.safetensors',
          'voxtral': ROOT / 'models/voxtral-mini-4b-realtime-2602/model.safetensors',
          'qwen3-asr': ASR_MODELS / 'Qwen3-ASR-1.7B',
          'forced-aligner': ASR_MODELS / 'Qwen3-ForcedAligner-0.6B',
          'zipformer': ASR_MODELS / 'sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01'}
BANDIT = ROOT / 'models/bandit-v2/manifest.json'
ASSETS_FIX = 'restore it; WORKFLOW.md "Required local assets" lists every path'
# One 24-minute episode leaves about 2.2 GB under output/<stem>/work.
EPISODE_DISK_GB = 5
# GPU memory another process may hold; the ASR workers and the Gemma draft need the whole card.
FOREIGN_GPU_MIB = 1024
_FIND_SPEC = 'import importlib.util, sys; print(*(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))'


class Result(NamedTuple):
    ok: bool
    name: str
    detail: str
    fix: str = ''


def _run(argv: list[str], timeout: float = 30) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def _present(path: Path) -> bool:
    return path.is_file() or (path.is_dir() and any(path.iterdir()))


def check_tools() -> list[Result]:
    return [Result(bool(shutil.which(tool)), tool, shutil.which(tool) or 'not on PATH', fix)
            for tool, fix in (('ffmpeg', 'sudo apt install ffmpeg'), ('ffprobe', 'sudo apt install ffmpeg'),
                              ('nvidia-smi', 'install the NVIDIA driver'))]


def check_runtimes() -> list[Result]:
    out = []
    for venv, modules in RUNTIMES.items():
        python = ROOT / venv / 'bin/python'
        probe = _run([str(python), '-c', _FIND_SPEC, *modules], timeout=120) if python.exists() else None
        if probe is None or probe.returncode:
            out.append(Result(False, venv, f'{python} does not run', RUNTIME_FIX[venv]))
        else:
            missing = probe.stdout.split()
            out.append(Result(not missing, venv, f'missing {" ".join(missing)}' if missing else str(python),
                              RUNTIME_FIX[venv]))
    return out


def check_models() -> list[Result]:
    out = [Result(_present(path), name, str(path), ASSETS_FIX) for name, path in MODELS.items()]
    if not BANDIT.is_file():
        return [*out, Result(False, 'bandit-v2', f'{BANDIT} missing', ASSETS_FIX)]
    manifest = json.loads(BANDIT.read_text(encoding='utf-8'))
    gone = [key for key in ('python_executable', 'checkpoint_path') if not Path(manifest[key]).exists()]
    return [*out, Result(not gone, 'bandit-v2', f'missing {", ".join(manifest[k] for k in gone)}' if gone
                         else str(BANDIT.parent), ASSETS_FIX)]


def check_gemma() -> list[Result]:
    profile = json.loads(WRITER_PROFILE.read_text(encoding='utf-8'))
    weights = (WRITER_PROFILE.parent / profile['weights']).resolve()
    binary = Path(profile['server_binary'])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(('127.0.0.1', LOCAL_PORT))
            port_free = True
        except OSError:
            port_free = False
    return [Result(weights.is_file(), 'gemma weights', str(weights), ASSETS_FIX),
            Result(binary.is_file() and os.access(binary, os.X_OK), 'llama-server', str(binary),
                   f'build llama.cpp, or point server_binary in {WRITER_PROFILE.relative_to(ROOT)} at it'),
            Result(port_free, 'draft port', f'127.0.0.1:{LOCAL_PORT} {"free" if port_free else "in use"}',
                   f'probably a llama-server left by an interrupted draft: ss -ltnp "sport = :{LOCAL_PORT}", '
                   'then stop that process')]


def warden_status() -> dict | None:
    try:
        with urllib.request.urlopen(ADMIN + '/status', timeout=5) as response:
            state = json.loads(response.read())
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None


def check_warden(state: dict | None) -> list[Result]:
    if state is None:
        return [Result(False, 'warden', f'{ADMIN}/status unreachable', WARDEN_START)]
    busy = state.get('refcount') or 0
    return [Result(QWEN_MODEL in (state.get('registry') or []), 'warden model', QWEN_MODEL,
                   'not in the Warden registry; add it to the Warden model roster'),
            Result(not busy, 'warden', f"{state.get('state')}, loaded: {state.get('loaded_model') or 'nothing'}, "
                   f'{busy} requests in flight',
                   'another client is using the LLM; wait until it finishes (GPU stages must unload Warden)')]


def _parent(pid: int) -> int | None:
    try:
        stat = Path(f'/proc/{pid}/stat').read_text(encoding='utf-8')
    except OSError:
        return None
    return int(stat.rsplit(')', 1)[1].split()[1])


def descends(pid: int | None, ancestor: int | None) -> bool:
    while ancestor and pid and pid > 1:
        if pid == ancestor:
            return True
        pid = _parent(pid)
    return False


def _listener(port: int) -> int | None:
    listing = _run(['ss', '-ltnpH', f'sport = :{port}'])
    found = re.search(r'pid=(\d+)', listing.stdout) if listing else None
    return int(found.group(1)) if found else None


def foreign_gpu(apps: str, warden: int | None) -> list[str]:
    """GPU processes outside the Warden process tree that hold more than FOREIGN_GPU_MIB."""
    out = []
    for line in apps.splitlines():
        pid, rest = line.split(',', 1)
        name, mib = (x.strip() for x in rest.rsplit(',', 1))
        if mib.isdigit() and int(mib) > FOREIGN_GPU_MIB and not descends(int(pid), warden):
            out.append(f'pid {pid.strip()} {name} {mib} MiB')
    return out


def check_gpu() -> list[Result]:
    if not shutil.which('nvidia-smi'):
        return []
    query = _run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'])
    if query is None or query.returncode:
        return [Result(False, 'gpu', 'nvidia-smi failed', 'check the NVIDIA driver')]
    foreign = foreign_gpu(query.stdout, _listener(WARDEN_PORT))
    return [Result(not foreign, 'gpu', '; '.join(foreign) or 'no other process holds GPU memory',
                   'the workflow needs the whole GPU: wait for these processes or stop them')]


def check_runs() -> list[Result]:
    others = []
    for proc in Path('/proc').iterdir():
        if proc.name.isdigit() and int(proc.name) != os.getpid():
            try:
                if b'src.evidence_first' in (proc / 'cmdline').read_bytes().split(b'\0'):
                    others.append(proc.name)
            except OSError:
                continue
    return [Result(not others, 'other runs', f'src.evidence_first already running: pid {", ".join(others)}'
                   if others else 'none', 'GPU work is strictly serial: wait for that run to finish')]


def check_disk(episodes: int) -> list[Result]:
    free = shutil.disk_usage(ROOT / 'output').free / 1024 ** 3
    need = EPISODE_DISK_GB * max(1, episodes)
    return [Result(free >= need, 'disk', f'{free:.0f} GB free under output/, about {need} GB needed',
                   'free disk space; each finished episode can drop output/<stem>/work/')]


def check_media(paths: list[Path]) -> list[Result]:
    out = []
    for path in paths:
        if not path.is_file():
            out.append(Result(False, 'media', f'{path} not found', 'pass an existing media file'))
            continue
        probe = _run(['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index',
                      '-of', 'csv=p=0', str(path)])
        audio = bool(probe and probe.stdout.strip())
        out.append(Result(audio, 'media', f'{path.name}: {"audio stream found" if audio else "no audio stream"}',
                          'the workflow needs a file with an audio track'))
    return out


def run_checks(media: list[Path]) -> list[Result]:
    return [*check_tools(), *check_runtimes(), *check_models(), *check_gemma(), *check_warden(warden_status()),
            *check_gpu(), *check_runs(), *check_disk(len(media)), *check_media(media)]


def report(results: list[Result]) -> int:
    for result in results:
        print(f"{'ok  ' if result.ok else 'FAIL'}  {result.name:<14} {result.detail}")
        if not result.ok and result.fix:
            print(f"{'':20} fix: {result.fix}")
    failed = sum(not result.ok for result in results)
    print(f'preflight: {failed} check(s) failed' if failed else 'preflight: ready')
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('media', type=Path, nargs='*', help='media files about to be processed')
    return report(run_checks(parser.parse_args().media))


if __name__ == '__main__':
    raise SystemExit(main())
