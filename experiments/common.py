"""Shared helpers for experiment scripts: result files with run metadata.

Every experiment writes results/<track>/<name>.json:

    {"meta": {...git commit, seed, versions, hardware, wall time...},
     "data": {...the raw measurements...},
     "macros": {"MacroName": "value", ...}}     # picked up by scripts/make_tables.py

Numbers that appear in the paper come only from "macros" or from tables that
scripts/make_tables.py builds from "data".
"""
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def metadata(seed=None, extra=None):
    meta = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "seed": seed,
    }
    try:
        import numpy
        meta["numpy"] = numpy.__version__
    except ImportError:
        pass
    try:
        import torch
        meta["torch"] = torch.__version__
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import transformers
        meta["transformers"] = transformers.__version__
    except ImportError:
        pass
    if extra:
        meta.update(extra)
    return meta


def save(track, name, data, macros=None, seed=None, wall_time=None, extra_meta=None):
    out_dir = os.path.join(ROOT, "results", track)
    os.makedirs(out_dir, exist_ok=True)
    meta = metadata(seed, extra_meta)
    if wall_time is not None:
        meta["wall_time_s"] = round(wall_time, 2)
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "data": data, "macros": macros or {}}, f, indent=1)
    print(f"wrote {os.path.relpath(path, ROOT)}")
    return path


@contextmanager
def timer():
    t = {"start": time.time()}
    yield t
    t["elapsed"] = time.time() - t["start"]


def pool_map(fn, items, processes=None, chunksize=64):
    """Process-parallel map (each worker builds its own circuits once)."""
    import multiprocessing as mp
    processes = processes or max(1, (os.cpu_count() or 2) - 1)
    with mp.get_context("spawn").Pool(processes) as pool:
        return pool.map(fn, items, chunksize=chunksize)
