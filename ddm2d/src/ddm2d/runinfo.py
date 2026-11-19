"""Run metadata: git commit, host, threads, peak memory."""
from __future__ import annotations

import os
import platform
import resource
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def git_commit() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:  # pragma: no cover
            return ""
    status = run("status", "--porcelain", "--", "src", "configs")
    return {"commit": run("rev-parse", "HEAD"), "dirty_src_or_configs": bool(status)}


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0      # Linux: KiB


def environment(threads: int) -> dict:
    import numpy
    import torch
    return {"host": socket.gethostname(), "python": sys.version.split()[0], "torch": torch.__version__, "numpy": numpy.__version__,
            "platform": platform.platform(), "device": "cpu", "torch_threads": torch.get_num_threads(), "threads_requested": threads,
            "affinity_cpus": len(os.sched_getaffinity(0)), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "cuda_available": torch.cuda.is_available()}
