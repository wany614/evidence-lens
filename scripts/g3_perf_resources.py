"""Resource observations for recorded engineering measurements. One busy logical core is 100%."""

from __future__ import annotations

import contextlib
import io
import os
import platform
import threading
import time

import numpy as np
import psutil


def available_cpus():
    try:
        return len(psutil.Process().cpu_affinity())
    except (AttributeError, NotImplementedError, psutil.Error):
        return os.cpu_count() or 1


def environment(method):
    pools = None
    try:
        from threadpoolctl import threadpool_info

        pools = threadpool_info()
    except ImportError:
        pass
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        np.show_config()
    env = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "logical_cpus": os.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "available_logical_cpus": available_cpus(),
        "numpy_build": buf.getvalue(),
        "loaded_thread_pools": pools,
        "thread_env": {
            k: os.getenv(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")
        },
        "cpu_normalization": "100% per busy logical CPU; machine fraction divides by available logical CPUs",
    }
    if method in ("MS-ft", "MS-frozen"):
        import torch

        env.update(
            torch=str(torch.__version__),
            torch_intraop_threads=torch.get_num_threads(),
            torch_interop_threads=torch.get_num_interop_threads(),
            torch_parallel_backend=torch.__config__.parallel_info(),
        )
    return env


class Monitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.proc = psutil.Process()
        self.stop = threading.Event()
        self.samples = []
        self.first, self.latest = {}, {}

    def sample(self):
        rss = threads = 0
        try:
            procs = [self.proc, *self.proc.children(recursive=True)]
        except psutil.Error:
            procs = [self.proc]
        for p in procs:
            try:
                identity = (p.pid, p.create_time())
                cpu = p.cpu_times()
                self.first.setdefault(identity, cpu.user + cpu.system)
                self.latest[identity] = cpu.user + cpu.system
                rss += p.memory_info().rss
                threads += p.num_threads()
            except psutil.Error:
                continue
        self.samples.append(
            {
                "monotonic": time.monotonic(),
                "tree_rss_bytes": rss,
                "tree_os_threads": threads,
                "tree_cpu_seconds_since_first_observation": sum(
                    self.latest[k] - v for k, v in self.first.items()
                ),
            }
        )

    def watch(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.wall = time.monotonic()
        self.cpu = time.process_time()
        self.sample()
        self.thread = threading.Thread(target=self.watch, name="resource-observer", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()
        self.span = time.monotonic() - self.wall
        self.own_cpu = time.process_time() - self.cpu

    def summary(self):
        # Scoring has no child process. Current-process CPU supplies sub-tick resolution here.
        child_seen = len(self.latest) > 1
        cpu = self.samples[-1]["tree_cpu_seconds_since_first_observation"] if child_seen else self.own_cpu
        return {
            "wall_seconds": self.span,
            "process_cpu_seconds": self.own_cpu,
            "process_tree_cpu_seconds": cpu,
            "cpu_seconds_source": "sum of observed PID CPU deltas"
            if child_seen
            else "process_time; no child observed",
            "mean_busy_logical_cpus": cpu / self.span,
            "available_cpu_fraction": cpu / self.span / available_cpus(),
            "sample_period_seconds": self.interval,
            "samples": len(self.samples),
            "rss_start_bytes": self.samples[0]["tree_rss_bytes"],
            "rss_end_bytes": self.samples[-1]["tree_rss_bytes"],
            "peak_sampled_tree_rss_bytes": max(x["tree_rss_bytes"] for x in self.samples),
            "peak_sampled_os_threads": max(x["tree_os_threads"] for x in self.samples),
        }
