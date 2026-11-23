"""Sample the launched process and its descendants while the payload is running."""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path

import psutil

from ._bundle import save, utc

FIELDS = {
    "utc": "RFC 3339 UTC", "monotonic_seconds": "seconds on producer monotonic clock",
    "elapsed_seconds": "seconds since run started", "pid": "launched process ID",
    "sampled_pids": "JSON list of observed payload/descendant PIDs",
    "tree_process_count": "count of observed processes",
    "tree_rss_bytes": "sum of psutil RSS bytes over observed processes; shared pages may repeat",
    "tree_cpu_percent": "sum of per-process CPU time deltas, 100 per logical core",
    "system_available_bytes": "psutil.virtual_memory().available, bytes",
    "system_memory_used_percent": "psutil.virtual_memory().percent, percent",
    "mac_memory_pressure_level": "kern.memorystatus_vm_pressure_level integer, if exported",
    "gpu_devices": "JSON device-wide nvidia-smi samples, MiB / percent / degC / W",
    "gpu_process_memory_bytes": "per-process GPU memory; unavailable in this sampler",
    "missing_reason": "JSON field-to-reason map; blank measurements are null",
}


def _read(command: list[str]) -> tuple[str, str | None]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=2,
                                check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return "", type(error).__name__
    return result.stdout.strip(), None if result.returncode == 0 else f"exit_{result.returncode}"


class Sampler:
    def __init__(self, bundle: Path, start: float, interval: float):
        self.bundle = bundle
        self.start = start
        self.interval = interval
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.failure: Exception | None = None
        self.previous: dict = {}
        self.previous_time: float | None = None
        self.peak_rss: int | None = None
        self.nvidia = shutil.which("nvidia-smi")
        self.mac = platform.system() == "Darwin"
        self.log = (bundle / "telemetry.csv").open("w", encoding="utf-8", newline="")
        self.rows = csv.DictWriter(self.log, fieldnames=FIELDS)
        self.rows.writeheader()
        self.log.flush()
        save(bundle / "telemetry_schema.json",
             {"schema_version": "1.0", "interval_seconds": interval, "fields": FIELDS,
              "rss_scope": "payload plus descendants, excludes recorder",
              "peak_kind": "maximum observed sample; short-lived peaks can be missed",
              "gpu_scope": "device-wide, including other processes; background load unknown"})

    def launch(self, pid: int) -> None:
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            process = None

        def run() -> None:
            try:
                while True:
                    self.sample(pid, process)
                    if self.stop.wait(self.interval):
                        break
            except Exception as error:
                self.failure = error

        self.thread = threading.Thread(target=run, name="evidence-lens-telemetry")
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        if self.thread:
            self.thread.join()
        self.log.close()

    def sample(self, pid: int, process: psutil.Process | None) -> None:
        clock = time.monotonic()
        row = {"utc": utc(), "monotonic_seconds": clock,
               "elapsed_seconds": clock - self.start, "pid": pid}
        reasons = {"gpu_process_memory_bytes": "per_process_counter_not_collected"}
        children = []
        if process is not None:
            try:
                children = [process, *process.children(recursive=True)]
            except psutil.Error:
                children = [process]
        usage = {}
        pids = []
        rss = 0
        for child in children:
            try:
                with child.oneshot():
                    ident = (child.pid, child.create_time())
                    cpu = child.cpu_times()
                    memory = child.memory_info().rss
                usage[ident] = cpu.user + cpu.system
                pids.append(child.pid)
                rss += memory
            except psutil.Error:
                reasons["sampled_pids"] = "some_processes_exited_or_inaccessible"
        row.update(sampled_pids=json.dumps(sorted(pids)), tree_process_count=len(pids))
        if pids:
            row["tree_rss_bytes"] = rss
            self.peak_rss = max(self.peak_rss or 0, rss)
        else:
            reasons["tree_rss_bytes"] = "no_live_process_observed"
        common = usage.keys() & self.previous.keys()
        if self.previous_time is not None and clock > self.previous_time and common:
            row["tree_cpu_percent"] = sum(max(0, usage[k] - self.previous[k]) for k in common) * 100 / (clock - self.previous_time)
        else:
            reasons["tree_cpu_percent"] = "no_preceding_sample_for_live_process"
        self.previous_time, self.previous = clock, usage
        vm = psutil.virtual_memory()
        row.update(system_available_bytes=vm.available, system_memory_used_percent=vm.percent)
        if self.mac:
            value, error = _read(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
            if value.isdigit() and error is None:
                row["mac_memory_pressure_level"] = int(value)
            else:
                reasons["mac_memory_pressure_level"] = error or "unsupported_field"
        else:
            reasons["mac_memory_pressure_level"] = "not_macos"
        if self.nvidia:
            fields = ["uuid", "memory.used", "utilization.gpu", "temperature.gpu", "power.draw"]
            result, error = _read([self.nvidia, "--query-gpu=" + ",".join(fields),
                                   "--format=csv,noheader,nounits"])
            devices = []
            for values in csv.reader(result.splitlines(), skipinitialspace=True):
                if len(values) != len(fields):
                    continue
                measurements = {"uuid": values[0]}
                for field, value in zip(fields[1:], values[1:]):
                    try:
                        measurements[field] = float(value)
                    except ValueError:
                        measurements[field] = None
                        reasons[f"{values[0]}.{field}"] = value or "unsupported_field"
                devices.append(measurements)
            if devices and not error:
                row["gpu_devices"] = json.dumps(devices)
            else:
                reasons["gpu_devices"] = error or "no_device_sample"
        else:
            reasons["gpu_devices"] = "nvidia_smi_unavailable"
        row["missing_reason"] = json.dumps(reasons, sort_keys=True)
        self.rows.writerow(row)
        self.log.flush()
