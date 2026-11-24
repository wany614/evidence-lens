"""Record the exact source, historical controls, native controller and reachable Linux host."""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from evidence_lens._bundle import save, sha, utc


def output(args):
    return subprocess.check_output(args, text=True, timeout=40).strip()


def inventory(args):
    git = ["git", "-C", str(args.repo)]
    head = output(git + ["rev-parse", "HEAD"])
    refs = dict(line.split()[::-1] for line in output(
        git + ["ls-remote", "origin", "refs/heads/main", "refs/tags/v0.4.0", "refs/tags/v0.5.0"]
    ).splitlines())
    if head != refs.get("refs/heads/main"):
        raise ValueError("controller is not based on the observed public main")
    image = output(["docker", "image", "inspect", "--format", "{{.Id}}", "python:3.12-slim"])
    try:
        linux = json.loads(output(["docker", "run", "--rm", "--pull", "never", "--network", "none", "--read-only",
            image, "python", "-I", "-c", "import json,platform;print(json.dumps([platform.system(),platform.machine(),platform.python_version()]))"]))
    except (OSError, subprocess.SubprocessError) as exc:
        linux = {"not_executed": str(exc)}
    usage = shutil.disk_usage(args.store)
    environment = []
    for folder in [".venv", ".venv-cpu", ".venv-vision-cpu", ".venv-reference", ".venv-train"]:
        exe = args.repo / folder / "bin/python"
        if exe.exists():
            script = ("import importlib.metadata as m,json,platform;"
                      "wanted={'numpy','pillow','torch','bpy','evidence-lens','safetensors','psutil'};"
                      "print(json.dumps({'python':platform.python_version(),'packages':"
                      "{d.metadata['Name'].lower():d.version for d in m.distributions() if "
                      "d.metadata['Name'].lower() in wanted}}))")
            environment.append(dict(name=folder, **json.loads(output([str(exe), "-I", "-c", script]))))
    save(args.out / "start.json", {
        "schema":"el-dual-start-1", "run_id":os.environ["EL_RUN_ID"], "utc":utc(),
        "base_commit":head, "observed_refs":refs, "protocol_sha256":sha(args.protocol),
        "worktrees_private":output(git+["worktree","list","--porcelain"]),
        "observed_working_status":output(git+["status","--short","--branch"]),
        "mac":{"arch":platform.machine(),"os":platform.mac_ver()[0],"cpu":output(["sysctl","-n","machdep.cpu.brand_string"]),
               "memory_bytes":int(output(["sysctl","-n","hw.memsize"])),"environments":environment},
        "linux":{"observed":linux,"local_image":image,"network":"none","mode":"Docker under OrbStack"},
        "windows":"Separate native receipt requested; prior platform results belong to the previous release.",
        "external_store_free_bytes":usage.free,"external_store_capacity_bytes":usage.total,
        "prior_release_proof":json.loads(args.proof.read_text()),"formal_new_training":0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["repo","store","out","protocol","proof"]:
        parser.add_argument("--"+key, type=Path, required=True)
    inventory(parser.parse_args())
