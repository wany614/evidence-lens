"""Inventory the Mac controller and an actual network-isolated Linux container for this build."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

from evidence_lens._bundle import save, utc


def output(*args):
    return subprocess.check_output(args, text=True).strip()


def inventory(store, out, image):
    docker = shutil.which("docker")
    inside = ("import json,platform,sys,os;print(json.dumps(dict(system=platform.system(),"
              "arch=platform.machine(),python=platform.python_version(),logical_cpus=os.cpu_count())))")
    linux = json.loads(output(docker,"run","--rm","--network","none","--read-only",image,"python","-I","-c",inside))
    linux["image_digest"] = output(docker,"image","inspect","--format","{{.Id}}",image)
    linux["environment"] = "Docker container under OrbStack"
    usage=shutil.disk_usage(store)
    save(out/"platforms.json",{"schema":"el-general-G0-mac-linux-1","run_id":os.environ["EL_RUN_ID"],
         "utc":utc(),"mac":{"system":platform.system(),"arch":platform.machine(),"version":platform.mac_ver()[0],
         "python":platform.python_version(),"logical_cpus":os.cpu_count(),"total_ram_bytes":int(output("sysctl","-n","hw.memsize")),
         "cpu":output("sysctl","-n","machdep.cpu.brand_string")},"linux":linux,
         "external_store_free_bytes":usage.free,"external_store_total_bytes":usage.total,
         "recorder_interpreter_installed_distribution":version("evidence-lens"),
         "previous_verified_portable_wheel":"0.4.0b1","acceptance_of_new_build":"pending"})


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--image",required=True)
    a=p.parse_args()
    inventory(a.store,a.out,a.image)
