"""Run one original Windows optimizer and its endpoint renderer from an authenticated T0 job ZIP."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import zipfile


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def extract(files, out):
    with zipfile.ZipFile(files) as z:
        names = z.namelist()
        if len(set(names)) != len(names):
            raise ValueError('Duplicate ZIP member')
        for name in names:
            parts = PurePosixPath(name)
            if parts.is_absolute() or '..' in parts.parts or '\\' in name or ':' in name or str(parts) != name:
                raise ValueError('Invalid portable ZIP member')
            target = out / parts
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)


def main(args):
    if os.name != 'nt' or digest(args.package) != args.sha256:
        raise ValueError('Run the SHA-pinned package under the project Windows CUDA interpreter')
    args.store.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.store.parent) as tmp:
        stage = Path(tmp)
        extract(args.package, stage)
        ready = json.loads((stage / 'READY.json').read_bytes())
        for name, spec in ready['members'].items():
            f = stage / name
            if f.stat().st_size != spec['bytes'] or digest(f) != spec['sha256']:
                raise ValueError('Incorrect transfer member ' + name)
        work = stage / 'original-worker'
        extract(stage / 'worker.zip', work)
        target = args.returns / ready['trial_id']
        common = ['--store', str(args.store.resolve()), '--returns', str(target.resolve()), '--gpu-slot',
                  str(args.gpu_slot.resolve()), '--cell', str(stage / 'cell'), '--job-sha256', ready['job_sha256']]
        prefix = [sys.executable, '-B', str(work / 'scripts/recovery_windows.py')]
        rows = []
        for leg in ('optimize', 'render'):
            if leg == 'optimize':
                more = ['--recipe', str(stage / 'recipe.json')]
            else:
                gen = rows[0]
                more = ['--generation', gen['run'] + '=' + gen['manifest_sha256'], '--steps', '5000']
            call = subprocess.run([*prefix, leg, *common, *more], capture_output=True, text=True)
            if not call.stdout.strip():
                raise RuntimeError(call.stderr or 'Original recorder produced no return receipt')
            rec = json.loads(call.stdout.splitlines()[-1])
            if rec['trial_id'] != ready['trial_id'] or rec['cell_sha256'] != ready['job_sha256']:
                raise ValueError('Wrong recorded cell')
            rows.append(rec)
            if call.returncode or rec['status'] != 'succeeded':
                break
        files = {}
        for p in sorted(target.rglob('*')):
            if p.is_file() and p.name != 'completed.json':
                files[p.relative_to(target).as_posix()] = {'sha256': digest(p), 'bytes': p.stat().st_size}
        receipt = {'schema': 'el-t0-windows-return-1', 'trial_id': ready['trial_id'], 'package_sha256': args.sha256,
                   'job_sha256': ready['job_sha256'], 'runs': [{k: r[k] for k in ('run', 'run_kind', 'manifest_sha256', 'sha256', 'status')} for r in rows],
                   'files': files, 'status': 'succeeded' if len(rows) == 2 and all(r['status'] == 'succeeded' for r in rows) else 'failed'}
        data = (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode()
        final = target / 'completed.json'
        if final.exists() and final.read_bytes() != data:
            raise ValueError('Existing immutable return summary differs')
        if not final.exists():
            with final.open('xb') as stream:
                stream.write(data)
        print(json.dumps({'trial_id': ready['trial_id'], 'summary_sha256': digest(final), 'runs': receipt['runs'], 'status': receipt['status']}))
        return 0 if receipt['status'] == 'succeeded' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'store', 'returns', 'gpu-slot'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--sha256', required=True)
    raise SystemExit(main(p.parse_args()))
