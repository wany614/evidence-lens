"""Seal T0 controller stages using the existing recorder and explicit parent members."""
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

from evidence_lens._bundle import encoded, load, save, sha
from evidence_lens._verify import require
from dtu_eval import read_pinned


def bind(store, rid, member):
    """Turn an already identified recorder member into a portable content pin."""
    folder = store / 'runs' / rid
    return {'run_id': rid, 'manifest_sha256': load(folder/'completion.json')['manifest_sha256'],
            'member': member, 'sha256': sha(folder/member)}


def resolve(roots, pin):
    return read_pinned(roots, pin)[1]


def record(store, history, name, kind, pins, config, argv, receipt, estimate, extra_files=None, retry=None):
    project = Path(__file__).resolve().parents[1]
    work = store / 'work' / 'controller' / name
    work.mkdir(parents=True, exist_ok=True)
    roots = (store, history)
    with (work/'controller.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sources = {role:resolve(roots, pin) for role,pin in pins.items()}
        config = {'experiment_id':'t0-dtu-confirmation-v1', 'trial_id':'t0-'+name,
                  'recording':{'expected_output_bytes':estimate},'sources':pins, **config}
        extra_files = extra_files or {}
        config['extra_sha256'] = {n:sha(p) for n,p in extra_files.items()}
        conf = work/'config.json'
        if conf.exists():
            require(conf.read_bytes() == encoded(config), 'the registered stage configuration has changed')
        else:
            save(conf, config)
        if (work/'selected.json').exists():
            pin=load(work/'selected.json'); resolve(roots,pin);return pin
        logs=sorted(work.glob('attempt-*.jsonl'))
        if logs:
            prev=json.loads(logs[-1].read_text().splitlines()[-1])
            if prev['execution_status']=='succeeded':
                selected=bind(store,prev['run_id'],receipt); save(work/'selected.json',selected);return selected
            require(retry==prev['run_id'], 'failed stage requires a recorded fix and an explicit retry')
        else:
            require(retry is None,'unexpected retry')
        cmd=[sys.executable,'-m','evidence_lens.recorder','--source-root',str(project),'--root',str(store),
             '--kind',kind,'--config',str(conf)]
        for rid,m in sorted({p['run_id']:p['manifest_sha256'] for p in pins.values()}.items()):
            cmd+=['--parent',rid+'='+m]
        for role,path in sources.items():
            cmd+=['--input',role+'='+str(path),'--input-parent',role+'='+pins[role]['run_id']]
        for role,path in extra_files.items():
            cmd+=['--input',role+'='+str(path)]
        if retry: cmd+=['--retry-of',retry]
        log=work/f'attempt-{len(logs)+1:02d}.jsonl'
        with log.open('xb') as f:
            proc=subprocess.run(cmd+['--',*argv],cwd=project,stdout=f,stderr=f,
                                env={**os.environ,'PYTHONPATH':str(project/'src'),'PYTHONDONTWRITEBYTECODE':'1'})
        require(proc.returncode==0,'recorded failure in '+str(log)+': '+log.read_text()[-500:])
        selected=bind(store,json.loads(log.read_text().splitlines()[-1])['run_id'],receipt)
        resolve(roots,selected)
        save(work/'selected.json',selected)
        return selected
