"""Execute the owner's fixed B confirmation queue using the published native Windows runners."""
import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

HASH = re.compile(r"[0-9a-f]{64}")
RID = re.compile(r"[0-9]{8}T[0-9]{12}Z-[0-9a-f]{16}")
QUEUE = "v11-b-confirmation-v1"
QUOTA = "sealed computation retained; outgoing packaging deferred by configured disk limits"
ROLES = {"B0": "generation", "R1": "generation", "Q1": "derivation"}
RUNNERS = {"B0": "run_windows_batch.py", "stress": "run_conditions_batch.py"}


def need(condition, msg):
    if not condition:
        raise ValueError(msg)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def signed(path, wanted):
    path = Path(path)
    need(isinstance(wanted, str) and HASH.fullmatch(wanted) and path.is_file() and not path.is_symlink()
         and sha(path) == wanted, 'Wrong pinned file: ' + path.name)
    return load(path)


def check_plan(p, digest, self_path):
    plan = signed(p, digest)
    need(plan['schema'] == 'el-stage2-confirmation-plan-1' and plan['queue'] == QUEUE
         and HASH.fullmatch(plan['catalog_sha256']) and plan['display_rule'] == 'v01'
         and plan['windows_sources'][Path(self_path).name] == sha(self_path), 'Wrong confirmation queue or script')
    rows = plan['families']
    need(len(rows) == 8 and len({r['family'] for r in rows}) == 8, 'Expected eight distinct families')
    for n, row in enumerate(rows, 41):
        family = row['family']
        need(re.fullmatch(r'scene_'+str(n).zfill(3)+r'_[a-z0-9_]+', family) and len(row['batches']) == 2,
             'Wrong family order or batch count')
        for block, label, subset in zip(row['batches'], ('B0','stress'), (('B0',),('R1','Q1'))):
            need(block['kind'] == label and block['batch'] == f'v11-b-confirm-{n:03}-{label.lower()}'
                 and block['runner'] == RUNNERS[label]
                 and block['trials'] == [family+'-'+c for c in subset], 'Unexpected batch or condition')
    return plan


def auth(path, expected, plan, plan_sha):
    a = signed(path, expected)
    need(a['schema'] == 'el-stage2-confirmation-authorization-1' and a['plan_sha256'] == plan_sha
         and a['catalog_sha256'] == plan['catalog_sha256'] and a['queue'] == QUEUE
         and a['test_family_ids'] == [r['family'] for r in plan['families']]
         and RID.fullmatch(a['freeze_run_id']) and all(HASH.fullmatch(a[k]) for k in
         ('freeze_manifest_sha256','opening_sha256','method_sha256')), 'Wrong opening authority')
    return a


def check_batch(path, digest, plan, block, opening):
    batch = signed(path, digest)
    key = 'family_catalog_sha256' if block['kind'] == 'B0' else 'catalog_sha256'
    schema = 'el-windows-b0-batch-1' if block['kind'] == 'B0' else 'el-windows-condition-batch-1'
    need(batch['schema'] == schema and batch['batch_id'] == block['batch']
         and batch[key] == plan['catalog_sha256'] and [j['trial_id'] for j in batch['jobs']] == block['trials']
         and batch['runner_sha256'] == plan['windows_sources'][block['runner']], 'Registered native batch differs')
    if block['kind'] == 'stress':
        e = batch['opening']
        need(e['opening_sha256'] == opening['opening_sha256'] and e['method_sha256'] == opening['method_sha256']
             and e['freeze_run_id'] == opening['freeze_run_id'] and e['freeze_manifest_sha256'] == opening['freeze_manifest_sha256']
             and batch['transport_sha256'] == plan['windows_sources'][RUNNERS['B0']], 'Stress opening/transport differs')
    for j in batch['jobs']:
        need(re.fullmatch(r'[a-z0-9A-Z_-]+\.zip', j['name']) and type(j['bytes']) is int and j['bytes'] > 0
             and HASH.fullmatch(j['sha256']) and HASH.fullmatch(j['handoff_sha256']), 'Invalid job descriptor')
    return batch


def return_state(folder, jobs, *, check_bytes=True):
    results = {j['trial_id']: {'status':'waiting_for_return','attempts':[], 'in_transit':[]} for j in jobs}
    bindings = {j['trial_id']:j['handoff_sha256'] for j in jobs}
    for path in sorted(folder.glob('*.receipt.json')):
        record = load(path)
        trial = record.get('trial_id')
        need(trial in results, 'Return has no registered job in this exclusive batch')
        rid = record['run']; kind = record['run_kind']; status = record['status']
        need(record['schema'] == 'el-worker-return-1' and RID.fullmatch(rid) and path.name == rid+'.receipt.json'
             and kind in (ROLES[trial.rsplit('-',1)[1]],'render') and record['handoff_sha256'] == bindings[trial]
             and record['package'] == rid+'.zip' and HASH.fullmatch(record['sha256'])
             and HASH.fullmatch(record['manifest_sha256']) and type(record['bytes']) is int and record['bytes'] > 0
             and status in {'succeeded','failed','interrupted','cancelled'}
             and ((status == 'succeeded') == (record['exit_code'] == 0)), 'Malformed or unrelated return envelope')
        package = folder/record['package']
        if not package.exists() or package.stat().st_size < record['bytes']:
            results[trial]['in_transit'].append(rid); continue
        need(package.is_file() and not package.is_symlink() and package.stat().st_size == record['bytes']
             and (not check_bytes or sha(package) == record['sha256']), 'Return container integrity differs')
        results[trial]['attempts'].append({'run_id':rid, 'manifest_sha256':record['manifest_sha256'], 'kind':kind,
            'status':status, 'container_sha256':record['sha256'], 'bytes':record['bytes'], 'receipt_sha256':sha(path)})
    for trial, state in results.items():
        complete = {a['kind'] for a in state['attempts'] if a['status']=='succeeded'}
        if state['in_transit']:
            state['status']='in_transit'
        elif complete == {ROLES[trial.rsplit('-',1)[1]],'render'}:
            state['status']='returned'
        elif any(a['status']!='succeeded' for a in state['attempts']):
            state['status']='sealed_failure_review'
    return results


def notes(folder, jobs):
    expected = {j['trial_id']:j['handoff_sha256'] for j in jobs}
    result = []
    for p in sorted(folder.glob('controller-*.json')):
        item = load(p); trial = item.get('trial_id', item.get('trial'))
        need(trial in expected and item.get('handoff_sha256', item.get('handoff')) == expected[trial], 'Foreign controller note')
        err = str(item.get('stderr_tail',''))+' '+str(item.get('stdout_tail',''))
        result.append({'file':p.name, 'sha256':sha(p), 'trial':trial,
                       'kind':'packaging_quota' if QUOTA in err else 'controller_error'})
    return result


@contextlib.contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db=sqlite3.connect(path,timeout=1)
    try:
        db.execute('BEGIN IMMEDIATE'); yield; db.commit()
    finally:
        db.close()


def save_local(path, value):
    tmp=path.with_suffix('.partial'); tmp.write_text(json.dumps(value,sort_keys=True)+'\n',encoding='utf8'); os.replace(tmp,path)


def once(args, plan, opening):
    base = args.exchange/'Jobs'/QUEUE
    for row in plan['families']:
        for b in row['batches']:
            label=b['batch']; marker=base/'ready'/(label+'.json'); local=args.state/(label+'.json')
            if not marker.is_file(): return {'status':'awaiting_mac_ticket','batch':label}
            ticket=load(marker)
            need(ticket['schema']=='el-stage2-confirmation-batch-ticket-1' and ticket['plan_sha256']==args.plan_sha256
                 and ticket['authorization_sha256']==args.authorization_sha256 and ticket['batch']==label
                 and ticket['family']==row['family'] and ticket['trials']==b['trials'], 'Unbound queue ticket')
            folder=args.exchange/'Jobs'/label
            paths=[folder/'batch.json',folder/b['runner']]
            if b['kind']=='stress': paths.append(folder/RUNNERS['B0'])
            if not all(p.is_file() for p in paths): return {'status':'awaiting_job_sync','batch':label}
            batch=check_batch(paths[0],ticket['batch_sha256'],plan,b,opening)
            for f in paths[1:]: need(sha(f)==plan['windows_sources'][f.name], 'Unexpected native worker script')
            incoming=batch['jobs']
            if any(not (folder/j['name']).is_file() or (folder/j['name']).stat().st_size<j['bytes'] for j in incoming):
                return {'status':'awaiting_job_sync','batch':label}
            for j in incoming:
                jpath=folder/j['name']; need(jpath.stat().st_size==j['bytes'] and sha(jpath)==j['sha256'],'Job bytes differ')
            ret=args.exchange/'Returns'/label; prior=load(local) if local.is_file() else {}
            if prior:
                need(prior.get('ticket_sha256')==sha(marker) and prior['authorization_sha256']==args.authorization_sha256,
                     'Local resume belongs to a different batch or opening')
            if prior.get('status')=='published': continue
            returns=return_state(ret,incoming)
            states={p['status'] for p in returns.values()}
            record={'ticket_sha256':sha(marker),'authorization_sha256':args.authorization_sha256,'batch':label}
            if states=={'returned'}:
                save_local(local,dict(record,status='published',returns=returns)); continue
            if 'in_transit' in states: return dict(record,status='awaiting_return_sync')
            if 'sealed_failure_review' in states: return dict(record,status='sealed_failure_review',returns=returns)
            if prior.get('status') in ('packaging_quota','controller_review') and args.resume_batch!=label:
                return prior
            cmd=[sys.executable,str(folder/b['runner']),'--batch',str(paths[0]),'--sha256',ticket['batch_sha256']]
            for key,value in [('staged',args.staged),('store',args.store),('outgoing',args.outgoing),('weights',args.weights),('publish',ret)]:
                cmd.extend(['--'+key,str(value)])
            args.resume_batch=None
            out=subprocess.run(cmd,check=False)
            statuses=return_state(ret,incoming); diagnostics=notes(ret,incoming)
            if all(t['status']=='returned' for t in statuses.values()): status='published'
            elif any(t['status']=='sealed_failure_review' for t in statuses.values()): status='sealed_failure_review'
            elif out.returncode: status='packaging_quota' if any(x['kind']=='packaging_quota' for x in diagnostics) else 'controller_review'
            else: status='awaiting_return_sync'
            record.update(status=status,exit_code=out.returncode,returns=statuses,controller_notes=diagnostics,
                          free_bytes=shutil.disk_usage(args.store).free)
            save_local(local,record); return record
    return {'status':'all_16_batches_published','families':8,'conditions':24}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True,type=Path); p.add_argument('--plan-sha256',required=True)
    p.add_argument('--dry-run',action='store_true'); p.add_argument('--once',action='store_true')
    for k in ('exchange','staged','store','state','outgoing','weights'): p.add_argument('--'+k,type=Path)
    p.add_argument('--authorization-sha256'); p.add_argument('--resume-batch'); p.add_argument('--poll',type=int,default=15)
    a=p.parse_args(argv); plan=check_plan(a.plan,a.plan_sha256,Path(__file__))
    if a.dry_run:
        print(json.dumps({'schema':'el-stage2-queue-dry-run-1','operation':'list_only','plan_sha256':a.plan_sha256,
              'families':8, 'trial_count':24,'batches':[b for row in plan['families'] for b in row['batches']]})); return
    need(os.name=='nt' and all((a.exchange,a.staged,a.store,a.state,a.outgoing,a.weights,a.authorization_sha256))
         and 1<=a.poll<=60, 'Live queue needs Windows, explicit local paths and owner authorization digest')
    opening=auth(a.exchange/'Jobs'/QUEUE/'authorization.json',a.authorization_sha256,plan,a.plan_sha256)
    with lock(a.state/'queue.sqlite3'):
        previous=None
        while True:
            result=once(a,plan,opening)
            if result!=previous: print(json.dumps(result),flush=True); previous=result
            if a.once or result['status'] in ('all_16_batches_published','sealed_failure_review','controller_review','packaging_quota'): return
            if result['status']!='published': time.sleep(a.poll)


if __name__=='__main__': main()
