"""Bind the T0 roster and authorize one collective reference opening from terminal score or failure receipts."""
import argparse
import os
from pathlib import Path

from dtu_score import METHODS, RELEASE_SHA, SCOPES, VIEWS
from evidence_lens._bundle import load, save, sha, utc
from evidence_lens._verify import require
from evidence_lens.multisource_io import Case
from dtu_eval import read_pinned, score_map


def freeze(config, roster_file, baseline_file, release_file, protocol_file, out):
    protocol = load(protocol_file)
    roster = load(roster_file)
    baseline = load(baseline_file)
    release = load(release_file)
    project = Path(__file__).resolve().parents[1]
    for name, digest in protocol['source_sha256'].items():
        require(sha(project/name) == digest, 'historical implementation or protocol differs: '+name)
    for name, digest in config['new_scientific_files'].items():
        require(sha(project/name) == digest, 'T0 implementation changed before freezing: '+name)
    require(sha(release_file) == RELEASE_SHA == protocol['release_index_sha256'] and release['models'] == protocol['models'], 'release changed')
    require(baseline['recipe_sha256'] == protocol['source_sha256']['protocol/recovery-gsplat-scale-decay-v1.json'] and baseline['stop_step'] == protocol['frozen_endpoint_updates'] == 5000, 'not the selected endpoint')
    cam = protocol['camera']
    require(roster['source_view_ids'] == cam['sources'] and roster['target_view_ids'] == cam['targets'] and roster['lighting_id'] == cam['light'] and roster['camera']['crop_sha256'] == cam['crop_sha256'] and roster['camera']['view_set_sha256'] == cam['views_sha256'] and roster['original_photo_bodies_read_during_selection'] == 0, 'non-matching camera, crop, lighting or intake exposure')
    objects = roster['objects']
    ids = [r['object_id'] for r in objects]
    scans = [r['scan_id'] for r in objects]
    require(roster['schema']=='el-t0-dtu-roster-1' and 0 < len(objects) <= 6 and scans == sorted(set(scans)) and all(n not in (1,6) for n in scans), 'invalid T0 object register')
    require(len(set(ids)) == len(set(r['physical_family_id'] for r in objects)) == len(ids), 'duplicate object or family')
    for obj in objects:
        require(obj['object_id']=='set'+str(obj['scan_id']) and obj.get('physical_identity_evidence') and set(obj['source_members'])==set(protocol['camera']['sources']) and set(obj['target_members'])==set(VIEWS), 'incomplete object or source authority')
    freeze = {'schema':'el-t0-freeze-1','run_id':os.environ['EL_RUN_ID'],'utc':utc(),'sources':config['sources'],
              'protocol_sha256':sha(protocol_file),'protocol':protocol,'roster_sha256':sha(roster_file),'object_ids':ids,
              'families':{o['object_id']:o['physical_family_id'] for o in objects},'conditions':['M2','M4'],'target_ids':list(VIEWS),
              'recipe_sha256':baseline['recipe_sha256'],'stop_step':5000,'models':release['models'],
              'release_index_sha256':RELEASE_SHA,'new_scientific_files':config['new_scientific_files'],
              'expected_fits':len(ids)*2,'expected_scores':len(ids)*2*len(VIEWS)*len(METHODS),
              'target_reference_permission':'closed','required_before_scientific_dispatch':'successful independent D forced-fallback fixture receipt'}
    save(out/'freeze.json', freeze)


def scored_cell(roots, obj, cond, cell, freeze_pin, fr):
    _, casefile = read_pinned(roots,cell['case'])
    _, proof = read_pinned(roots,cell['readback'])
    _, index = read_pinned(roots,cell['scores'])
    case=Case(casefile.parent,cell['case']['sha256'])
    rec=load(proof); scores=load(index)
    require(rec['object_id']==obj and rec['condition']==case.condition==cond and rec['verified'] is True and rec['target_reference_files_opened']==0 and rec['case_sha256']==case.sha256 and not rec['forced_fallback_fixture'] and rec['freeze']==freeze_pin and rec['completed_steps']==fr['stop_step'] and rec['recipe_sha256']==fr['recipe_sha256'], 'wrong terminal reconstruction')
    require(cell['case']['run_id']==cell['readback']['run_id'] and scores['schema']=='el-dtu-frozen-scores-1' and scores['method_scopes']==SCOPES and scores['release_sha256']==fr['release_index_sha256'] and scores['models']==fr['models'] and scores['case_sha256']==case.sha256 and scores['checkpoint_sha256']==case.meta['checkpoint_sha256'] and scores['reconstruction_run_id']==case.meta['reconstruction_run_id'] and scores['condition']==cond and set(scores['views'])==set(VIEWS),'unbound prediction or scores')
    ready = 0
    for view, entry in scores['views'].items():
        if view in case.failed:
            require(entry['status']=='render_failed' and entry['reason']==case.failed[view] and entry['methods']=={}, 'missing render failure evidence')
        else:
            require(entry['status']=='ready' and set(entry['methods'])==set(METHODS), 'incomplete scores')
            for method in METHODS:
                # Structural/hash/state audit only. No reference, label or error is available here.
                score_map(index.parent,scores,method,view,case,fr['models'][method]['threshold'])
                ready += 1
    return ready


def opening(config, roots, request_file, out):
    request=load(request_file)
    fp=request['freeze']; _, frp=read_pinned(roots,fp); fr=load(frp)
    require(fr['schema']=='el-t0-freeze-1' and fr['target_reference_permission']=='closed', 'unbound study')
    cells=request['cells'];keys={f'{o}-{c}' for o in fr['object_ids'] for c in fr['conditions']}
    require(keys==set(cells) and fr['expected_scores']==len(keys)*24, 'all registered cells required before opening')
    ready=0; terminal=[]
    for obj in fr['object_ids']:
        for cond in fr['conditions']:
            cell=cells[f'{obj}-{cond}']
            if cell['status']=='scored':
                ready+=scored_cell(roots,obj,cond,cell,fp,fr)
            else:
                require(cell['status'] in ('failed','not_executed') and bool(cell.get('reason')), 'missing terminal disposition')
                _, f=read_pinned(roots,cell['disposition']); d=load(f)
                require(d['schema']=='el-t0-cell-disposition-1' and (d['object_id'],d['condition'],d['freeze'],d['status'],d['reason'])==(obj,cond,fp,cell['status'],cell['reason']) and d['target_reference_reads']==0, 'missing frozen failure or nonexecution receipt')
                require(d['causal_receipt'] in config['sources'].values(), 'missing immutable causal evidence')
            terminal.append({'object_id':obj,'condition':cond,**cell})
    authority={'schema':'el-t0-reference-opening-1','run_id':os.environ['EL_RUN_ID'],'utc':utc(),
               'freeze':fp,'roster':fr['sources']['roster'],'object_ids':fr['object_ids'],'conditions':fr['conditions'],
               'target_ids':list(VIEWS),'scored_entries':ready,'scheduled_entries':fr['expected_scores'],
               'terminal_cells':terminal,'roster_sha256':fr['roster_sha256'],'predictions_and_risks_sealed':True,'sealed_matrix_sha256':sha(request_file),
               'purpose':'one independent evaluation of the complete frozen T0 matrix','permitted_reader':'independent_evaluator'}
    save(out/'opening.json',authority)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('freeze');b=sub.add_parser('open')
    for pp in (a,b):
        pp.add_argument('--config',type=Path,required=True);pp.add_argument('--out',type=Path,required=True)
    for n in ('roster','baseline','release','protocol'):a.add_argument('--'+n,type=Path,required=True)
    for n in ('store','history','request'):b.add_argument('--'+n,type=Path,required=True)
    x=p.parse_args();cfg=load(x.config)
    if x.action=='freeze':freeze(cfg,x.roster,x.baseline,x.release,x.protocol,x.out)
    else:opening(cfg,(x.store,x.history),x.request,x.out)
