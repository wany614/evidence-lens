"""Independent T0 evaluation after the collective opening, using the released D1 metrics."""
import argparse
import csv
import hashlib
import html
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from dtu_eval import photo, read_pinned, score_map
from dtu_eval_board import board
from dtu_eval_metrics import labels, observation
from dtu_score import METHODS, VIEWS
from evidence_lens._bundle import load, save, utc
from evidence_lens._verify import require
from evidence_lens.multisource_io import Case
from recovery_eval_coverage import PERCENTS, counts_at, ranks
from recovery_eval_risk import coverage_totals, reasons

COUNTS = ('V','E','Y','S','SY','C','CY','L','LY','H','HY','U','UY','V_minus_E','Y3','rgb_mae_sum','aux3_mae_sum')
RATIOS = {'MAE':('rgb_mae_sum','E'),'error_rate':('Y','E'),'scoreable':('S','E'), 'coverage':('L','V'),'low_error':('LY','L'),'unknown':('U','E')}


def table(dest, name, rows):
    cols = list(dict.fromkeys(k for row in rows for k in row))
    with (dest/name).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f,fieldnames=cols,lineterminator='\n');writer.writeheader();writer.writerows(rows)


def ratio(n, d):
    return n / d if d is not None and d > 0 else None


def mask_id(mask):
    return hashlib.sha256(np.asarray(mask,dtype='uint8',order='C').tobytes()).hexdigest()


def population(v, e, y, scores, thresholds, object_id, condition, view):
    common = e & np.logical_and.reduce([np.isfinite(scores[m]) for m in METHODS])
    rank = ranks(v.shape,object_id,condition,view)
    n = int(v.sum())
    quotas = [(f'{q}%_V',(2*q*n+99)//200) for q in PERCENTS]
    quotas.append(('new_frozen_MS-L1',int((common & (scores['MS-L1'].astype(np.float64)<thresholds['MS-L1'])).sum())))
    base = {'object_id':object_id,'condition':condition,'view':view,'domain':'within_new','common_mask_sha256':mask_id(common)}
    return common, [{**base,'method':m,**r} for m in METHODS for r in counts_at(scores[m][common],y[common],rank[common],n,quotas)]


def reduce_rows(rows):
    metrics = {k:sum(r[k] for r in rows if r[k] is not None) if any(r[k] is not None for r in rows) else None for k in COUNTS}
    metrics.update({key:ratio(metrics[a],metrics[b]) for key,(a,b) in RATIOS.items()})
    for domain in ('own','common'):
        defined=[r for r in rows if r[domain+'_AP'] is not None]
        metrics[domain+'_AP_defined_views']=len(defined)
        for what in ('_AP','_prevalence'):
            metrics[domain+what]=mean(r[domain+what] for r in defined) if defined else None
    return metrics


def evaluate(roots, opening_pin, refs, out):
    _, p = read_pinned(roots,opening_pin); opened=load(p)
    require(opened['schema']=='el-t0-reference-opening-1' and opened['permitted_reader']=='independent_evaluator' and set(refs)==set(opened['object_ids']), 'independent collective opening and every object required')
    _, p=read_pinned(roots,opened['freeze']); frozen=load(p)
    require(opened['object_ids']==frozen['object_ids'] and opened['scheduled_entries']==frozen['expected_scores'], 'changed roster')
    _, rp=read_pinned(roots,frozen['sources']['roster']); roster={r['object_id']:r for r in load(rp)['objects']}
    tau={m:spec['threshold'] for m,spec in frozen['models'].items()}
    all_views=[]; all_matched=[]; causes=[]; source_fit=[]; system=[]; object_rows=[]; matched_conditions=[]; boards=[]; accessed=[]
    public=out/'public';private=out/'private';public.mkdir();private.mkdir()
    access_time=utc()
    for obj in opened['object_ids']:
        _, path=read_pinned(roots,refs[obj]); refbook=load(path)
        require(refbook['schema']=='el-t0-dtu-references-1' and refbook['object_id']==obj and refbook['opening']==opening_pin and set(refbook['views'])==set(VIEWS),'targets not bound to collective reference opening')
        reference={}
        for v in VIEWS:
            row=refbook['views'][v]
            require(row['official_view_id']==frozen['protocol']['camera']['targets'][v], 'target camera differs from freeze')
            expected=roster[obj]['target_members'][v]
            require(row['archive_member']==(expected if isinstance(expected,str) else expected['name']), 'reference member differs from roster')
            reference[v]=photo(path.parent,row['rgb']);accessed.append({'object_id':obj,'view':v,'official_view_id':row['official_view_id'],'cropped_sha256':row['rgb']['sha256'],'original_sha256':row['original_sha256']})
        for cond in opened['conditions']:
            cell,= [c for c in opened['terminal_cells'] if (c['object_id'],c['condition'])==(obj,cond)]
            case=None; scores=None; source_images={}
            if cell['status']=='scored':
                _,cp=read_pinned(roots,cell['case']);case=Case(cp.parent,cell['case']['sha256'])
                _,ss=read_pinned(roots,cell['scores']);scores=load(ss)
                for s,(im,frame,_) in case.sources('all').items():
                    source_images[s]=im
                    sv,se,sy,serror,_=labels(im,frame['rgb'] if frame else None,np.ones(im.shape[:2],bool))
                    source_fit.append({'object_id':obj,'condition':cond,'view':s,'V':int(sv.sum()),'E':int(se.sum()),'Y':int((sy==1).sum()),'source_MAE':ratio(float(serror[se].sum(dtype=np.float64)),int(se.sum()))})
            per_method=defaultdict(list); coverage=defaultdict(list)
            for v in VIEWS:
                target=case.frame(v) if case else None
                ready=case is not None and scores['views'][v]['status']=='ready' and target is not None
                rgb=target['rgb'] if target else None
                valid,e,y,err,aux=labels(reference[v],rgb,np.ones((256,256),bool))
                risks={m:score_map(ss.parent,scores,m,v,case,tau[m]) if ready else np.full(valid.shape,np.nan,dtype=np.float32) for m in METHODS}
                results,_=observation(valid,e,y,err,aux,risks,tau)
                common, matched=population(valid,e,y,risks,tau,obj,cond,v)
                for row in matched: coverage[(row['method'],row['anchor'])].append(row)
                all_matched.extend(matched)
                if ready:
                    why,_,_=reasons(case,v,target,risks,e,y)
                    for m,w in why.items():
                        for reason,counts in w.items():causes.append({'object_id':obj,'condition':cond,'view':v,'method':m,'reason':reason,**counts})
                for m,r in results.items():
                    meta={'object_id':obj,'condition':cond,'view':v,'method':m,'status':'scored' if ready else cell.get('reason') or 'render_failed',
                          'threshold':tau[m],'V_mask_sha256':mask_id(valid),'E_mask_sha256':mask_id(e),'common_mask_sha256':mask_id(common),'S_mask_sha256':mask_id(e&np.isfinite(risks[m])),
                          'label_sha256':mask_id(y),'reference_sha256':refbook['views'][v]['rgb']['sha256'],'prediction_sha256':case.meta['renders'][v]['sha256'] if ready else None}
                    item={**meta,**{k:r['counts'][k] for k in COUNTS}}
                    for domain in ('own','common'):
                        rankspec=r[domain];item[domain+'_AP']=rankspec['ap'];item[domain+'_prevalence']=ratio(rankspec['positive'],rankspec['n']);item[domain+'_AP_reason']=rankspec['missing_reason']
                    per_method[m].append(item);all_views.append(item)
                name=f'{obj}-{cond}-{v}.png'; board(private/name,obj,cond,v,source_images,rgb,reference[v],(valid,e,y,err,aux),risks,tau,target['alpha'] if target else None);boards.append(name)
            for m,rows in per_method.items():
                object_rows.append({'object_id':obj,'physical_family_id':frozen['families'][obj],'condition':cond,'method':m,'status':cell['status'],'views_scheduled':len(VIEWS),'views_scored':sum(r['status']=='scored' for r in rows),'threshold':tau[m],**reduce_rows(rows)})
            for (m,anchor),rows in coverage.items():matched_conditions.append({'object_id':obj,'condition':cond,'method':m,'domain':'within_new','anchor':anchor,**coverage_totals(rows)})
            system.append({'object_id':obj,'condition':cond,'status':cell['status'],'scheduled_entries':len(VIEWS)*len(METHODS),'scored_entries':sum(r['status']=='scored' for rows in per_method.values() for r in rows),'reason':cell.get('reason')})
    require(len(all_views)==opened['scheduled_entries'] and sum(s['scored_entries'] for s in system)==opened['scored_entries'], 'closed-matrix and evaluated totals differ')
    equal=[]
    for cond in opened['conditions']:
        for m in METHODS:
            items=[r for r in object_rows if r['condition']==cond and r['method']==m]
            metrics=['MAE','error_rate','coverage','low_error','unknown','common_AP','common_prevalence','own_AP','own_prevalence']
            for key in metrics:
                values=[r[key] for r in items if r[key] is not None];equal.append({'condition':cond,'method':m,'metric':key,'objects_scheduled':len(items),'objects_defined':len(values),'object_equal_mean':mean(values) if values else None,'minimum':min(values,default=None),'maximum':max(values,default=None)})
            for anchor in [f'{q}%_V' for q in PERCENTS]+['new_frozen_MS-L1']:
                rows=[r for r in matched_conditions if r['condition']==cond and r['method']==m and r['anchor']==anchor];values=[r['risk'] for r in rows if r['risk'] is not None]
                equal.append({'condition':cond,'method':m,'metric':'within_new_risk_'+anchor,'objects_scheduled':len(items),'objects_defined':len(values),'object_equal_mean':mean(values) if values else None,'minimum':min(values,default=None),'maximum':max(values,default=None)})
    for name,rows in [('method-views.csv',all_views),('method-conditions.csv',object_rows),('matched-views.csv',all_matched),('matched-conditions.csv',matched_conditions),('object-equal.csv',equal),('source-fit.csv',source_fit),('unknown-reasons.csv',causes),('system.csv',system)]:table(public,name,rows)
    (private/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Private DTU T0 panels</title><h1>Private DTU T0 evaluation</h1><ul>'+''.join(f'<li><a href="{html.escape(n)}">{html.escape(n[:-4])}</a></li>' for n in boards)+'</ul>')
    save(public/'evaluation.json',{'schema':'el-t0-evaluation-1','run_id':os.environ['EL_RUN_ID'],'opening':opening_pin,'references':refs,'target_reference_read_utc':access_time,'source_role':'independent evaluator','accessed_references':accessed,'scheduled_entries':len(all_views),'scored_entries':opened['scored_entries'],'objects':opened['object_ids'],'object_mask':'N/A: no verified independent image-space mask','method_weighting':'six scheduled views per object and condition; AP and companion prevalence average over that same set of defined views; object means assign equal weight to each defined physical family','risk_training':0,'threshold_fits':0,'NBV_experiments':0})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('store','history','request','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();req=load(a.request);evaluate((a.store,a.history),req['opening'],req['references'],a.out)
