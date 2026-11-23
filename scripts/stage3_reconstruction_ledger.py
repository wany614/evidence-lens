"""Read the preregistered acquisition matrix, align paired raw reconstructions, and append immutable condition receipts."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from evidence_lens import labels,scoring_io
from evidence_lens._bundle import encoded,load,sha
from evidence_lens._verify import require,verify_bundle
from make_handoff import publish
from stage3_reference import located,perform

TARGETS=tuple(f'v{i:02d}' for i in range(1,7))
SOURCES={'M2':('s00','s03'),'M4':('s00','s01','s02','s03')}
FIELDS=('V','surface','E','Y','Z','ZY','invalid_rgb_V','low_support_surface','unsupported_depth_surface','alpha_valid_surface','rgb_abs_sum','z_abs_sum','z_signed_sum')


def counts(lab,e,z):
    rgb_ok=lab['reference_valid'];surface=lab['reference_surface']
    return {'V':int(rgb_ok.sum()),'surface':int(surface.sum()),'E':int(e.sum()),'Y':int((lab['rgb_high_error'][e]==1).sum()),
            'Z':int(z.sum()),'ZY':int((lab['depth_high_error'][z]==1).sum()),
            'invalid_rgb_V':int((rgb_ok&~lab['rgb_evaluable']).sum()),'low_support_surface':int(lab['low_support'].sum()),
            'unsupported_depth_surface':int((surface&~lab['depth_evaluable']).sum()),'alpha_valid_surface':int((surface&lab['alpha_valid']).sum()),
            'rgb_abs_sum':float(lab['rgb_error'][e].sum(dtype=np.float64)),'z_abs_sum':float(lab['depth_error'][z].sum(dtype=np.float64)),
            'z_signed_sum':float(lab['signed_depth_error'][z].sum(dtype=np.float64))}


def rates(val):
    out={**val}
    for name,num,den in [('rgb_mae','rgb_abs_sum','E'),('rgb_event_rate','Y','E'),('rgb_invalid_fraction','invalid_rgb_V','V'),
                         ('depth_mae_m','z_abs_sum','Z'),('depth_event_rate','ZY','Z'),('depth_signed_bias_m','z_signed_sum','Z'),
                         ('low_support_surface_fraction','low_support_surface','surface')]:
        out[name]=val[num]/val[den] if val[den] else None
    return out


def family_quality(store,cells):
    expected={c:cells[c]['selection'] for c in SOURCES}
    refpin=expected['M2']['reference'];require(refpin==expected['M4']['reference'],'Pair does not share its original native world')
    ref=located(store,refpin)/'payload';info=load(ref/'reference.json');render={};predmeta={}
    for c,selected in expected.items():
        if 'render' in selected:
            rr=located(store,selected['render'])/'payload';mr=load(rr/'prediction.json')
            require(mr['depth_verified'] and mr['reconstruction_run_id']==selected['reconstruction']['run_id'] and mr['run_id']==selected['render']['run_id'],
                    'A render is not from the selected joint checkpoint')
            render[c]=rr;predmeta[c]=mr
    targets=[]
    for target in TARGETS:
        fp=ref/'reference'/(target+'.npz');h=info['views'][target]['npz']['sha256'];require(sha(fp)==h,'Private target reference changed')
        with np.load(fp,allow_pickle=False) as ar:
            truth={k:ar[k] for k in ('rgb','z','position','edge',*('visibility_'+s for s in SOURCES['M4']))}
        arrays={};own={};provenance={}
        for c,ids in SOURCES.items():
            pred=None;digital=None
            if c in render:
                file=render[c]/(target+'.npz');digital=predmeta[c]['views'][target]['sha256'];require(sha(file)==digital,'Original numerical prediction changed')
                with np.load(file,allow_pickle=False) as ar:pred={k:ar[k] for k in ('rgb','alpha','z')}
            vis=np.stack([truth['visibility_'+s] for s in ids]);union=np.select([(vis==i).any(0) for i in (1,2,3)],[1,2,3],0).astype(np.uint8)
            lab=labels.label_view({**truth,'visibility':union},pred,pred is not None);arrays[c]=lab
            own[c]=rates(counts(lab,lab['rgb_evaluable'],lab['depth_evaluable']))
            provenance[c]={'reference_npz_sha256':h,'render_npz_sha256':digital,'reference':refpin,'render':expected[c].get('render')}
        e=arrays['M2']['rgb_evaluable']&arrays['M4']['rgb_evaluable'];z=arrays['M2']['depth_evaluable']&arrays['M4']['depth_evaluable']
        paired={c:rates(counts(arrays[c],e,z)) for c in SOURCES}
        targets.append({'target':target,'origins':provenance,'own':own,'paired':paired})
    document={'views':targets}
    for domain in ('own','paired'):
        document[domain]={c:rates({k:sum(v[domain][c][k] for v in targets) for k in FIELDS}) for c in SOURCES}
    document['M4_minus_M2']={}
    for k in ('E','Z','rgb_mae','rgb_event_rate','depth_mae_m','depth_event_rate','low_support_surface_fraction'):
        m2,m4=(document['paired'][c][k] for c in SOURCES)
        document['M4_minus_M2'][k]=m4-m2 if m2 is not None and m4 is not None else None
    document['M4_minus_M2_own_counts']={k:(document['own']['M4'][k]-document['own']['M2'][k]) for k in ('E','Z','Y','ZY')}
    return document


def planned(catalog,registration,budget):
    rows=[]
    for f in catalog['families']:
        for c,ids in SOURCES.items():
            t=f['trials'][c];rows.append({'trial_id':t,'family':f['scene_family_id'],'split':f['split'],'condition':c,'source_ids':list(ids),'target_ids':list(TARGETS),
                                         'planned_views':len(ids)+6,'point_count':4096,'updates':256,'source_pixels':256*256**2,'actual_receipt':'cells/'+t+'.json'})
    require(len(rows)==len({r['trial_id'] for r in rows})==32 and sum(r['planned_views'] for r in rows)==288,'Registered joint matrix changed')
    return {'schema':'el-stage3-acquisition-roster-1','origin':'Projection of the original family preregistration and the preproduction 256-step budget.',
            'reference_registration':registration,'production_budget':budget,'rows':rows}


def work(a):
    spec=load(a.request);doc=load(a.cost);catalog=load(a.catalog)
    require(spec['catalog_sha256']==sha(a.catalog)==doc['catalog_sha256'] and doc['scope']==spec['scope'] and sha(a.cost)==spec['costs_sha256'],
            'Acquisition source/cost digest mismatch')
    for module in (labels,scoring_io):
        require(sha(Path(module.__file__))==spec['label_source_sha256'][module.__name__],'Frozen error semantics implementation differs')
    if spec['scope']=='full':
        from evidence_lens.stage3_gate import require_opening
        require_opening(a.store,spec['catalog_sha256'],opening_sha256=spec['opening_sha256'],shelf=a.store.parent/'cold-shelf')
    lookup={r['trial_id']:r for r in doc['trials'] if r['phase']=='production'};engineering=[r for r in doc['trials'] if r['phase']!='production']
    regions={};actual=[];pending=[];table=[];per_view=[]
    cols=('family','split','condition','trial_id','status','planned_views','complete_views','V','E','rgb_mae','Y','rgb_event_rate','Z','depth_mae_m','ZY','depth_event_rate',
          'low_support_surface','paired_E','paired_rgb_mae','paired_rgb_event_rate','paired_Z','paired_depth_mae_m','m4_minus_m2_paired_rgb_mae','m4_minus_m2_paired_rgb_event_rate',
          'm4_minus_m2_paired_depth_mae_m','actual_updates','source_pixels','initial_count','final_count','unchanged_full_rows','optimization_seconds',
          'generation_wall_seconds','render_wall_seconds','return_zip_bytes_all','failed_attempts')
    reg=planned(catalog,spec['reference_registration'],spec['production_budget']);(a.out/'roster.json').write_bytes(encoded(reg))
    for family in catalog['families']:
        f=family['scene_family_id'];split=family['split'];pairs={c:lookup.get(family['trials'][c]) for c in SOURCES}
        if split=='test' and spec['scope']!='full':
            require(not any(x is not None for x in pairs.values()),'Preopening source attempted to supply test data')
            for c,row in pairs.items():
                planned_row=next(y for y in reg['rows'] if y['trial_id']==family['trials'][c]);pending.append(planned_row['trial_id']);table.append({**planned_row,'status':'sealed_at_this_checkpoint'})
            continue
        require(all(x is not None and x['split']==split and x['family']==f for x in pairs.values()),'A scheduled pair is absent')
        quality=family_quality(a.store,pairs);regions[f]=quality
        for c,r in pairs.items():
            t=family['trials'][c];own=quality['own'][c];paired=quality['paired'][c];diff=quality['M4_minus_M2'];item={**r,'own':own,'common_pair':paired,'M4_minus_M2':diff,
                'target_receipts':[{'target':v['target'],**v['origins'][c],'own':v['own'][c],'common_pair':v['paired'][c]} for v in quality['views']]}
            actual.append(t);where=a.out/'cells'/(t+'.json');where.parent.mkdir(exist_ok=True);where.write_bytes(encoded(item))
            table.append({**r,'status':r['selection_status'],**own,'paired_E':paired['E'],'paired_Z':paired['Z'],'paired_rgb_mae':paired['rgb_mae'],
                         'paired_rgb_event_rate':paired['rgb_event_rate'],'paired_depth_mae_m':paired['depth_mae_m'],'m4_minus_m2_paired_rgb_mae':diff['rgb_mae'],
                         'm4_minus_m2_paired_rgb_event_rate':diff['rgb_event_rate'],'m4_minus_m2_paired_depth_mae_m':diff['depth_mae_m']})
            for v in quality['views']:per_view.append({'family':f,'split':split,'condition':c,'trial_id':t,'view':v['target'],**v['own'][c],
                                                      'paired_E':v['paired'][c]['E'],'paired_Z':v['paired'][c]['Z'],**v['origins'][c]})
    with (a.out/'scientific-matrix.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,cols,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(table)
    with (a.out/'target-matrix.csv').open('w',encoding='utf-8',newline='') as stream:
        keys=('family','split','condition','trial_id','view','reference_npz_sha256','render_npz_sha256','V','E','Y','rgb_mae','rgb_event_rate','surface','Z','ZY','depth_mae_m','depth_event_rate',
              'low_support_surface','unsupported_depth_surface','paired_E','paired_Z');writer=csv.DictWriter(stream,keys,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(per_view)
    (a.out/'family-pairs.json').write_bytes(encoded(regions))
    ops=load(a.operations);require(sha(a.operations)==spec['operations_sha256'],'Engineering case selection differs')
    for event in ops['events']:
        if 'run' in event:
            pin=event['run'];where=a.store/event['original_tree']/pin['run_id'];verified=verify_bundle(where);require(verified['manifest_sha256']==pin['manifest_sha256'] and
                     verified['run']['execution_status']=='failed','Engineering failure origin differs')
        else:
            original=a.store/event['original_member'];require(sha(original)==event['sha256'],'Engineering exception differs')
    (a.out/'engineering.json').write_bytes(encoded({'schema':'el-stage3-acquisition-engineering-1','pilot':[r for r in engineering if r['phase']=='engineering-pilot'],
                              'same_input_repeat':[r for r in engineering if r['phase']=='engineering-repeat'],'operational_exceptions':ops['events'],
                              'scientific_failed_attempts':[x for x in doc['attempts'] if x['status']!='succeeded']}))
    main={'schema':'el-stage3-acquisition-ledger-1','run_id':os.environ['EL_RUN_ID'],'scope':spec['scope'],'catalog_sha256':spec['catalog_sha256'],'cost':spec['cost_pin'],
          'source_costs_sha256':sha(a.cost),'opening_sha256':spec['opening_sha256'],'planned_trials':len(reg['rows']),'planned_views':sum(x['planned_views'] for x in reg['rows']),
          'with_actual_records':actual,'still_sealed':pending,'selection_statuses':{t:lookup[t]['selection_status'] for t in actual},'quality_target_views':len(per_view),
          'tables':{x.name:sha(x) for x in a.out.iterdir() if x.is_file()},'scientific_note':'World geometry and target RGB enter only this retrospective evaluation. Signed errors are M4 minus M2 on their common evaluable pixels. Depth comparison is auxiliary expected ED versus Blender first-hit Z, with alpha at least 0.5.'}
    (a.out/'ledger.json').write_bytes(encoded(main))


def record(a):
    store,project,assets=a.store.resolve(),a.project.resolve(),a.assets.resolve();cf=assets/'protocol/stage3-families-v1.json';cat=sha(cf);base=store/'stage3'/cat;area=base/'acquisition-ledger'
    require(sha(a.operations)==a.operations_sha256,'The explicit engineering exception inventory changed')
    costpin=dict(zip(('run_id','manifest_sha256'),a.cost_run.split('='),strict=True));cost=located(store,costpin)/'payload/costs.json';c=load(cost)
    require(sha(cost)==a.cost_sha256 and a.scope==c['scope'] and cat==c['catalog_sha256'],'A different pinned cost cohort was supplied')
    marker=area/(a.scope+'.json');reg=load(base/'registration.json')['registration'];budget=load(base/'production-budget.json')['budget'];parents={'cost':costpin,'registration':reg,'budget':budget}
    require(sha(located(store,reg)/'payload/catalog.json')==cat,'Preregistered scientific roster differs')
    register=planned(load(cf),reg,budget);publish(area/'roster.json',encoded(register))
    if a.scope=='full':
        from evidence_lens.stage3_gate import require_opening
        require(a.opening_sha256,'Real test opening required');parents['method']=require_opening(store,cat,opening_sha256=a.opening_sha256,shelf=store.parent/'cold-shelf')['method']
    if marker.exists():
        original=load(marker);out=located(store,original['run'])/'payload';require(original['cost']==costpin and original['operations_sha256']==a.operations_sha256 and
                sha(out/'ledger.json')==original['ledger_sha256'],'Inconsistent scientific ledger pointer');return original
    with tempfile.TemporaryDirectory(dir=store/'work',prefix='stage3-quality-ledger-') as tmp:
        tmp=Path(tmp);src=tmp/'engine';(src/'scripts').mkdir(parents=True);(src/'analysis').mkdir()
        for file in ('stage3_reconstruction_ledger.py','stage3_reference.py'):shutil.copyfile(assets/'scripts'/file,src/'scripts'/file)
        for file in ('make_handoff.py','formal_lineage.py'):shutil.copyfile(project/'scripts'/file,src/'scripts'/file)
        label_sources={}
        for module in (labels,scoring_io):
            native=Path(module.__file__);shutil.copyfile(native,src/'analysis'/native.name);label_sources[module.__name__]=sha(native)
        spec={'catalog_sha256':cat,'scope':a.scope,'opening_sha256':a.opening_sha256,'cost_pin':costpin,'costs_sha256':a.cost_sha256,'operations_sha256':a.operations_sha256,
              'reference_registration':reg,'production_budget':budget,'label_source_sha256':label_sources};p=tmp/'request.json';p.write_bytes(encoded(spec))
        for r in c['trials']:
            if r['phase']!='production':continue
            for key in ('reference','render'):
                if key in r['selection']:pin=r['selection'][key];parents[pin['run_id']]=pin
        roles={'cost':(cost,'cost'),'catalog':(cf,'registration'),'request':(p,None),'operations':(a.operations,None)}
        cfg={'experiment_id':'stage3-original-reconstruction-ledger','trial_id':'joint-quality-'+a.scope,'dataset_hash':cat,'split':'mixed','source_measurements_sha256':a.cost_sha256,
             'recording':{'expected_output_bytes':3200000}}
        argv=['{python}','{source}/scripts/stage3_reconstruction_ledger.py','work','--request','{input:request}','--catalog','{input:catalog}','--cost','{input:cost}',
              '--operations','{input:operations}','--store',str(store),'--out','{out}'];pin=perform(store,project,src,cfg,'report',roles,parents,argv)
    out=located(store,pin)/'payload';value=load(out/'ledger.json');require(sha(area/'roster.json')==sha(out/'roster.json'),'Unchanged planned roster required')
    for tid in value['with_actual_records']:
        data=out/'cells'/(tid+'.json');receipt=area/'cells'/(tid+'.json')
        if receipt.exists():
            existing=load(receipt);require(existing['source_sha256']==sha(data),'Previously sealed reconstruction result changed');continue
        publish(receipt,encoded({'schema':'el-stage3-acquisition-cell-1','trial_id':tid,'run':pin,'member':'payload/cells/'+tid+'.json','source_sha256':sha(data)}))
    result={'schema':'el-stage3-acquisition-ledger-selection-1','scope':a.scope,'run':pin,'ledger_sha256':sha(out/'ledger.json'),'cost':costpin,'opening_sha256':a.opening_sha256,
            'operations_sha256':a.operations_sha256,'roster_sha256':sha(area/'roster.json'),'planned':len(register['rows']),'actual':len(value['with_actual_records']),
            'sealed':len(value['still_sealed']),'tables':value['tables']};publish(marker,encoded(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('work','record'));p.add_argument('--scope',choices=('development','full'))
    for n in ('assets','store','project','request','catalog','cost','operations','out'):p.add_argument('--'+n,type=Path)
    for n in ('cost-run','cost-sha256','operations-sha256','opening-sha256'):p.add_argument('--'+n)
    a=p.parse_args()
    if a.action=='work':work(a)
    else:print(json.dumps(record(a),sort_keys=True))
