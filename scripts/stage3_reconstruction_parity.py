"""Compare raw-native joint reconstruction measurements with the independently built target-label dataset."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from evidence_lens._bundle import encoded,load,sha
from evidence_lens._verify import require
from make_handoff import publish
from stage3_reference import located,perform

TARGETS=tuple(f'v{i:02d}' for i in range(1,7))


def verify_comparison(original,label,e,z):
    result={'V':int(label['reference_valid'].sum()),'surface':int(label['reference_surface'].sum()),'E':int(e.sum()),'Y':int((label['rgb_high_error'][e]==1).sum()),
            'Z':int(z.sum()),'ZY':int((label['depth_high_error'][z]==1).sum()),'invalid_rgb_V':int((label['reference_valid']&~label['rgb_evaluable']).sum()),
            'low_support_surface':int(label['low_support'].sum()),'unsupported_depth_surface':int((label['reference_surface']&~label['depth_evaluable']).sum()),
            'alpha_valid_surface':int((label['reference_surface']&label['alpha_valid']).sum()),'rgb_abs_sum':float(label['rgb_error'][e].sum(dtype=np.float64)),
            'z_abs_sum':float(label['depth_error'][z].sum(dtype=np.float64)),'z_signed_sum':float(label['signed_depth_error'][z].sum(dtype=np.float64))}
    for k in result:require(original[k]==result[k],'Signed raw/source label counts or error sums differ for '+k)
    return len(result)


def work(a):
    request=load(a.request);data=load(a.dataset);pairs=load(a.pairs);study=load(a.ledger)
    require(sha(a.dataset)==request['dataset_sha256'] and sha(a.ledger)==request['ledger_sha256'] and sha(a.pairs)==study['tables']['family-pairs.json'],
            'Source data and independent original comparison must be pinned')
    require(data['split']==request['split'] and data['catalog_sha256']==study['catalog_sha256']==request['catalog_sha256'] and
            data['opening_sha256']==request['opening_sha256'] and (data['split']!='test' or study['opening_sha256']==data['opening_sha256'] and not study['still_sealed']),
            'Wrong cohort or method opening')
    refs={}
    for row,path in zip(data['views'],a.truth,strict=True):
        k=(row['family'],row['condition'],row['view']);meta=row['truth'];require(k not in refs and sha(path)==meta['sha256'] and path.stat().st_size==meta['bytes'],
                'Duplicated or changed archived labels');refs[k]=(row,path)
    original_cells={(c['family'],c['condition']):c for c in data['cells']}
    targets=[];family_checks={};counts=0
    for f in data['families']:
        family=pairs[f];checks={};n=0
        for v in TARGETS:
            src=next(r for r in family['views'] if r['target']==v);arr={}
            for c in ('M2','M4'):
                r,p=refs[f,c,v];cell=original_cells[f,c];orig=src['origins'][c]
                require(orig['reference']==cell['selected']['reference'] and orig['render']==cell['selected'].get('render') and r['selection_sha256']==cell['selection_sha256'] and
                        r['reference_member_sha256']==orig['reference_npz_sha256'] and r['prediction_member_sha256']==orig['render_npz_sha256'],
                        'Raw-native and risk-dataset origins differ')
                with np.load(p,allow_pickle=False) as archive:
                    arr[c]={k:archive[k] for k in ('reference_valid','reference_surface','rgb_evaluable','depth_evaluable','rgb_high_error','depth_high_error','rgb_error',
                                                   'depth_error','signed_depth_error','low_support','alpha_valid')}
            e=arr['M2']['rgb_evaluable']&arr['M4']['rgb_evaluable'];z=arr['M2']['depth_evaluable']&arr['M4']['depth_evaluable']
            for c in ('M2','M4'):
                lab=arr[c];n+=verify_comparison(src['own'][c],lab,lab['rgb_evaluable'],lab['depth_evaluable']);n+=verify_comparison(src['paired'][c],lab,e,z)
            targets.append({'family':f,'view':v,'truth_sha256':{c:refs[f,c,v][0]['truth']['sha256'] for c in ('M2','M4')},'passed':True})
        family_checks[f]={'target_pairs':6,'cell_and_common_count_sum_checks':n};counts+=n
    proof={'schema':'el-stage3-reconstruction-parity-1','run_id':os.environ['EL_RUN_ID'],'passed':True,'split':request['split'],'families':family_checks,
           'dataset':request['dataset'],'source_ledger':request['ledger'],'opening_sha256':request['opening_sha256'],'views':targets,'validated_view_pairs':len(targets),
           'distinct_source_labels':len(refs),'cell_and_common_count_sum_checks':counts,
           'quantities':'Across each M2/M4 pair, own and common RGB/ED counts, validity, event counts and original float64 summed pixel errors are exactly equal.'}
    (a.out/'parity.json').write_bytes(encoded(proof))


def record(a):
    store,project,assets=a.store.resolve(),a.project.resolve(),a.assets.resolve();cat=sha(assets/'protocol/stage3-families-v1.json');area=store/'stage3'/cat/'acquisition-ledger'
    orig={}
    if a.split=='test':
        from evidence_lens.stage3_gate import require_opening
        require(a.opening_sha256,'Explicit opening required');require_opening(store,cat,opening_sha256=a.opening_sha256,shelf=store.parent/'cold-shelf')
    roles={};parents={}
    for name,pair,path in [('dataset',a.dataset_run,'dataset.json'),('ledger',a.ledger_run,'ledger.json')]:
        rid,h=pair.split('=');pin={'run_id':rid,'manifest_sha256':h};folder=located(store,pin)/'payload';p=folder/path;roles[name]=(p,name);parents[name]=pin;orig[name]=(folder,load(p),sha(p))
    ds,d,dsha=orig['dataset'];root,lg,lsha=orig['ledger'];require(d['split']==a.split and d['catalog_sha256']==lg['catalog_sha256']==cat,'Wrong actual dataset')
    require(len(d['views'])==len(d['families'])*12 and dsha==a.dataset_sha256 and lsha==a.ledger_sha256,'Original split population/receipt changed')
    marker=area/(a.split+'-risk-dataset-parity.json')
    if marker.exists():
        saved=load(marker);require(saved['dataset']==parents['dataset'] and saved['ledger']==parents['ledger'],'Existing parity uses another ledger/dataset');return saved
    roles['pairs']=(root/'family-pairs.json','ledger');cmd=['{python}','{source}/scripts/stage3_reconstruction_parity.py','work','--out','{out}','--request','{input:request}',
         '--dataset','{input:dataset}','--ledger','{input:ledger}','--pairs','{input:pairs}']
    for i,row in enumerate(d['views']):
        label=ds/row['truth']['path'];require(sha(label)==row['truth']['sha256'],'Original native dataset target changed');tag='t'+str(i);roles[tag]=(label,'dataset');cmd+=['--truth','{input:'+tag+'}']
    with tempfile.TemporaryDirectory(dir=store/'work',prefix='stage3-recon-parity-') as tmp:
        tmp=Path(tmp);src=tmp/'engine';(src/'scripts').mkdir(parents=True)
        for n in ('stage3_reconstruction_parity.py','stage3_reference.py'):shutil.copyfile(assets/'scripts'/n,src/'scripts'/n)
        for n in ('make_handoff.py','formal_lineage.py'):shutil.copyfile(project/'scripts'/n,src/'scripts'/n)
        spec={'split':a.split,'catalog_sha256':cat,'opening_sha256':a.opening_sha256,'ledger':parents['ledger'],'dataset':parents['dataset'],'ledger_sha256':lsha,'dataset_sha256':dsha}
        req=tmp/'request.json';req.write_bytes(encoded(spec));roles['request']=(req,None)
        cfg={'experiment_id':'stage3-reconstruction-label-parity','trial_id':'raw-versus-risk-'+a.split,'split':a.split,'dataset_hash':cat,
             'source_selector_sha256':sha(req),'recording':{'expected_output_bytes':90000}}
        run=perform(store,project,src,cfg,'archive_verify',roles,parents,cmd)
    proof=located(store,run)/'payload/parity.json';data=load(proof);require(data['passed'],'Original scientific cross-check failed');receipt={'schema':'el-stage3-reconstruction-parity-selection-1',
             'run':run,'parity_sha256':sha(proof),'dataset':parents['dataset'],'ledger':parents['ledger'],'split':a.split,'views':data['distinct_source_labels'],'checks':data['cell_and_common_count_sum_checks']};publish(marker,encoded(receipt));return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('work','record'));p.add_argument('--split',choices=('train','development','test'))
    for n in ('assets','project','store','out','request','dataset','ledger','pairs'):p.add_argument('--'+n,type=Path)
    for n in ('dataset-run','ledger-run','dataset-sha256','ledger-sha256','opening-sha256'):p.add_argument('--'+n)
    p.add_argument('--truth',action='append',default=[],type=Path);a=p.parse_args()
    if a.action=='work':work(a)
    else:print(json.dumps(record(a),sort_keys=True))
