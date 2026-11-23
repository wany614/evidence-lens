"""Assemble and verify the registered 512-pixel public-plane attachment from named original records."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import vision_release as rel
from vision_public_smoke import CONTROL_INPUTS, METHODS, VIEWS, control_inputs, control_result

FAMILY = 'scene_037_bottle_carrier'
CONDITION = 'public-plane'
CASE_PIN = rel.pin('20261123T134307201400Z-a5e16b2b6c7f4dd2=ecc31a41cb0db4ae0319ac6f5057bf9f0326a776cb641a2a9b961172fa0c79e8')
REFERENCE_PIN = rel.pin('20261123T134308150391Z-402d42333024458b=df36889a5cae9222e586bedbe2a64a2fb047376d73cfaf6bc5150ffae707c7a2')
RAW_PIN = rel.pin('20261123T134826982683Z-1ad06f33c09d4541=4ea0bda9ae8bef9fec6874a782dd04f6c49dd93f977b72af27f6ee7bb35b22d6')
REGISTERED = '261b8520d09267c881c152a035b9c8606e802f61eb7224971562cd12ae879c6e'
REFERENCE = 'cc17f2eb753176cafac3afa3299d2fa8c427e0218d23fdc2698c3c081e4c0e53'
PLANE = '60343c45e13f4caff9cba726bf5bf8a143ffc47ec656f139b790c028f41d8370'
CASE_INDEX = '1323cd8fd285eea2af58133d2b7f63fefed676fd1e44d6eb43e6795ab85a97e0'
REFERENCE_INDEX = '07c4fa246d6fb78fb7d59d24817d5a80f5d2311dd256faed77df07f3f0da2715'
RAW_SUMMARY = 'f7f4e32aff9b1eb0b54d01e7eb04e835fd3d6b69f9364c23dc89e05e21629543'
RAW_FIGURES = ('prediction-v01.png','reference-v01.png','rgb-error-v01.png')
PRIMARY = ('L1-full', *rel.BRANCHES)
SCORE_FIGURES = {f'v01-{method}-{kind}.png' for method in PRIMARY for kind in ('risk','state')}
CASE_NAMES = {'CASE/'+n for n in CONTROL_INPUTS|{'case.json'}}
REF_NAMES = {'REFERENCE/reference.json', *('REFERENCE/'+v+'.npz' for v in VIEWS)}
REG_PATH = 'configs/vision-public-control-v1.json'
EVAL_PATH = 'scripts/evaluate_public_control.py'
RAW_REPORT = 'EVALUATION/raw-measurement.json'
SCORED_REPORT = 'EVALUATION/scored-measurement.json'
SEED_NAMES = CASE_NAMES | REF_NAMES | {REG_PATH,EVAL_PATH,'scripts/example_plane.py','PROVENANCE/case-evidence.json',
    'PROVENANCE/plane-source.json','PROVENANCE/depth-probe.json',RAW_REPORT,*('FIGURES/'+p for p in RAW_FIGURES)}
FINAL_FILES = SEED_NAMES | {'RESULTS/control-scores.json', SCORED_REPORT, 'README.md', 'LICENSE','licenses/CC-BY-4.0.txt',
    *('RESULTS/scores/'+v+'.npz' for v in VIEWS), *('FIGURES/'+p for p in SCORE_FIGURES)}
NAME = 'evidence-lens-'+rel.TAG+'-public-plane-512.zip'


def signature_members(root, names):
    return {n:rel.sig(root/n) for n in sorted(names)}


def files(root):
    return {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() or p.is_symlink()}


def compact(value):
    return {k:value[k] for k in ('run_id','manifest_sha256')}


def record(store, shelf, pin, kind, *, control=True):
    r=rel.Original(store,shelf,pin)
    info=r.run.run
    rel.require(info['run_kind']==kind,'wrong record kind for '+r.pin['run_id'])
    if control:
        rel.require(info['experiment_id']=='vision-public-native-control' and info['scene_family_id']==FAMILY and
                    info['condition_id']==CONDITION and info['split']=='development','unexpected scientific control identity')
    return r


def checked_evaluation(run, case, reference, with_scores, score_run=None):
    rel.same_parent(run.parents(),case.pin);rel.same_parent(run.parents(),reference.pin)
    if score_run:rel.same_parent(run.parents(),score_run.pin)
    doc=run.json('payload/measurement.json')
    rel.require(doc['schema']=='el-native-plane-control-evaluation-1' and doc['scores_included'] is with_scores and
                doc['case_sha256']==rel.sha(case.file('payload/CASE/case.json')) and
                doc['reference_sha256']==rel.sha(reference.file('payload/REFERENCE/reference.json')) and
                doc['family']==FAMILY and doc['condition']==CONDITION and set(doc['views'])==set(VIEWS),
                'independent measurement is for a different control or input')
    for v,r in doc['views'].items():
        rel.require(0<=r['rgb_high_error']<=r['E']<=r['V']<=512**2 and 0<=r['depth_evaluable']<=512**2 and
                    (set(r.get('methods',{}))==(set(METHODS) if with_scores else set())), 'invalid measured view '+v)
    return doc


def replay(packet, expected, scores=None):
    with tempfile.TemporaryDirectory(prefix='evidence-lens-control-replay-') as td:
        out=Path(td)/'result'
        cmd=[sys.executable,'-I','-B',str(packet/EVAL_PATH),'--case',str(packet/'CASE'),
             '--case-sha256',rel.sha(packet/'CASE/case.json'),'--reference',str(packet/'REFERENCE'),
             '--reference-sha256',rel.sha(packet/'REFERENCE/reference.json'),'--output',str(out)]
        if scores:cmd+=['--scores',str(scores)]
        subprocess.run(cmd,check=True,capture_output=True)
        rel.require(files(out)=={'measurement.json',*RAW_FIGURES},'independent evaluator emitted different members')
        # JSON is compared semantically so platform newline conventions leave identical numbers intact.
        rel.require(rel.read(out/'measurement.json')==rel.read(expected),'replayed six-view measurements differ from the sealed evaluation')
        from PIL import Image
        import numpy as np
        for name in RAW_FIGURES:
            with Image.open(out/name) as actual,Image.open(packet/'FIGURES'/name) as original:
                rel.require(actual.size==original.size==(512,512) and np.array_equal(np.asarray(actual),np.asarray(original)),
                            'independently reconstructed fixed view differs: '+name)


def material_sources(args, root):
    import numpy as np
    rel.require((rel.pin(args.case_export),rel.pin(args.reference_export),rel.pin(args.evaluation))==(CASE_PIN,REFERENCE_PIN,RAW_PIN),
                'original public study records required')
    case=record(args.store,args.shelf,CASE_PIN,'derivation');ref=record(args.store,args.shelf,REFERENCE_PIN,'derivation')
    raw=record(args.store,args.shelf,RAW_PIN,'evaluation')
    cfg=case.file('inputs/registration/vision-public-control-v1.json')
    rel.require(rel.sha(cfg)==REGISTERED==rel.sha(ref.file('inputs/registration/vision-public-control-v1.json')),'registered control configuration differs')
    reg=rel.read(cfg);evidence=case.json('payload/case-evidence.json');refdoc=ref.json('payload/REFERENCE/reference.json')
    plane_pin={'run_id':evidence['plane_run_id'],'manifest_sha256':PLANE}
    origin_pin={'run_id':reg['reference_run_id'],'manifest_sha256':REFERENCE}
    plane=record(args.store,args.shelf,plane_pin,'derivation');origin=record(args.store,args.shelf,origin_pin,'scene_render',control=False)
    rel.same_parent(case.parents(),plane.pin);rel.same_parent(ref.parents(),origin.pin);rel.same_parent(plane.parents(),origin.pin)
    rel.require(origin.run.run['scene_family_id']==FAMILY and origin.run.run['split']=='development' and origin.run.run['condition_id']=='B0'
                and reg['reference_manifest_sha256']==REFERENCE==refdoc['reference_manifest_sha256']==evidence['reference_manifest_sha256']
                and origin.pin['run_id']==refdoc['reference_run_id']==evidence['reference_run_id'] and refdoc['family']==FAMILY and
                refdoc['schema']=='el-vision-public-control-reference-1' and set(refdoc['files'])=={v+'.npz' for v in VIEWS},'wrong Blender reference origin')
    for k,c,n in [('source_rgb_sha256','original_source_sha256','input.png'),('source_cameras_sha256','original_cameras_sha256','cameras.json')]:
        rel.require(reg[k]==evidence[c]==rel.sha(origin.file('payload/'+n)),'authored reference image or cameras differ')
    cdoc=case.json('payload/CASE/case.json');case_sha=rel.sha(case.file('payload/CASE/case.json'))
    rel.require(evidence['config_sha256']==REGISTERED and evidence['case_sha256']==case_sha and cdoc['files']==evidence['inference_files'] and
                cdoc['files']['input.png']['sha256']==reg['source_rgb_sha256'] and cdoc['files']['cameras.json']['sha256']==reg['source_cameras_sha256'],
                'derived input packet differs from the authored source or registration')
    for n in sorted(CASE_NAMES):
        p=case.file('payload/'+n)
        if n!='CASE/case.json':
            name=n.removeprefix('CASE/');rel.require(rel.sig(p)==cdoc['files'][name]==rel.sig(plane.file('payload/example/'+name)),
                                                      'registered plane differs from the exported input '+name)
        rel.copy(root,n,p)
    with tempfile.TemporaryDirectory(prefix='evidence-lens-control-inputs-') as td:
        isolated=Path(td);control_inputs(root/'CASE',case_sha,isolated,reg)
        rel.require(files(isolated)==CONTROL_INPUTS,'scoring stage did not receive exactly the ten authorized inputs')
    descriptions=origin.json('payload/reference.json')['views']
    rel.copy(root,'REFERENCE/reference.json',ref.file('payload/REFERENCE/reference.json'))
    for v in VIEWS:
        n='REFERENCE/'+v+'.npz';exported=ref.file('payload/'+n);source=origin.file('payload/reference/'+v+'.npz')
        rel.require(rel.sig(exported)==refdoc['files'][v+'.npz'] and rel.sha(source)==refdoc['original_reference_npz_sha256'][v]==descriptions[v]['reference_sha256'],
                    'original target or exported reference digest differs for '+v)
        with np.load(exported,allow_pickle=False) as public,np.load(source,allow_pickle=False) as original:
            surface=np.isfinite(original['z']) & (original['z']>0) & np.isfinite(original['position']).all(axis=-1)
            rel.require(set(public.files)=={'rgb','z','surface'} and public['surface'].dtype==np.bool_ and public['surface'].shape==(512,512) and
                        np.array_equal(public['surface'],surface) and all(public[k].dtype==np.float32 and np.array_equal(public[k],original[k],equal_nan=True)
                            for k in ('rgb','z')), 'reference conversion changed original permitted values for '+v)
        rel.copy(root,n,exported)
    rel.copy(root,REG_PATH,cfg);adapter=plane.file('source/scripts/example_plane.py')
    rel.require(rel.sha(adapter)==reg['adapter_sha256'],'wrong authored plane adapter');rel.copy(root,'scripts/example_plane.py',adapter)
    rel.copy(root,'PROVENANCE/case-evidence.json',case.file('payload/case-evidence.json'))
    for name,source,key in [('plane-source.json','provenance.json','plane_provenance_sha256'),('depth-probe.json','predictions/depth-probe.json','depth_probe_sha256')]:
        p=plane.file('payload/example/'+source);rel.require(rel.sha(p)==evidence[key],'different plane provenance or depth check');rel.copy(root,'PROVENANCE/'+name,p)
    checked_evaluation(raw,case,ref,False);rel.copy(root,RAW_REPORT,raw.file('payload/measurement.json'))
    for n in RAW_FIGURES:rel.copy(root,'FIGURES/'+n,raw.file('payload/'+n))
    rel.copy(root,EVAL_PATH,raw.file('source/'+EVAL_PATH))
    rel.require(files(root)==SEED_NAMES,'unexpected material packet member')
    replay(root,root/RAW_REPORT)
    proof={'schema':'el-vision-public-control-materials-1','code_version':rel.TAG,'family':FAMILY,'condition':CONDITION,'case_sha256':case_sha,
           'reference_sha256':rel.sha(root/'REFERENCE/reference.json'),'registration_sha256':REGISTERED,'case_export':case.pin,'reference_export':ref.pin,
           'evaluation':raw.pin,'plane':plane.pin,'original_reference':origin.pin,'original_source_sha256':reg['source_rgb_sha256'],
           'original_cameras_sha256':reg['source_cameras_sha256'],'checked_reference_targets':6,'staged_scoring_inputs':sorted(CONTROL_INPUTS),
           'independently_replayed_raw_measurement_sha256':rel.sha(root/RAW_REPORT),'files':signature_members(root,SEED_NAMES)}
    return proof,case,ref,raw


def materials(args):
    from evidence_lens.scoring_io import new_output
    with new_output(args.output) as root:
        proof,*_=material_sources(args,root)
        rel.write(root,'materials.json',rel.entries(proof))
    return {'materials_sha256':rel.sha(args.output/'materials.json'),'case_sha256':proof['case_sha256'],'reference_sha256':proof['reference_sha256'],
            'members':len(proof['files']),'bytes':sum(r['bytes'] for r in proof['files'].values()),'case_export':proof['case_export'],
            'reference_export':proof['reference_export'],'evaluation':proof['evaluation'],'replayed_six_native_views':True}


def public_results(args, root, case, ref, raw, freeze, model_info):
    scores=record(args.store,args.shelf,rel.pin(args.scores),'model_score')
    graded=record(args.store,args.shelf,rel.pin(args.scored_evaluation),'evaluation')
    rel.same_parent(scores.parents(),case.pin);rel.same_parent(scores.parents(),freeze.pin)
    s=scores.json('payload/control-scores.json');pins={k:v['model_sha256'] for k,v in model_info.items()}
    rel.require(s['schema']=='el-vision-public-control-scores-1' and s['case_sha256']==rel.sha(root/'CASE/case.json') and
                s['inputs']==rel.read(root/'CASE/case.json')['files'] and s['condition']==CONDITION and s['family']==FAMILY and
                s['generator']=='authored-two-metre-plane' and s['models']==pins and s['status']=='scored' and s['runtime']['device']=='cpu' and
                set(s['score_files'])=={v+'.npz' for v in VIEWS},'wrong model score run or named input set')
    for v in VIEWS:
        name=v+'.npz';path=scores.file('payload/scores/'+name)
        rel.require(rel.sig(path)==s['score_files'][name],'wrong scored attachment map '+v)
        rel.copy(root,'RESULTS/scores/'+name,path)
    method=freeze.json('payload/method-freeze.json')
    with tempfile.TemporaryDirectory(prefix='evidence-lens-control-figures-') as td:
        target=Path(td);summary=control_result(root/'RESULTS/scores',target,method['thresholds'])
        rel.require(summary['score_files']==s['score_files'] and summary['views']==s['views'],'recorded per-method score counts differ from its actual arrays')
        for n in sorted(SCORE_FIGURES):
            saved=target/'maps'/n
            if 'map_files' in s:
                source=scores.file('payload/maps/'+n)
                from PIL import Image
                import numpy as np
                with Image.open(source) as original,Image.open(saved) as replayed:
                    rel.require(rel.sig(source)==s['map_files'][n] and np.array_equal(np.asarray(original),np.asarray(replayed)),
                                'fixed-view score figure differs from its recorded numerical map')
                saved=source
            rel.copy(root,'FIGURES/'+n,saved)
    # Publish numbers and declared identities, keeping any recorder/private runtime diagnostics in their original record.
    original_sha=rel.sha(scores.file('payload/control-scores.json'))
    public={k:s[k] for k in ('schema','status','case_sha256','family','condition','generator','models','inputs','views','score_files')}
    public['source_receipt_sha256']=original_sha
    public['runtime']={k:s['runtime'][k] for k in ('python','torch','numpy','architecture','system','device') if k in s['runtime']}
    rel.write(root,'RESULTS/control-scores.json',rel.entries(public))
    checked_evaluation(graded,case,ref,True,scores)
    scored=graded.json('payload/measurement.json');raw_result=raw.json('payload/measurement.json')
    rel.require(scored['six_view_totals']==raw_result['six_view_totals'] and all({k:x for k,x in detail.items() if k!='methods'}==raw_result['views'][v]
                    for v,detail in scored['views'].items()),'scoring changed the independent RGB/depth measurement')
    rel.require(rel.sha(graded.file('source/'+EVAL_PATH))==rel.sha(root/EVAL_PATH),'scored and unscored evaluation sources differ')
    rel.copy(root,SCORED_REPORT,graded.file('payload/measurement.json'))
    replay(root,root/SCORED_REPORT,root/'RESULTS/scores')
    return {'scores':scores.pin,'scored_evaluation':graded.pin,'original_score_receipt_sha256':original_sha}


def readme(doc,totals):
    return f'''# Bottle-carrier deployment control, 512 × 512

This attachment to Evidence Lens {rel.TAG} contains wany614's two-metre textured plane under the calibrated cameras of the first newly registered development family. Its condition ID is `public-plane`. The formal SHARP outcomes and the development procedure that chose the models are reported in the CPU archive.

`CASE/` holds ten deployment inputs and their manifest. `REFERENCE/` holds the six authored Blender targets for independent evaluation. `RESULTS/` holds six native NPZ outputs from the frozen CPU models. `FIGURES/` shows the fixed v01 Blender target, plane prediction, observed RGB error, and risk/state maps of the three fitted model packages. Each target produces seven named risk maps. Camera-space depth uses metres.

## Reproduce on a CPU

Extract the main CPU ZIP and this attachment into separate directories. From the CPU bundle on macOS or Linux, run the following command with Python 3.11 or 3.12 and adjust the paths to your extraction directories.

```sh
python3 -I -B smoke.py --case ../attachment/CASE --case-sha256 {doc['case_sha256']} --output ../fresh-plane
```

In Windows PowerShell use `py -3.12` in place of `python3`. The inference process only receives the ten declared image, calibration and prediction files. The six `scores/vNN.npz` files, 36 previews and `control-scores.json` are written in `fresh-plane`.

## Independently measure the six targets

From the attachment directory use Python with NumPy and Pillow, as specified by the main release requirements.

```sh
python3 -I -B scripts/evaluate_public_control.py --case CASE --case-sha256 {doc['case_sha256']} --reference REFERENCE --reference-sha256 {doc['reference_sha256']} --scores ../fresh-plane/scores --output ../fresh-measurement
```

Omit `--scores` to reproduce the raw RGB and depth measurement. The native evaluator opens the targets after scoring and reproduces `EVALUATION/scored-measurement.json` and the three v01 evaluation images. The raw measurement recorded {totals['V']:,} reference-valid pixels, {totals['E']:,} RGB-evaluable pixels and {totals['rgb_high_error']:,} mean-channel absolute display-sRGB errors above 0.05. The auxiliary depth metric uses {totals['depth_evaluable']:,} depth-evaluable pixels. The JSON retains the per-target numerators, denominators, sums and method counts.

Credit **wany614**. The project-authored images, predictions, numerical records and documentation are offered under CC BY 4.0. Project source and configuration are Apache-2.0. Full texts accompany the attachment. `control.json` identifies the upstream Blender reference, plane derivation, two material exports, two evaluations, selected models, code commit and every attachment member. Verify `SHA256SUMS` or the separately supplied SHA-256 for the complete ZIP before use.
'''


def build(args):
    from evidence_lens.scoring_io import new_output
    rel.require(rel.git_blob(args.repository,args.commit,'scripts/vision_control_attachment.py')==HERE.read_bytes(),'run the control builder from its declared public commit')
    freeze=record(args.store,args.shelf,rel.pin(args.method),'report',control=False);method=freeze.json('payload/method-freeze.json')
    rel.require(freeze.run.run['experiment_id']=='stage2-method-freeze' and method['run_id']==freeze.pin['run_id'],'joint published method freeze required')
    models=rel.approved_models(method)
    for k,v in models.items():
        origin=rel.Original(args.store,args.shelf,v);name='payload/model/model.json' if k=='L1' else 'payload/package/model.json'
        rel.same_parent(freeze.parents(),origin.pin);rel.require(rel.sha(origin.file(name))==v['model_sha256'],'different original model package '+k)
    clean_models={k:{j:v[j] for j in ('run_id','manifest_sha256','model_sha256')} for k,v in models.items()}
    with new_output(args.output) as dest,tempfile.TemporaryDirectory(prefix='evidence-lens-control-attachment-') as td:
        root=Path(td);material,case,ref,raw=material_sources(args,root);msha=rel.fingerprint(rel.entries(material))
        rel.require(msha==args.materials_sha256,'material sources differ from the independently prepared original record')
        for n in (EVAL_PATH,REG_PATH,'scripts/example_plane.py'):
            rel.require(rel.git_blob(args.repository,args.commit,n)==(root/n).read_bytes(),'the publicly tagged '+n+' differs from the recorded source')
        for n in ('LICENSE','licenses/CC-BY-4.0.txt'):rel.write(root,n,rel.git_blob(args.repository,args.commit,n))
        learned=public_results(args,root,case,ref,raw,freeze,models)
        doc={k:v for k,v in material.items() if k not in ('files','schema')}
        doc.update(schema='el-vision-public-plane-attachment-1',version=rel.TAG,repository='wany614/evidence-lens',code_commit=args.commit,
                   materials_sha256=msha,method={**freeze.pin,'sha256':rel.sha(freeze.file('payload/method-freeze.json'))},models=clean_models,**learned,
                   scope='Independent licensed two-metre plane deployment control; six registered development target views.',
                   rights={'attribution':'wany614','original_code_and_configs':'Apache-2.0','original_data_records_and_media':'CC-BY-4.0'},
                   figure_view='v01',methods=list(METHODS),coordinates_per_method=6*512**2)
        rel.write(root,'README.md',readme(doc,rel.read(root/RAW_REPORT)['six_view_totals']).encode())
        rel.require(files(root)==FINAL_FILES,'unexpected final public control members')
        doc['files']=signature_members(root,FINAL_FILES);rel.write(root,'control.json',rel.entries(doc))
        sums={**{k:v['sha256'] for k,v in doc['files'].items()},'control.json':rel.sha(root/'control.json')}
        rel.write(root,'SHA256SUMS',''.join(sums[n]+'  '+n+'\n' for n in sorted(sums)).encode())
        rel.archive_dir(root,dest/NAME);outer=rel.sig(dest/NAME)
        verified=read_public(dest/NAME,doc['case_sha256'],args.commit,doc['method']['sha256'],doc['models'])
        receipt={'schema':'el-vision-public-control-attachment-receipt-1','name':NAME,**outer,'case_sha256':doc['case_sha256'],
                 'reference_sha256':doc['reference_sha256'],'manifest_sha256':rel.sha(root/'control.json'),'sources':{k:doc[k] for k in
                 ('case_export','reference_export','evaluation','plane','original_reference','scores','scored_evaluation','method')},'members':len(verified['files'])+2}
        rel.write(dest,'attachment-receipt.json',rel.entries(receipt))
        if args.rebuild_of:
            for n in (NAME,'attachment-receipt.json'):
                rel.require(rel.sig(dest/n)==rel.sig(args.rebuild_of/n),'independent attachment rebuild differs: '+n)
    return receipt


def read_public(archive, case_sha, commit, method_sha, models):
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();expected=FINAL_FILES|{'control.json','SHA256SUMS'}
        rel.require(names==sorted(expected) and all(i.external_attr>>16==0o100644 for i in z.infolist()) and
                    sum(i.file_size for i in z.infolist())<1024**3,'public control ZIP differs from its fixed positive allowlist')
        raw=z.read('control.json');doc=json.loads(raw)
        rel.require(doc['schema']=='el-vision-public-plane-attachment-1' and doc['repository']=='wany614/evidence-lens' and doc['version']==rel.TAG and
                    doc['code_commit']==commit and doc['case_sha256']==case_sha and doc['method']['sha256']==method_sha and doc['models']==models and
                    doc['family']==FAMILY and doc['condition']==CONDITION and doc['plane']=={'run_id':'20261123T133630067002Z-4f3c2e0880774b9b','manifest_sha256':PLANE}
                    and doc['original_reference']=={'run_id':'20261123T124705616092Z-5739968b93d64b4d','manifest_sha256':REFERENCE}
                    and doc['case_export']==CASE_PIN and doc['reference_export']==REFERENCE_PIN and doc['evaluation']==RAW_PIN and doc['registration_sha256']==REGISTERED and
                    set(doc['files'])==FINAL_FILES and set(doc['staged_scoring_inputs'])==CONTROL_INPUTS and doc['figure_view']=='v01' and
                    doc['methods']==list(METHODS) and doc['rights']=={'attribution':'wany614','original_code_and_configs':'Apache-2.0','original_data_records_and_media':'CC-BY-4.0'},
                    'public control originals, models, grant, fixed view or study identities differ')
        for n,v in doc['files'].items():
            with z.open(n) as stream:
                rel.require(v['bytes']==z.getinfo(n).file_size and hashlib.file_digest(stream,'sha256').hexdigest()==v['sha256'], 'public attachment member differs: '+n)
        for p in (doc['scores'],doc['scored_evaluation'],doc['method'],*doc['models'].values()):
            rel.require(rel.RID.fullmatch(p['run_id']) and rel.HASH.fullmatch(p['manifest_sha256']),'incorrect scientific producer or seal in the attachment')
        sums={**{k:v['sha256'] for k,v in doc['files'].items()},'control.json':rel.fingerprint(raw)}
        registered=json.loads(z.read(REG_PATH));case=json.loads(z.read('CASE/case.json'));reference=json.loads(z.read('REFERENCE/reference.json'))
        rel.require(z.read('SHA256SUMS')==''.join(sums[n]+'  '+n+'\n' for n in sorted(sums)).encode() and
                    doc['files']['CASE/case.json']['sha256']==case_sha==CASE_INDEX and doc['files']['REFERENCE/reference.json']['sha256']==doc['reference_sha256']==REFERENCE_INDEX and
                    doc['files'][REG_PATH]['sha256']==REGISTERED and doc['files'][RAW_REPORT]['sha256']==RAW_SUMMARY and
                    case['files']['input.png']['sha256']==registered['source_rgb_sha256']==doc['original_source_sha256'] and
                    case['files']['cameras.json']['sha256']==registered['source_cameras_sha256']==doc['original_cameras_sha256'] and
                    all(doc['files']['CASE/'+n]==v for n,v in case['files'].items()) and all(doc['files']['REFERENCE/'+n]==v for n,v in reference['files'].items()),
                    'control checksums, registered source image, cameras or reference pins differ')
    return doc


def risk_readback(expected, actual):
    import numpy as np
    rows={};all_ok=True
    for v in VIEWS:
        row={}
        with np.load(expected/(v+'.npz'),allow_pickle=False) as a,np.load(actual/(v+'.npz'),allow_pickle=False) as b:
            keys={name+suffix for name in METHODS.values() for suffix in ('','_valid','_state')}|{'l0_reason'}
            rel.require(set(a.files)==set(b.files)==keys,'unexpected public score fields for '+v)
            reasons=np.array_equal(a['l0_reason'],b['l0_reason'])
            for m,k in METHODS.items():
                x,y=a[k],b[k];u,w=a[k+'_valid'],b[k+'_valid'];s,t=a[k+'_state'],b[k+'_state']
                rel.require(x.shape==y.shape==u.shape==w.shape==s.shape==t.shape==(512,512) and x.dtype==y.dtype==np.float32 and
                            u.dtype==w.dtype==np.bool_ and s.dtype==t.dtype==np.uint8 and np.array_equal(u,np.isfinite(x)) and
                            np.array_equal(w,np.isfinite(y)) and (s[~u]==255).all() and (t[~w]==255).all(),'invalid native score representation '+v+'/'+m)
                common=u&w;error=np.abs(x[common].astype(np.float64)-y[common].astype(np.float64));mx=float(error.max()) if error.size else None
                support_difference=int(np.count_nonzero(u!=w));state_difference=int(np.count_nonzero(s!=t));tol=.0002 if m in rel.BRANCHES else .000001
                passed=support_difference==state_difference==0 and (mx is None or mx<=tol) and (m!='L0' or reasons)
                row[m]={'pixels':int(x.size),'finite_expected':int(u.sum()),'finite_observed':int(w.sum()),'support_difference':support_difference,
                        'threshold_state_difference':state_difference,'maximum_absolute_difference':mx,'absolute_tolerance':tol,
                        'l0_reason_codes_equal':bool(reasons) if m=='L0' else None,'passed':bool(passed)}
                all_ok &= passed
        rows[v]=row
    return rows,bool(all_ok)


def grade_download(args):
    rel.require(rel.sha(args.archive)==args.archive_sha256 and rel.sha(args.release)==args.release_sha256,'independent outer ZIP or CPU release manifest differs')
    released=rel.read(args.release)
    rel.require(released['schema']=='el-vision-public-release-1' and released['version']==rel.TAG and released['repository']=='wany614/evidence-lens' and
                released['scientific_status']=='confirmed_heldout_report_recorded' and set(released['models'])=={'L1',*rel.BRANCHES},
                'use the original published CPU release')
    doc=read_public(args.archive,args.case_sha256,released['code_commit'],released['method_freeze']['sha256'],released['models'])
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);rel.require(not any(out.iterdir()),'select an empty public-control readback directory')
    evidence={'schema':'el-vision-public-control-readback-1','scope':'Separate licensed public-plane deployment control','archive_sha256':args.archive_sha256,
              'release_sha256':args.release_sha256,'models':doc['models'],'case_sha256':doc['case_sha256'],'case_export':doc['case_export'],
              'reference_export':doc['reference_export'],'original_reference':doc['original_reference'],'plane':doc['plane'],'evaluation':doc['evaluation'],
              'scored_evaluation':doc['scored_evaluation'],'scoring_run':doc['scores'],'original_source_sha256':doc['original_source_sha256'],
              'original_cameras_sha256':doc['original_cameras_sha256'],'checked_method_coordinates':len(METHODS)*len(VIEWS)*512**2}
    try:
        with tempfile.TemporaryDirectory(prefix='evidence-lens-control-independent-evaluation-') as td:
            work=Path(td)
            # The caller has completed input-only deployment. Only this separate process receives Blender targets.
            wanted=CASE_NAMES|REF_NAMES|{EVAL_PATH,SCORED_REPORT,*('RESULTS/scores/'+v+'.npz' for v in VIEWS),*('FIGURES/'+n for n in RAW_FIGURES)}
            with zipfile.ZipFile(args.archive) as z:
                for n in sorted(wanted):
                    target=work/n;target.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(n) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,2**20)
            rows,match=risk_readback(work/'RESULTS/scores',args.scores)
            target=out/'evaluation';cmd=[sys.executable,'-I','-B',str(work/EVAL_PATH),'--case',str(work/'CASE'),'--case-sha256',doc['case_sha256'],
                 '--reference',str(work/'REFERENCE'),'--reference-sha256',doc['reference_sha256'],'--scores',str(args.scores.resolve()),'--output',str(target)]
            subprocess.run(cmd,check=True,capture_output=True)
            rel.require(files(target)=={'measurement.json',*RAW_FIGURES},'independent public evaluation omitted its fixed view')
            from PIL import Image
            import numpy as np
            pictures={}
            for n in RAW_FIGURES:
                with Image.open(target/n) as a,Image.open(work/'FIGURES'/n) as b:pictures[n]=bool(a.size==b.size==(512,512) and np.array_equal(np.asarray(a),np.asarray(b)))
            measurement_match=rel.read(target/'measurement.json')==rel.read(work/SCORED_REPORT)
            evidence.update(views=rows,independent_measurement_exact=measurement_match,reproduced_measurement_sha256=rel.sha(target/'measurement.json'),
                            independent_figure_pixels_equal=pictures,passed=match and measurement_match and all(pictures.values()))
            rel.write(out,'public-control-readback.json',rel.entries(evidence))
            rel.require(evidence['passed'],'numerical or threshold readback differed; detailed per-method/per-view records are retained')
    except Exception as exc:
        rel.write(out,'failure.json',rel.entries({'schema':'el-vision-public-control-readback-failure-1','archive_sha256':args.archive_sha256,
                  'case_sha256':args.case_sha256,'error':str(exc),'exception':type(exc).__name__}))
        raise
    return {'passed':True,'case_sha256':doc['case_sha256'],'receipt_sha256':rel.sha(out/'public-control-readback.json'),
            'coordinates':evidence['checked_method_coordinates']}


def main():
    if sys.version_info < (3,11):raise SystemExit('This control attachment requires Python 3.11 or newer.')
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True)
    m=sub.add_parser('materials');b=sub.add_parser('build');g=sub.add_parser('grade-download')
    for p in (m,b):
        p.add_argument('--store',type=Path,required=True);p.add_argument('--shelf',type=Path)
        p.add_argument('--repository',type=Path,default=Path('.'))
        for n in ('case-export','reference-export','evaluation'):p.add_argument('--'+n,required=True)
    for n in ('method','scores','scored-evaluation','commit','materials-sha256'):b.add_argument('--'+n,required=True)
    b.add_argument('--rebuild-of',type=Path)
    for name in ('release','archive','scores'):g.add_argument('--'+name,type=Path,required=True)
    for name in ('archive-sha256','release-sha256','case-sha256'):g.add_argument('--'+name,required=True)
    for p in (m,b,g):p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.operation in ('materials','build'):
        sys.path.insert(0,str(args.repository.resolve()/'src'))
        result=materials(args) if args.operation=='materials' else build(args)
    else:result=grade_download(args)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
