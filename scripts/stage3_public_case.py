"""Export the preselected first development family's two actual native deployment cases and separate evaluation reference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from evidence_lens._bundle import encoded,load,local_path,sha
from evidence_lens._verify import require,verify_bundle
from make_handoff import publish
from stage3_public_blend import decode,portable_copy
from stage3_reference import located,perform

FAMILY='scene_057_offset_bell_cage'
CONDS=('M2','M4')
SOURCES=('s00','s01','s02','s03')
TARGETS=tuple(f'v{i:02d}' for i in range(1,7))
APACHE='Apache-2.0'
CC='CC-BY-4.0'


def work(a):
    spec=load(a.request)
    require(spec['family']==FAMILY and spec['case_selection']=='first registered development family' and
            set(spec['trials'])==set(CONDS),'Public example must use exactly the registered development family')
    ref=load(a.reference/'reference.json')
    roles={}
    def original(path,dst,permission,pin,member,expected=None):
        n=local_path(dst)
        if expected:require(sha(path)==expected,'Changed candidate '+n)
        destination=a.out/n;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
        roles[n]={'license':permission,'original':{**pin,'member':member,'sha256':sha(path)}}
    def derived(name,content,lic=CC):
        dest=a.out/local_path(name)
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
        roles[name]={'license':lic}
    def copy_ref(name,where,lic=CC,expected=None):
        original(a.reference/name,where,lic,spec['reference'],'payload/'+name,expected)
    copy_ref('cameras.json','CASE/cameras.json',expected=ref['camera_sha256'])
    copy_ref('cameras.json','REFERENCE/cameras.json',expected=ref['camera_sha256'])
    original(a.reference_config,'REFERENCE/world/config.json',APACHE,spec['reference'],'config.resolved.yaml',spec['reference_config_sha256'])
    original(a.gaussian_worker,'CASE/code/stage3_gaussians.py',APACHE,spec['trials']['M2']['reconstruction'],
             'source/src/evidence_lens/stage3_gaussians.py',spec['gaussian_worker_sha256'])
    for s in SOURCES:copy_ref('sources/'+s+'.png','CASE/sources/'+s+'.png',expected=ref['source_images'][s]['source']['sha256'])
    numerical={}
    for i,c in enumerate(CONDS):
        case,rec,pred=a.case[i],a.generation[i],a.prediction[i]
        trial=spec['trials'][c];case_doc=load(case/'case.json');r=load(rec/'reconstruction.json');m=load(pred/'prediction.json')
        ids=('s00','s03') if c=='M2' else SOURCES
        require(case_doc['schema']=='el-stage3-deploy-case-1' and case_doc['condition']==c and tuple(case_doc['generator_source_ids'])==ids and
                tuple(case_doc['target_ids'])==TARGETS and r['run_id']==trial['reconstruction']['run_id'] and r['gaussian_optimization_steps']==256 and
                r['final_gaussian_count']==4096 and m['reconstruction_run_id']==r['run_id'] and m['run_id']==trial['render']['run_id'] and m['depth_verified'] and
                r['checkpoint_sha256']==m['checkpoint_sha256']==case_doc['checkpoint_sha256'] and case_doc['reconstruction_run_id']==r['run_id'],
                'Public case must come from actual formal joint optimization at the registered budget')
        required={'case.json','source_cameras.json','render_cameras.json','prediction.json',*('sources/'+s+'.png' for s in ids),
                  *('predictions/'+v+'.npz' for v in ids+TARGETS)}
        require({f.relative_to(case).as_posix() for f in case.rglob('*') if f.is_file()}==required,'Unexpected deployed case member')
        entries=[case_doc[n] for n in ('source_cameras','render_cameras','prediction')]+list(case_doc['source_images'].values())+list(case_doc['renders'].values())
        for v in entries:
            source=case/local_path(v['path'])
            require(sha(source)==v['sha256'] and source.stat().st_size==v['bytes'],'Changed numerical deployment input')
        for name in sorted(required):
            original(case/name,'CASE/'+c+'/'+name,CC,spec['dataset'],'payload/cases/'+FAMILY+'/'+c+'/'+name)
        # The native SDK consumes prediction.json at the condition root. The full native evidence uses identical bytes.
        require(sha(pred/'prediction.json')==sha(case/'prediction.json'),'Scoring dataset and selected native render disagree')
        for v in ids+TARGETS:
            require(sha(pred/(v+'.npz'))==m['views'][v]['sha256']==sha(case/'predictions'/(v+'.npz')),'Native prediction differs from scored input')
            original(pred/(v+'.png'),'CASE/'+c+'/predictions/'+v+'.png',CC,trial['render'],'payload/'+v+'.png',m['views'][v]['png_sha256'])
        for name,lic,expected in [('scene.npz',APACHE,r['checkpoint_sha256']),('optimizer.jsonl',CC,r['trace_sha256']),('reconstruction.json',CC,None)]:
            original(rec/name,'CASE/'+c+'/'+name,lic,trial['reconstruction'],'payload/'+name,expected)
        for role,cfg in [('generation',a.generation_config[i]),('render',a.render_config[i])]:
            original(cfg,'CASE/'+c+'/'+role+'-config.json',APACHE,trial['reconstruction' if role=='generation' else 'render'],
                     'config.resolved.yaml',trial[role+'_config_sha256'])
        if i==0:original(rec/'initial.npz','CASE/common-initial.npz',APACHE,trial['reconstruction'],'payload/initial.npz',r['initial_checkpoint_sha256'])
        else:require(sha(a.out/'CASE/common-initial.npz')==r['initial_checkpoint_sha256']==sha(rec/'initial.npz'),'Condition initialization is not byte-identical')
        original(pred/'depth-probe.json','CASE/'+c+'/predictions/depth-probe.json',CC,trial['render'],'payload/depth-probe.json',m['depth_probe_sha256'])
        numerical[c]={'original_sdk_case_sha256':sha(case/'case.json'),'registered_sources':list(ids),'reconstruction':trial['reconstruction'],'render':trial['render'],
            'scene_sha256':r['checkpoint_sha256'],'common_initial_sha256':r['initial_checkpoint_sha256'],'updates':r['gaussian_optimization_steps'],
            'source_pixels':r['sampled_source_pixels'],'final_gaussian_count':r['final_gaussian_count'],'source_sampling':r['source_sampling']}
    for v in TARGETS:
        for suffix in ('npz','png'):
            n='reference/'+v+'.'+suffix
            copy_ref(n,'REFERENCE/'+n,expected=ref['views'][v][suffix]['sha256'])
    for n in ('recipe.json','settings.json','runtime.json'):
        copy_ref('scene/'+n,'REFERENCE/world/'+n,CC if n=='runtime.json' else APACHE)
    for n in ('scene.py','passes.py','validation.py','stage3_native.py'):
        target='REFERENCE/world/code/src/evidence_lens/'+('' if n=='stage3_native.py' else 'blender_reference/')+n
        copy_ref('scene/'+n,target,APACHE)
    original(a.optimizer,'CASE/optimizer-settings.json',APACHE,spec['registration'],'payload/optimizer.json')
    original(a.readback,'REFERENCE/proofs/original-native-readback.json',CC,spec['readback'],'payload/readback.json')
    copy_ref('reference.json','REFERENCE/proofs/original-private-bundle-index.json')
    native=a.reference/ref['blend']['path']; dest=a.out/'REFERENCE/world/offset-bell-cage.blend'
    require(sha(native)==ref['blend']['sha256'],'Unexpected source blend')
    clean=portable_copy(native,dest)
    roles['REFERENCE/world/offset-bell-cage.blend']={'license':CC,'derived_from':{**spec['reference'],'member':'payload/'+ref['blend']['path'],'sha256':sha(native)}}
    before=decode(native);after=dest.read_bytes()
    diffs=[{'offset':i,'before':x,'after':y} for i,(x,y) in enumerate(zip(before,after,strict=True)) if x!=y]
    (a.out/'private-ui-byte-differences.json').write_bytes(encoded({'field':clean['field'],'delta':diffs,'never_distribute':True}))
    derived('REFERENCE/proofs/portable-blend.json',encoded(clean))
    replay=a.out/'REFERENCE/proofs/rerender-v01.json'
    with tempfile.TemporaryDirectory() as scratch:
        subprocess.run([str(a.blender_python),'-B',str(Path(__file__).with_name('stage3_case_replay.py')),'--original',str(native),
            '--portable',str(dest),'--helpers',str(a.reference/'scene'),'--cameras',str(a.reference/'cameras.json'),
            '--settings',str(a.reference/'scene/settings.json'),'--scratch',scratch,'--truth',str(a.reference/'reference/v01.npz'),'--out',str(replay)],check=True)
    require(load(replay)['passed'],'Public native verification failed')
    roles['REFERENCE/proofs/rerender-v01.json']={'license':CC}
    fixed={'family':FAMILY,'development_case_order':1,'display_view':'v01','target_ids':list(TARGETS),'creator':'wany614',
           'catalog_sha256':spec['catalog_sha256'],'study_sha256':spec['study_sha256'],'reference':spec['reference'],'budget':spec['budget']}
    derived('CASE/case-index.json',encoded({**fixed,'schema':'el-stage3-public-joint-case-1','conditions':numerical,
        'encoding':'display-sRGB [0,1]','camera_axes':'OpenCV right, down, forward in metres. Integer-centre K.','alpha':'accumulated opacity',
        'depth':'foreground-opacity-normalized camera-Z in metres; NaN when unsupported','source_usage':'Each Gaussian scene was jointly optimized on only its indicated RGB sources.',
        'dataset':spec['dataset']}))
    derived('REFERENCE/reference.json',encoded({**fixed,'schema':'el-stage3-public-target-reference-1','views':{v:{k:ref['views'][v][k] for k in ('npz','png')} for v in TARGETS},
        'blend_sha256':sha(dest),'visibility_fields':['visibility_'+s for s in SOURCES],'visibility_codes':{'0':'invalid/background','1':'observed','2':'disoccluded','3':'outside'},
        'position':'float32 world XYZ metres','z':'float32 positive first-hit camera-Z metres','rgb':'float32 display-sRGB [0,1]','geometry_invalid':'NaN for Position/Z, ID zero for background'}))
    derived('CASE/README.md',('''# Two- and four-image joint reconstructions of the offset bell cage

This is the first registered development family of the Evidence Lens C study. M2 used s00 and s03. M4 added the two equally spaced interior source images within the same 0.72 m endpoint span. Both models started from the exact same 4,096 world-space Gaussians and completed 256 updates with one complete 256 × 256 source per update. Source losses, per-update gradients, checkpoints and separate native render identities are retained.

Start with `M2/case.json` or `M4/case.json` for the deployment interface. Each exact input list includes the allowed original source PNGs and cameras, source self-renders, six target predictions, and `prediction.json`. Every numerical prediction has float32 RGB in display sRGB, alpha and expected optical-axis depth in metres. Neighbouring PNGs are previews. `case-index.json` locates the two reconstruction and render identities. The corresponding independent study reference is distributed separately.

For CUDA replay, `code/stage3_gaussians.py` contains the original `optimize` and `render` commands. Each condition includes its own `generation-config.json` and `render-config.json`. Rendering the distributed checkpoint uses the original `scene.npz`, `reconstruction.json` and `render_cameras.json`. Supply a separate empty output directory and assign a new local `EL_RUN_ID` when invoking the standalone worker.

Credit wany614, Evidence Lens. The two optimized scene parameter files, their common initialization and the optimizer configuration use Apache-2.0. The original RGB, camera records, predicted arrays, images, traces, provenance records and documentation use CC BY 4.0. gsplat is installed independently. [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) · [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) · [gsplat 1.5.3](https://github.com/nerfstudio-project/gsplat/tree/v1.5.3)
''').encode())
    derived('REFERENCE/README.md',('''# Independent Blender reference for the offset bell cage

The six target NPZs preserve the registered native-resolution RGB, first-hit world Position, positive optical-axis Z in metres, object identity, geometric edge and source-specific visibility. `visibility_s00` through `visibility_s03` distinguish unknown/background, observed, newly disoccluded and outside-source-frame geometry. Use only the sources allowed by the condition when aggregating visibility. These are evaluation-only assets.

The native world in `world/offset-bell-cage.blend` contains the original scene, cameras and authored materials. The public copy clears Blender's saved file-browser directory. `proofs/portable-blend.json` records the exact binary scope and original digest. `proofs/rerender-v01.json` compares an actual new render from both the original and portable worlds with the selected v01 measurements. The complete original private ten-view bundle is identified separately in `proofs/original-private-bundle-index.json`. The public target inventory is `reference.json`.

To regenerate the original world, run `world/code/src/evidence_lens/stage3_native.py produce` in the registered Blender Python environment. Supply `--config world/config.json --settings world/settings.json --recipe world/recipe.json --cameras cameras.json`, an empty `--out` directory and a new local `EL_RUN_ID`.

Credit wany614, Evidence Lens. The native world, measured reference data, images, camera records, verification records and documentation use CC BY 4.0. The Python sources and fixed numeric configurations use Apache-2.0. The Blender runtime is distributed by the Blender Foundation. [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) · [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) · [Blender licensing](https://www.blender.org/about/license/)
''').encode())
    for path in (a.out/'CASE',a.out/'REFERENCE'):
        for p in path.rglob('*'):
            if p.is_file() and p.suffix.lower() in ('.json','.md','.jsonl','.py'):
                data=p.read_bytes()
                require(all(needle not in data for needle in (b'/Users/',b'/home/',b'CloudStorage/',b'@gmail.com',b'AppData\\Local')),
                        'A private path entered the public projection')
    require(set(roles)=={p.relative_to(a.out).as_posix() for dr in ('CASE','REFERENCE') for p in (a.out/dr).rglob('*') if p.is_file()},'Public allowlist differs')
    (a.out/'asset-origin-ledger.json').write_bytes(encoded({'schema':'el-stage3-public-example-origins-1','science':fixed,'files':roles}))


def record(a):
    project,assets,store=a.project.resolve(),a.assets.resolve(),a.store.resolve()
    catalog=load(assets/'protocol/stage3-families-v1.json');cat=sha(assets/'protocol/stage3-families-v1.json')
    require(next(r for r in catalog['families'] if r['split']=='development')['scene_family_id']==FAMILY,'Frozen public case differs')
    study=load(a.study);require(study['catalog_sha256']==cat and 'first development family' in study['display']['native_example'],'Study has no fixed public example')
    base=store/'stage3'/cat; receipt=load(base/'references'/(FAMILY+'.json'));ref=receipt['reference'];rb=receipt['readback'];reg=receipt['registration']
    bp=load(base/'production-budget.json')['budget']
    ds_id, ds_hash=a.dataset.split('=')
    ds={'run_id':ds_id,'manifest_sha256':ds_hash};dataset=located(store,ds)
    data=load(dataset/'payload/dataset.json')
    dmeta=load(dataset/'run.json')
    require(dmeta['execution_status']=='succeeded' and dmeta['run_kind']=='dataset_build' and data.get('split')=='development','Official full development dataset required')
    source=located(store,ref)/'payload'
    control_sources={}
    def control(role,signed,pin,member,expected):
        verified=verify_bundle(signed)
        index={x['path']:x['sha256'] for x in verified['manifest']['files']}
        require(verified['manifest_sha256']==pin['manifest_sha256'] and index[member]==sha(signed/member)==expected,
                'Control file does not belong to its declared full scientific manifest')
        control_sources[role]=(signed/member,{'run_id':pin['run_id'],'manifest_sha256':pin['manifest_sha256'],'member':member,'sha256':expected})
    control('referenceconfig',source.parent,ref,'config.resolved.yaml',sha(source.parent/'config.resolved.yaml'))
    request={'family':FAMILY,'case_selection':'first registered development family','catalog_sha256':cat,'study_sha256':sha(a.study),
             'registration':reg,'reference':ref,'reference_config_sha256':sha(source.parent/'config.resolved.yaml'),
             'readback':rb,'budget':bp,'dataset':ds,'trials':{}}
    roles={'study':(a.study,None),'reference':(source,'reference'),'readback':(located(store,rb)/'payload/readback.json','readback'),
           'optimizer':(located(store,reg)/'payload/optimizer.json','registration')}
    parents={'registration':reg,'budget':bp,'reference':ref,'readback':rb,'dataset':ds}
    args=['{python}','{source}/scripts/stage3_public_case.py','work','--request','{input:request}','--reference','{input:reference}','--readback','{input:readback}',
          '--optimizer','{input:optimizer}','--reference-config','{input:referenceconfig}','--gaussian-worker','{input:gaussianworker}',
          '--out','{out}','--blender-python',str(a.blender_python.absolute())]
    for c in CONDS:
        selection=base/'trials'/(FAMILY+'-'+c)/'selection.json'
        q=load(selection);pins=q['selected']
        require(q['phase']=='production' and q['selection_status']=='selected' and q['production_budget']==bp and q['optimization_steps']==256
            and q['family']==FAMILY and q['condition']==c and q['split']=='development' and pins['reference']==ref,'Wrong public scientific selection')
        cd=dataset/'payload/cases'/FAMILY/c
        require(cd.is_dir(),'The selected development data does not contain the official source-native deploy case')
        roles['case'+c]=(cd,'dataset')
        args+=['--case','{input:case'+c+'}']
        request['trials'][c]={'selection_sha256':sha(selection),'reconstruction':pins['reconstruction'],'render':pins['render']}
        for role,key in [('generation','reconstruction'),('prediction','render')]:
            tag=role+c;parents[tag]=pins[key];sealed=located(store,pins[key]);roles[tag]=(sealed/'payload',tag)
            args+=['--'+role,'{input:'+tag+'}']
            name='generation' if key=='reconstruction' else 'render'
            cfg=sealed/'config.resolved.yaml'
            control(tag+'config',sealed,pins[key],'config.resolved.yaml',sha(cfg))
            request['trials'][c][name+'_config_sha256']=sha(cfg)
            args+=['--'+name+'-config','{input:'+tag+'config}']
            worker=sealed/'source/src/evidence_lens/stage3_gaussians.py'
            worker_hash=sha(worker)
            if 'gaussian_worker_sha256' not in request:
                request['gaussian_worker_sha256']=worker_hash
                control('gaussianworker',sealed,pins[key],'source/src/evidence_lens/stage3_gaussians.py',worker_hash)
            require(request['gaussian_worker_sha256']==worker_hash,'CUDA implementation differs across 057 original stages')
    out=base/'public-example';marker=out/'selection.json'
    if marker.exists():
        saved=load(marker);require(saved['dataset']==ds,'Published dataset changed');return saved
    with tempfile.TemporaryDirectory(dir=store/'work',prefix='stage3-public-source-') as tmp:
        src=Path(tmp)/'engine';(src/'scripts').mkdir(parents=True)
        for origin,n in [(assets,'stage3_public_case.py'),(assets,'stage3_public_blend.py'),(assets,'stage3_case_replay.py'),(assets,'stage3_reference.py'),
                         (project,'make_handoff.py'),(project,'formal_lineage.py')]:
            shutil.copyfile(origin/'scripts'/n,src/'scripts'/n)
        # Configs and source are recorder control records. Their parent manifest authenticates them separately
        # from the scientific payload inventory used by --input-parent.
        for name,(original,identity) in control_sources.items():
            destination=Path(tmp)/'signed-controls'/name/original.name
            destination.parent.mkdir(parents=True);shutil.copyfile(original,destination)
            require(sha(destination)==identity['sha256'],'Changed control while staging')
            request.setdefault('signed_control_members',{})[name]=identity
            roles[name]=(destination,None)
        spec=Path(tmp)/'public-projection.json';spec.write_bytes(encoded(request));roles['request']=(spec,None)
        cfg={'experiment_id':'stage3-public-case','trial_id':FAMILY+'-M2-M4-public','dataset_hash':cat,'scene_family_id':FAMILY,'split':'development',
             'public_projection_sha256':sha(spec),'recording':{'expected_output_bytes':65000000}}
        pin=perform(store,project,src,cfg,'report',roles,parents,args)
    sealed=located(store,pin)
    ledger=load(sealed/'payload/asset-origin-ledger.json')
    projections=[]
    for name,row in sorted(ledger['files'].items()):
        dest=sealed/'payload'/name
        item={'path':name,'source':{**pin,'member':'payload/'+name,'sha256':sha(dest)},'license':row['license'],'bytes':dest.stat().st_size}
        if 'original' in row:item['original_source']=row['original']
        if 'derived_from' in row:item['derived_from']=row['derived_from']
        projections.append(item)
    allow={'schema':'el-stage3-public-members-1','family':FAMILY,'files':projections}
    publish(out/'public-members.json',encoded(allow))
    receipt={'schema':'el-stage3-public-case-selection-1','family':FAMILY,'run':pin,'dataset':ds,'allowlist_sha256':sha(out/'public-members.json'),
             'files':len(projections),'bytes':sum(x['bytes'] for x in projections),'case_sha256':{c:sha(sealed/'payload/CASE'/c/'case.json') for c in CONDS},
             'portable_blend_sha256':sha(sealed/'payload/REFERENCE/world/offset-bell-cage.blend'),'public_native_verification_sha256':sha(sealed/'payload/REFERENCE/proofs/rerender-v01.json')}
    publish(marker,encoded(receipt))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('work','record'))
    for n in ('store','project','assets','study','request','reference','readback','optimizer','out','blender-python','reference-config','gaussian-worker'):
        p.add_argument('--'+n,type=Path)
    p.add_argument('--dataset')
    for n in ('case','generation','prediction','generation-config','render-config'):p.add_argument('--'+n,type=Path,action='append',default=[])
    a=p.parse_args()
    if a.action=='work':work(a)
    else:print(json.dumps(record(a),sort_keys=True))
