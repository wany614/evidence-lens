"""Read explicitly selected recovery artifacts and original deployment/reference material."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dtu_eval import mask,photo,read_pinned
from dtu_score import VIEWS
from evidence_lens._bundle import HASH
from evidence_lens.formal_inputs import require,verified_file
from evidence_lens.multisource import CONDITIONS
from evidence_lens.multisource_io import Case
from evidence_lens.scoring_io import file_hash

SYNTH='scene_057_offset_bell_cage'
OBJECTS=('set1','set6',SYNTH)
STEPS=(0,500,1500,3000,5000)
SHAPE=(256,256)

def original(roots,pin):
    run,p=read_pinned(roots,pin);return run,p,json.loads(p.read_bytes())

def authority(roots,request):
    p=request['protocol'];r,f,data=original(roots,p)
    require(data.get('schema')=='el-recovery-registration-1' and data.get('run_id')==p['run_id'] and data.get('protocol_sha256')==data.get('registered_files',{}).get('protocol/reconstruction-recovery-v1.json') and data['registered_files'].get('protocol/recovery-risk-comparison-v1.json')=='d08647fd05b3c3ae1fedab75ea77b9cf869b4b7d0a3d3f3afca0f45445b79e31','missing original registered scientific protocol')
    return data

def recipe(roots,request):
    r,p,data=original(roots,request['recipe']);require(isinstance(data,dict) and data.get('schema'),'missing original recipe')
    return data

def permit_frozen(roots,request):
    _,_,data=original(roots,request['freeze']); require(data.get('recipe_sha256')==request['recipe']['sha256'] and data.get('registered_selection') and data.get('run_id')==request['freeze']['run_id'] and data.get('schedule_steps')==5000 and data.get('stop_step') in STEPS[1:],'missing complete recipe and registered stopping-rule freeze')
    return data

def from_spec(roots,owner,index_path,spec):
    if 'run_id' in spec:
        run,p=read_pinned(roots,spec);require(p.stat().st_size==spec['bytes'],'different archived member byte count');return p
    p=verified_file(index_path.parent,spec['path'],spec);native=p.relative_to(owner.root).as_posix();require(owner.files[native]['sha256']==spec['sha256'] and owner.files[native]['bytes']==spec['bytes'],'member differs from its recorder manifest');return p


class Outputs:
    def __init__(self,roots,request,case=None):
        self.roots=roots;self.run,self.path,self.data=original(roots,request['outputs']);d=self.data
        cond=request['condition'];obj=request['object_id']; require(obj in OBJECTS and cond in CONDITIONS and (obj!=SYNTH or cond=='M4'),'unknown registered object/condition')
        require(d['schema']=='el-recovery-outputs-1' and d['object_id']==obj and d['condition']==cond and d['recipe_sha256']==request['recipe']['sha256'] and d['protocol_sha256']==authority(roots,request)['protocol_sha256'],'different returned run identity or recipe')
        require(d['status'] in ('completed','truncated','failed','not_run') and type(d['completed_steps']) is int and type(d['schedule_steps']) is int and 0<=d['completed_steps']<=d['schedule_steps']==5000,'unspecified or inconsistent schedule')
        require(d['generator_source_ids']==list(CONDITIONS[cond]) and set(d['source_rgb_sha256'])==set(CONDITIONS[cond]),'source roles differ from the condition')
        if case:
            require(case.condition==cond and d['source_rgb_sha256']=={s:r['sha256'] for s,r in case.meta['source_images'].items()} and d['source_camera_sha256']==case.meta['source_cameras']['sha256'] and d['render_camera_sha256']==case.meta['render_cameras']['sha256'],'different source photographs or cameras')
        for key,z in d['checkpoints'].items():
            step=int(key);require(key==str(step) and z['step']==step and 0<=step<=d['completed_steps'] and type(z['gaussians']) is int and z['gaussians']>0 and isinstance(z['phase'],str) and z['phase'] and HASH.fullmatch(z['checkpoint_sha256']),'invalid checkpoint metadata')
            require(set(z['views'])==set((*CONDITIONS[cond],*VIEWS)),'one checkpoint must enumerate all source/target dispositions')
        if d['status']=='truncated' or d['status']=='failed':require(bool(d.get('failure_reason') or d.get('reason')),'truncation or failure needs an original reason')
    def asset(self,step,view):
        z=self.data['checkpoints'][str(step)]['views'][view]
        if z.get('status') in ('failed','missing'):
            require(bool(z.get('reason')),'no missing frame reason');return None
        return from_spec(self.roots,self.run,self.path,z)
    def rgb(self,step,view):
        path=self.asset(step,view)
        if path is None:return None
        if path.suffix.lower()=='.npy':a=np.load(path,allow_pickle=False)
        else:
            require(path.suffix.lower()=='.npz','full precision render must be .npy or .npz')
            with np.load(path,allow_pickle=False) as z:require('rgb' in z,'missing predicted RGB');a=z['rgb']
        require(a.shape==(*SHAPE,3) and a.dtype==np.float32,'expected native 256-square display RGB float32');return a

class Originals:
    """Never opens a score, a target render to select a source, or an unlisted reference."""
    def __init__(self,roots,request):
        self.roots=roots;self.request=request;self.case_pin=request['original_case'];_,p=read_pinned(roots,self.case_pin);self.case=Case(p.parent,self.case_pin['sha256']);self.case_path=p
        self.obj=request['object_id'];require(self.case.condition==request['condition'] and self.obj in OBJECTS,'different original case')
        self.sources={s:t[0] for s,t in self.case.sources('all').items()};self.masks={};self.refs={};self.hashes={};self.object_masks={};self.object_reason={}
        if self.obj==SYNTH:
            require(self.case.condition=='M4' and set(request['reference_views'])==set(VIEWS),'incomplete synthetic positive control')
            for v in VIEWS:
                _,q=read_pinned(roots,request['reference_views'][v]);require(q.name==v+'.npz','wrong synthetic native target reference')
                with np.load(q,allow_pickle=False) as z:a=z['rgb']
                require(a.dtype==np.float32 and a.shape==(*SHAPE,3),'incorrect native synthetic FP32 RGB')
                self.refs[v]=a;self.masks[v]=np.ones(SHAPE,bool);self.hashes[v]=file_hash(q);self.object_masks[v]=None;self.object_reason[v]='No independently designated binary object-region criterion is registered for this control.'
        else:
            _,q,meta=original(roots,request['references']);_,_,prov=original(roots,request['original_provenance'])
            require(meta['schema']=='el-dtu-references-1' and meta['object_id']==self.obj and prov['object_id']==self.obj and prov['condition']==self.case.condition and prov['case_sha256']==self.case.sha256 and set(meta['views'])==set(VIEWS),'wrong original DTU crop or object')
            for k in ('image_origin','camera_plan_sha256','view_set_sha256','crop_sha256','camera_sha256','lighting_id','source_view_ids','target_view_ids'):require(meta.get(k)==prov.get(k),'photograph/camera identity mismatch: '+k)
            for v in VIEWS:
                row=meta['views'][v];require(row['official_view_id']==meta['target_view_ids'][v],'wrong official capture');self.refs[v]=photo(q.parent,row['rgb']);self.masks[v]=mask(q.parent,row['valid_mask']);self.hashes[v]=row['rgb']['sha256'];obj=row['object_mask']
                if obj is not None:require(bool(obj.get('evidence') or meta.get('object_mask_evidence')),'missing independent source of image-space object region')
                self.object_masks[v]=mask(q.parent,obj) if obj is not None else None;self.object_reason[v]=row.get('object_mask_reason')
