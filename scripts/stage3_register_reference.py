"""Materialize the C reference roster and camera rig before acquisition. Renders no images."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from stage3_scene_designs import designs


SOURCES=('s00','s01','s02','s03')
TARGETS=tuple(f'v{i:02d}' for i in range(1,7))
SETS={'M2':['s00','s03'],'M4':list(SOURCES)}


def encode(data):
    return (json.dumps(data,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def store(path, data):
    content=encode(data)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=content:
            raise ValueError('Already registered with different bytes: '+path.name)
    else:
        with path.open('xb') as handle:
            handle.write(content)
    return sha(path)


def register(assets, project, time):
    import bpy
    from evidence_lens.blender_reference.formal_catalog import geometry_fingerprint
    from stage2_reference_catalog import pair_structure
    sys.path.insert(0,str(project/'src/evidence_lens/blender_reference'))
    from scene import camera, camera_record, camera_bundle
    world=json.loads((project/'configs/formal-scenes-v1.json').read_text())['shared_world']
    base=json.loads((project/'configs/reference.json').read_text())['reference']
    positions={'s00':[-.36,-1.90,.86],'s01':[-.12,-1.90,.86],'s02':[.12,-1.90,.86],'s03':[.36,-1.90,.86],
               'v01':[-.23,-1.83,.91],'v02':[.23,-1.83,.91],'v03':[-.46,-1.87,.82],
               'v04':[.46,-1.87,.82],'v05':[-.06,-1.97,1.02],'v06':[.07,-1.79,.76]}
    rig={'positions':positions,'lens_mm':38,'sensor_mm':36,'clip_m':[.05,30.0],'aim':[0,.045,.405]}
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene
    scene.render.resolution_x=scene.render.resolution_y=256
    scene.render.resolution_percentage=100
    cams={i:camera(scene,i,rig,pos,rig['aim']) for i,pos in positions.items()}
    bpy.context.view_layer.update()
    cameras=camera_bundle({i:camera_record(scene,cams[i],'source' if i in SOURCES else 'research_target') for i in cams})
    camera_path='configs/stage3-cameras-v1.json'
    camera_sha=store(assets/camera_path,cameras)
    materials={**base['materials'],'teal':{'linear':[.055,.38,.33]},'rose':{'linear':[.63,.13,.20]},
               'chalk':{'linear':[.76,.73,.65],'checker_linear':[.70,.70,.63],'scale':9}}
    settings={**base,'schema':'el-stage3-native-reference-settings-1','width':256,'height':256,'preview_width':128,
              'samples':96,'seed':3367,'repeat_seed':5279,'camera':rig,'objects':world['objects'],
              'area_light':world['area_light'],'materials':materials,
              'validation':{**base['validation'],'occlusion_profile':'opaque_mesh_pairs','band_min_width_px':4,'band_min_interior_pixels':6}}
    for k in ('occluder_id','revealed_object_id'):
        settings['validation'].pop(k)
    settings_path='configs/stage3-reference-v1.json'
    ssha=store(assets/settings_path,settings)
    second=json.loads((project/'protocol/stage2-families-v1.json').read_text())
    original=json.loads((project/'configs/formal-scenes-v1.json').read_text())
    past=set()
    for r in second['families']:
        past.add(r['pair_structure_sha256'])
    other=[d['objects'] for d in original['families'].values()]
    other.append([o for o in base['objects'] if o['id']>=10])
    for p in other:
        if p:
            past.add(pair_structure(p))
    rows=[]
    for i,c in enumerate(designs(),49):
        split='train' if i<=56 else 'development' if i<=60 else 'test'
        recipe='configs/c-reference-v1/recipes/'+c.family+'.json'
        signature=pair_structure(c.objects)
        if signature in past: raise ValueError('Duplicate foreground structure '+c.family)
        past.add(signature)
        rsha=store(assets/recipe,{'schema':'el-stage3-world-recipe-1','title':c.title,'occlusion':c.relation,'objects':c.objects})
        rows.append({'scene_family_id':c.family,'scene_id':c.family+'/v001','scene_version':'v001','split':split,'pilot':i in (49,50),
                     'title':c.title,'occlusion':c.relation,'recipe_path':recipe,'recipe_sha256':rsha,
                     'geometry_fingerprint':geometry_fingerprint(c.objects),'pair_structure_sha256':signature,
                     'material_ids':sorted({o['material'] for o in c.objects}),'authored_parts':len(c.objects),
                     'camera_sha256':camera_sha,'trials':{m:c.family+'-'+m for m in SETS}})
    assert len(rows)==16 and len({r['geometry_fingerprint'] for r in rows})==16
    study_path='protocol/stage3-reference-v1.json'
    study={'schema':'el-stage3-reference-preregistration-1','registered_utc':time,'native_resolution':[256,256],
           'scene_rights':{'creator':'wany614','code':'Apache-2.0','original_geometry_source_media_and_allowed_reference_data':'CC-BY-4.0','privacy':'full scientific arrays private by default'},
           'cohort':{'train':8,'development':4,'test':4,'engineering_pilot':'first two training worlds only; distinct marked run identities; production scientific run uses frozen budget'},
           'capture':{'camera_sha256':camera_sha,'world_and_light_sha256':ssha,'poses':list(positions),'source_sets':SETS,'target_ids':list(TARGETS),'source_endpoint_distance_m':.72,
                      'camera':'OpenCV x-right y-down z-forward, optical-axis camera Z in metres, integer pixels; gsplat K principal point +0.5.',
                      'colour':'Blender Cycles Standard, no Look, exposure 0 gamma 1 sRGB. EXR linear combined to FP32 display-sRGB before 8-bit source encoding.'},
           'hypotheses':[{'prediction':'At fixed endpoint span, M4 may reduce family-wise target RGB error relative to M2 by resolving ambiguous layers.','confidence':'moderate','refutation':'Nonnegative M4-minus-M2 RGB incidence or MAE in the four-family macro comparison.'},
                         {'prediction':'Thin near-field occupiers can remain erroneous despite extra middle sources.','confidence':'moderate','refutation':'No residual edge or disoccluded errors in registered heldout evaluations.'},
                         {'prediction':'Additional images under an equal total update budget may impair some well-observed surfaces.','confidence':'low','refutation':'Every family improves for all registered RGB-error outcomes.'}],
           'visibility':'Per-source Blender opaque first-hit with 0 invalid/background 1 observed 2 disoccluded 3 outside; retain reason and first hit ID. Edge is object/surface boundary dilated by two pixels, independent of source count.',
           'boundary':'A reconstruction reads only its own source PNGs and source poses. A separate renderer reads frozen checkpoint and allowed poses. Reference truth is private evaluation data.',
           'display':'v01 fixed. Six target identifiers, all six and possible failures retained in evaluation.',
           'test_gate':'Before registered joint reconstruction and risk freeze, the test entrance cannot instantiate a test world, render test truth, create a test job, or resolve selected test data.'}
    study_sha=store(assets/study_path,study)
    catalog={'schema':'el-stage3-family-roster-1','dataset_id':'evidence-lens-stage3-multiview-v1','registered_utc':time,
             'preregistration_path':study_path,'preregistration_sha256':study_sha,'camera_path':camera_path,'camera_sha256':camera_sha,
             'reference_settings_path':settings_path,'reference_settings_sha256':ssha,'source_ids':list(SOURCES),'source_sets':SETS,
             'target_ids':list(TARGETS),'primary_view':'v01','pilot_families':[r['scene_family_id'] for r in rows if r['pilot']],
             'test_families':[r['scene_family_id'] for r in rows if r['split']=='test'],'families':rows}
    return store(assets/'protocol/stage3-families-v1.json',catalog)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--registered-utc',required=True)
    a=p.parse_args()
    print(register(a.assets.resolve(),a.project.resolve(),a.registered_utc))
