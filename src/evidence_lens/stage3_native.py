"""Blender 4.5 native ten-camera reference generation and independent sealed readback."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import bpy
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).parent/'blender_reference'))
from passes import REASONS, display_srgb, edges, png_codes, png_pixels, read_passes, save_png, sha256, visibility, write_json
from scene import build_world, camera_bundle, camera_record, evaluated_bvh, ocio_config_path, project, render_settings, validate_camera
from validation import calibration_fixture, render, seed_comparison

SOURCES=('s00','s01','s02','s03')
TARGETS=tuple(f'v{i:02d}' for i in range(1,7))
SETS={'M2':('s00','s03'),'M4':SOURCES}
VIEWS=SOURCES+TARGETS


def require(test, reason):
    if not test:
        raise ValueError(reason)


def spec(path):
    return {'bytes':path.stat().st_size,'sha256':sha256(path)}


def checked(path, expected):
    require(path.is_file() and spec(path)==expected,'Changed scientific member '+path.name)
    return path


def numerical_geometry(pos, obj, view, views, tree, tol):
    uv,z=project(pos,view)
    surface=(obj>0)&np.isfinite(pos).all(-1)&np.isfinite(z)&(z>0)
    require(np.any(surface),'Missing native mesh surface')
    yy,xx=np.indices(obj.shape)
    reproj=np.hypot(uv[...,0]-xx,uv[...,1]-yy)
    maximum=float(np.max(reproj[surface]))
    require(maximum<=tol['render_pixel_tolerance_px'],'Position pass misses integer pixel centres')
    data={'position':np.where(surface[...,None],pos,np.nan).astype(np.float32),
          'z':np.where(surface,z,np.nan).astype(np.float32),'object_id':obj,'reference_surface':surface,
          'edge':edges(obj,surface,np.zeros_like(obj),tol['edge_radius_px'])}
    counts={}
    for s in SOURCES:
        hit,classes,reason,owner=visibility(pos,obj,z,views[s],tree,tol)
        require(np.array_equal(hit,surface),'Source changed target surface validity')
        data['visibility_'+s]=classes
        data['visibility_reason_'+s]=reason
        data['first_source_hit_id_'+s]=owner
        counts[s]={'classes':np.bincount(classes.ravel(),minlength=4).tolist(),
                   'reasons':np.bincount(reason.ravel(),minlength=10).tolist()}
    return data,{'max_reprojection_px':maximum,'surface':int(surface.sum()),'visibility':counts}


def produce(cfg_file,settings_file,recipe_file,cameras_file,out):
    cfg=json.loads(cfg_file.read_text())
    s=json.loads(settings_file.read_text())
    recipe=json.loads(recipe_file.read_text())
    require(bpy.app.version==(4,5,16) and s['width']==s['height']==256,'Unexpected native runtime or image grid')
    require(sha256(recipe_file)==cfg['scene_hash'] and sha256(cameras_file)==cfg['camera_set_hash'],'Native preregistered inputs differ')
    require(not any(out.iterdir()),'Output must be empty')
    real_cameras=json.loads(cameras_file.read_text())
    s['objects']=s['objects']+recipe['objects']
    s['scene_name']=cfg['scene_family_id']+'_v001'
    for name in ('sources','scene','reference/linear','validation','condition/M2','condition/M4'):
        (out/name).mkdir(parents=True,exist_ok=True)
    tol=s['validation']
    fixture=calibration_fixture(s,out/'validation')
    require(all(fixture['tests'].values()),'Native known calibration fixture failed')
    write_json(out/'validation/calibration.json',fixture)
    original=s['camera']['positions']
    s['camera']['positions']={('source' if k=='s00' else k):v for k,v in original.items()}
    native,cams=build_world(s)
    cams['s00']=cams.pop('source')
    cams['s00'].name='s00'
    cams['s00'].data.name='s00'
    bpy.context.view_layer.update()
    views={k:camera_record(native,cams[k],'source' if k in SOURCES else 'research_target') for k in VIEWS}
    require(camera_bundle(views)==real_cameras,'Created native poses differ from preregistration')
    shutil.copyfile(cameras_file,out/'cameras.json')
    for condition,ids in SETS.items():
        for name,keys in [('source_cameras',ids),('render_cameras',ids+TARGETS)]:
            write_json(out/'condition'/condition/(name+'.json'),camera_bundle({i:views[i] for i in keys}))
    blender=out/'scene'/(s['scene_name']+'.blend')
    bpy.ops.wm.save_as_mainfile(filepath=str(blender),compress=True,copy=True,relative_remap=False)
    runtime=render_settings(native)
    runtime.update(ocio_config_sha256=sha256(ocio_config_path()),scene_seed=s['seed'],repeat_seed=s['repeat_seed'])
    write_json(out/'scene/runtime.json',runtime)
    shutil.copyfile(recipe_file,out/'scene/recipe.json')
    shutil.copyfile(settings_file,out/'scene/settings.json')
    for name in ('stage3_native.py','scene.py','passes.py','validation.py'):
        here=Path(__file__).parent
        shutil.copyfile(here/name if name=='stage3_native.py' else here/'blender_reference'/name,out/'scene'/name)
    points=[x['center'] for x in s['objects']]
    geometry={}
    records={}
    tree=evaluated_bvh(native)
    job=native.copy()
    source_check={}
    for k in VIEWS:
        camcheck=validate_camera(native,cams[k],views[k],points)
        require(camcheck['max_error_px']<=tol['projection_tolerance_px'] and camcheck['right_moves_right'] and
                camcheck['up_moves_up'] and camcheck['forward_increases_z'],'Camera orientation/calibration failed')
        exr=out/'reference/linear'/(k+'.exr')
        png=out/'reference'/(k+'.png')
        blender_png=out/'validation'/('native-'+k+'.png') if k in SOURCES else None
        job.camera=cams[k]
        job.cycles.seed=s['seed']
        render(job,exr,blender_png)
        linear,pos,obj,channels=read_passes(exr)
        rgb=display_srgb(linear)
        require(rgb.shape==(256,256,3) and np.isfinite(rgb).all() and np.isfinite(linear).all(),'Invalid native colour')
        data,metrics=numerical_geometry(pos,obj,views[k],views,tree,tol)
        geometry[k]={**metrics,'native_camera':camcheck}
        if k in SOURCES:
            self_seen=metrics['visibility'][k]['classes'][1]/metrics['surface']
            require(self_seen>=tol['source_min_observed_fraction'],'Source self-visibility below registered floor')
        dst=out/'reference'/(k+'.npz')
        np.savez_compressed(dst,rgb=rgb,**data)
        save_png(png,rgb)
        records[k]={'npz':spec(dst),'png':spec(png),'exr':spec(exr),'exr_channels':channels}
        if blender_png:
            source=out/'sources'/(k+'.png')
            shutil.copyfile(png,source)
            max_code=int(np.abs(png_pixels(source).astype(int)-png_pixels(blender_png).astype(int)).max())
            require(max_code<=tol['display_tolerance_code_values'],'Source vs native display transform differs')
            source_check[k]={'source':spec(source),'native_display_max_code_difference':max_code}
        if k=='v01':
            repeat=out/'validation/second-seed-v01.exr'
            job.cycles.seed=s['repeat_seed']
            render(job,repeat)
            second=display_srgb(read_passes(repeat)[0])
            noise=seed_comparison(rgb,second,tol)
            require(noise['passed'],'Two-seed native colour variation exceeds registered limits')
    validation={'schema':'el-stage3-reference-validation-1','passed':True,'views':geometry,'sources':source_check,
                'fixture':fixture['tests'],'noise_v01':noise,'thresholds':tol,'visibility_reasons':REASONS,
                'noise_scope':'Different path seeds with the same denoiser measure stochastic variation.'}
    write_json(out/'validation/results.json',validation)
    write_json(out/'reference.json',{'schema':'el-stage3-reference-1','run_id':cfg.get('run_id') or __import__('os').environ['EL_RUN_ID'],
             'family':cfg['scene_family_id'],'catalog_sha256':cfg['dataset_hash'],'camera_sha256':sha256(cameras_file),
             'source_ids':SOURCES,'target_ids':TARGETS,'source_sets':SETS,'views':records,'source_images':source_check,
             'condition_cameras':{c:{n:spec(out/'condition'/c/(n+'.json')) for n in ('source_cameras','render_cameras')} for c in SETS},
             'blend':{'path':blender.relative_to(out).as_posix(),**spec(blender)},'validation':spec(out/'validation/results.json'),
             'display_srgb':'FP32 native Standard display [0,1]','depth':'positive optical-axis camera Z metres; invalid NaN',
             'visibility':'visibility_sNN: 0 invalid/background,1 observed,2 disoccluded,3 outside. Indexed by real source first hit.'})


def readback(root,out):
    meta=json.loads((root/'reference.json').read_text())
    receipt=json.loads(checked(root/'validation/results.json',meta['validation']).read_text())
    require(receipt['passed'] and all(receipt['fixture'].values()),'Missing native geometric checks')
    tol=receipt['thresholds']
    cameras=json.loads((root/'cameras.json').read_text())
    require(sha256(root/'cameras.json')==meta['camera_sha256'] and set(cameras['views'])==set(VIEWS),'Native camera roster changed')
    blend=checked(root/meta['blend']['path'],{k:meta['blend'][k] for k in ('bytes','sha256')})
    bpy.ops.wm.open_mainfile(filepath=str(blend),load_ui=False)
    native=bpy.context.scene
    require(native.render.resolution_x==native.render.resolution_y==256 and native.render.resolution_percentage==100,'Saved blend grid changed')
    tree=evaluated_bvh(native)
    report={}
    for name in VIEWS:
        v=cameras['views'][name]
        require(camera_record(native,bpy.data.objects[name],v['purpose'])==v,'Reopened blend cameras differ')
        record=meta['views'][name]
        truth=checked(root/'reference'/(name+'.npz'),record['npz'])
        source_png=checked(root/'reference'/(name+'.png'),record['png'])
        exr=checked(root/'reference/linear'/(name+'.exr'),record['exr'])
        linear,pos,obj,_=read_passes(exr)
        rgb=display_srgb(linear)
        with np.load(truth,allow_pickle=False) as a:
            surf=a['reference_surface']
            require(np.array_equal(png_pixels(source_png),png_codes(rgb)) and np.array_equal(a['rgb'],rgb) and
                    np.array_equal(a['object_id'],obj) and np.array_equal(a['position'][surf],pos[surf]),'Native numerical channels differ')
            require(np.isnan(a['position'][~surf]).all() and np.isnan(a['z'][~surf]).all(),'Invalid geometry sentinel changed')
            w2c=np.linalg.inv(np.array(v['c2w']))
            cv=np.einsum('ab,hwb->hwa',w2c[:3,:3],pos)+w2c[:3,3]
            zz=cv[...,2]
            require(float(np.max(np.abs(a['z'][surf]-zz[surf])))<1e-5,'Metric optical depth differs')
            yy,xx=np.nonzero(surf)
            xy=cv[surf]@np.asarray(v['K']).T
            err=float(np.hypot(xy[:,0]/xy[:,2]-xx,xy[:,1]/xy[:,2]-yy).max())
            require(err<=tol['render_pixel_tolerance_px'],'Position reprojection exceeds registered tolerance')
            multiview={}
            for s in SOURCES:
                codes=a['visibility_'+s]
                reasons=a['visibility_reason_'+s]
                owners=a['first_source_hit_id_'+s]
                require(np.array_equal(codes,np.take(np.array([0,1,2,3,3,0,0,0,0,0],np.uint8),reasons)),'Class/reason mismatch')
                _,retry,rc,rh=visibility(pos[::13,::11],obj[::13,::11],zz[::13,::11],cameras['views'][s],tree,tol)
                require(np.array_equal(retry,codes[::13,::11]) and np.array_equal(rc,reasons[::13,::11]) and
                        np.array_equal(rh,owners[::13,::11]),'Independent mesh-ray readback differs')
                multiview[s]={'check_samples':int(retry.size),'classes':np.bincount(codes.ravel(),minlength=4).tolist(),
                              'unresolved_surface':int((surf & (codes==0)).sum())}
            if name in SOURCES:
                inp=checked(root/'sources'/(name+'.png'),meta['source_images'][name]['source'])
                require(inp.read_bytes()==source_png.read_bytes(),'Source image changed')
            report[name]={'max_position_reprojection_px':err,'valid_rgb':int(np.isfinite(rgb).all(-1).sum()),
                          'surface':int(surf.sum()),'visibility':multiview}
    for c,ids in SETS.items():
        for tag,names in [('source_cameras',ids),('render_cameras',ids+TARGETS)]:
            saved=checked(root/'condition'/c/(tag+'.json'),meta['condition_cameras'][c][tag])
            require(json.loads(saved.read_text())==camera_bundle({k:cameras['views'][k] for k in names}),'Wrong camera subset')
    write_json(out/'readback.json',{'schema':'el-stage3-native-readback-1','passed':True,'family':meta['family'],
               'reference_run_id':meta['run_id'],'reference_sha256':sha256(root/'reference.json'),'camera_sha256':meta['camera_sha256'],'views':report})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('produce','readback'))
    for name in ('out','config','settings','recipe','cameras','reference'):
        p.add_argument('--'+name,type=Path,required=name=='out')
    a=p.parse_args()
    if a.action=='produce':
        produce(a.config,a.settings,a.recipe,a.cameras,a.out)
    else:
        readback(a.reference,a.out)
