"""Independently reopen and render the original and portable native example at registered v01."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import bpy
import numpy as np


def run(a):
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(a.helpers))
    from passes import display_srgb,png_codes,read_passes,write_json
    from scene import camera_record
    from validation import render
    cameras=json.loads(a.cameras.read_text())['views']
    cfg=json.loads(a.settings.read_text())
    phases={}
    values={}
    with tempfile.TemporaryDirectory(dir=a.scratch) as tmp:
        for tag,file in [('sealed_original',a.original),('public_copy',a.portable)]:
            bpy.ops.wm.open_mainfile(filepath=str(file),load_ui=False)
            scene=bpy.context.scene
            if (scene.render.resolution_x,scene.render.resolution_y,scene.render.resolution_percentage)!=(256,256,100):
                raise ValueError('Native image size differs')
            if len(bpy.data.libraries) or bpy.utils.blend_paths(absolute=False,packed=True,local=True):
                raise ValueError('Native example uses external files')
            poses={v:camera_record(scene,bpy.data.objects[v],info['purpose']) for v,info in cameras.items()}
            if poses!=cameras:raise ValueError('Native calibrated poses changed')
            geometry=[]
            for o in sorted(scene.objects,key=lambda x:x.name):
                if o.type!='MESH':continue
                geom=hashlib.sha256()
                for vertex in o.data.vertices:geom.update(np.asarray(vertex.co,dtype='<f4').tobytes())
                for poly in o.data.polygons:geom.update(np.asarray(poly.vertices,dtype='<i4').tobytes())
                geometry.append({'name':o.name,'object_id':o.pass_index,'matrix_world':np.asarray(o.matrix_world).tolist(),
                                 'local_mesh_sha256':geom.hexdigest(),'materials':[m.name for m in o.data.materials]})
            ghash=hashlib.sha256(json.dumps(geometry,sort_keys=True).encode()).hexdigest()
            scene.camera=bpy.data.objects['v01'];scene.cycles.seed=cfg['seed']
            f=Path(tmp)/(tag+'.exr');render(scene,f)
            linear,pos,ids,_=read_passes(f)
            values[tag]=(display_srgb(linear),pos,ids)
            phases[tag]={'mesh_count':len(geometry),'camera_count':len(poses),'object_transform_material_assignment_sha256':ghash,
                         'samples':scene.cycles.samples,'seed':scene.cycles.seed,'native_bpy_version':bpy.app.version_string,
                         'external_library_count':len(bpy.data.libraries),'external_file_count':len(bpy.utils.blend_paths(absolute=False,packed=True,local=True))}
    if phases['sealed_original']!=phases['public_copy']:raise ValueError('Reopened objects, poses or environment differ')
    a0=values['sealed_original'];b=values['public_copy']
    with np.load(a.truth,allow_pickle=False) as z:
        archived=(z['rgb'],z['position'],z['object_id']);valid=z['reference_surface']
    checks={}
    for name,x,y in [('original_vs_public',a0,b),('archived_truth_vs_public',archived,b),('archived_truth_vs_original',archived,a0)]:
        diff=np.abs(x[0].astype(np.float64)-y[0].astype(np.float64))
        mae=diff.mean(-1)
        row={'rgb_bitwise_equal':x[0].tobytes()==y[0].tobytes(),'rgb_max_abs':float(diff.max()),'rgb_mean_channel_error':float(mae.mean()),
             'pixel_mae_over_0_05_fraction':float(np.mean(mae>.05)),
             'rendered_png_max_code_difference':int(np.abs(png_codes(x[0]).astype(int)-png_codes(y[0]).astype(int)).max()),
             'surface_positions_bitwise_equal':x[1][valid].tobytes()==y[1][valid].tobytes(),'object_ids_bitwise_equal':x[2].tobytes()==y[2].tobytes()}
        if not row['surface_positions_bitwise_equal'] or not row['object_ids_bitwise_equal'] or row['rgb_mean_channel_error']>cfg['validation']['noise_mae_max'] or row['pixel_mae_over_0_05_fraction']>cfg['validation']['noise_exceedance_fraction_max']:
            raise ValueError('Matched-scene readback exceeded registered geometric or stochastic colour controls: '+name+' '+str(row))
        checks[name]=row
    write_json(a.out,{'schema':'el-stage3-public-native-v01-verification-1','passed':True,'fixed_target':'v01','runtime':phases,'comparison':checks})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('helpers','original','portable','cameras','settings','scratch','truth','out'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
