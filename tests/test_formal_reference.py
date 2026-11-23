"""Catalog and geometry regressions; these tests never render or open held-out results."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

from evidence_lens.blender_reference.formal_catalog import geometry_fingerprint, resolved, validate
from evidence_lens.blender_reference.formal_designs import authored


ROOT = Path(__file__).resolve().parents[1]
HAS_REFERENCE = all(importlib.util.find_spec(k) is not None for k in ("bpy","OpenEXR","PIL","numpy"))


class FormalCatalog(unittest.TestCase):
    def test_frozen_recipes_match_the_authored_parts(self):
        catalog, designs, base, lock = validate(ROOT)
        for composition in authored():
            actual = designs["families"][composition.family]["objects"]
            self.assertEqual(actual, json.loads(json.dumps(composition.objects)))
        for row in catalog["families"][1:]:
            cfg = resolved(row,catalog,designs,base,lock)
            self.assertEqual(cfg["reference"]["scene_name"], row["scene_family_id"]+"_v001")
            self.assertEqual(cfg["reference"]["validation"]["occlusion_profile"], "opaque_mesh_pairs")
        self.assertNotIn("occlusion_profile",base["reference"]["validation"])

    def test_identical_layout_recoloured_or_renamed_cannot_be_a_new_family(self):
        objects = [{"id":1,"name":"left","center":[-1,0,0],"material":"cream"},
                   {"id":2,"name":"right","center":[1,0,0],"material":"green"}]
        recolored = [{**obj,"id":10+i,"material":"blue","name":"renamed"+str(i)} for i,obj in enumerate(objects[::-1])]
        self.assertEqual(geometry_fingerprint(objects),geometry_fingerprint(recolored))


@unittest.skipUnless(HAS_REFERENCE,"requires isolated Blender reference environment")
class OpaqueGeometry(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0,str(ROOT/"src/evidence_lens/blender_reference"))
        self.addCleanup(sys.path.pop,0)

    def test_measurement_does_not_merge_different_blockers(self):
        import numpy as np
        from passes import geometric_disocclusion
        from verify_reference import verify_disocclusion
        obj = np.full((2,18),20,np.uint16)
        hit = np.full_like(obj,1);hit[:,8:]=2
        vis=np.full_like(obj,2,dtype=np.uint8);edge=np.zeros_like(obj,dtype=bool)
        settings={"band_min_width_px":10,"band_min_interior_pixels":20}
        report=geometric_disocclusion(obj,vis,hit,edge,settings)
        self.assertEqual([p["occluder_id"] for p in report["measurable_pairs"]],[2])
        self.assertEqual(report["measurable_pairs"][0]["max_contiguous_horizontal_width_px"],10)
        data=dict(object_id=obj,first_source_hit_id=hit,visibility=vis,edge=edge)
        self.assertEqual(verify_disocclusion(data,settings,report,"fixture",True),report)
        hit[:,8:11]=3
        self.assertFalse(geometric_disocclusion(obj,vis,hit,edge,settings)["meets_width_and_interior"])

    def test_registered_r1_ties(self):
        import numpy as np
        from passes import reduce_half
        pixels=np.array([[[0],[0],[1],[1]],[[0],[2],[1],[3]]],dtype=np.uint8)
        np.testing.assert_array_equal(reduce_half(pixels,True).ravel(),[0,2])
        np.testing.assert_array_equal(reduce_half(pixels).ravel(),[1,2])

    def test_each_new_primitive_creates_an_evaluated_mesh(self):
        import bpy
        from scene import make_object, material, evaluated_bvh
        bpy.ops.wm.read_factory_settings(use_empty=True)
        mat=material("fixture only",[.2,.3,.4],.65)
        components=[("ellipsoid", {"radii":[.14,.11,.09]}), ("rod",{"axis":[.1,.2,.3],"radius":.02}),
                    ("torus",{"major_radius":.12,"minor_radius":.015,"rotation_deg":[90,0,0]}),
                    ("frustum",{"bottom_radius":.09,"top_radius":0,"depth":.15,"vertices":4})]
        for i,(primitive,params) in enumerate(components,1):
            obj=make_object({"primitive":primitive,"center":[i,0,0],"name":"fixture "+primitive,"id":i,**params},mat)
            self.assertGreater(len(obj.data.polygons),0)
            self.assertTrue(all(math.isfinite(v) for v in obj.matrix_world.translation))
        bpy.context.view_layer.update()
        _, owners=evaluated_bvh(bpy.context.scene)
        self.assertEqual(set(owners.tolist()),{1,2,3,4})


if __name__=="__main__":
    unittest.main()
