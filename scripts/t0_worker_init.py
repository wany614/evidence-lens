"""T0 source initialization and the one registered D-only forced fallback probe."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

import recovery_triangulate as geometry


def initialize(args):
    g = geometry
    cfg = g.load(args.config)
    if not cfg['forced_fallback_fixture']:
        # Run the unchanged qualified/unqualified source decision and all its original statistics.
        g.run(args)
        return
    started = time.perf_counter()
    g.require((cfg['scene_family_id'], cfg['condition_id'], cfg['trial_id']) ==
              ('set1', 'M2', 't0-d-forced-fallback-set1-M2-init'), 'Forced fallback is reserved for one D cell')
    ids = g.SETS['M2']
    protocol = g.load(args.protocol)
    g.require(g.sha(args.protocol) == cfg['initialization_protocol_sha256'] == g.PROTOCOL_SHA and
              g.sha(args.cameras) == cfg['source_camera_sha256'] and tuple(cfg['generator_source_ids']) == ids,
              'Changed D initialization contract')
    versions = {n: importlib.metadata.version(n) for n in ('opencv-python-headless', 'numpy', 'scipy', 'pillow')}
    g.require(all(versions[n] == protocol['runtime'][n.replace('-', '_')] for n in versions) and
              cv2.__version__ == protocol['runtime']['opencv'], 'Different frozen geometry runtime')
    g.require([s for s, _ in args.source] == list(ids), 'Expected the two ordered D source photographs')
    images = {}
    for sid, path in args.source:
        g.require(g.sha(path) == cfg['source_rgb_sha256'][sid], 'Wrong D source photograph')
        with Image.open(path) as image:
            g.require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (256, 256), 'Invalid source image')
            images[sid] = np.asarray(image)
    rig = g.cameras(g.load(args.cameras), ids)
    tracks, scale, quality, matching = g.triangulate(images, rig, ids)
    original = g.load(args.previous)
    g.require(quality == original['eligibility'] and original['source_camera_sha256'] == cfg['source_camera_sha256'] and
              original['source_rgb_sha256'] == cfg['source_rgb_sha256'], 'Natural D eligibility differs from original')
    points, fallback = g.random_prior(rig, ids, scale)
    g.require(len(points) == fallback['total'] == 2048 and fallback['samples_per_endpoint'] == 1024,
              'Original endpoint fallback count differs')
    arrays, parameter_stats = g.parameters(points, images, rig, ids, scale)
    g.details(args.out / 'source-tracks.npz', tracks, ids)
    np.savez(args.out / 'init.npz', **arrays)
    g.write(args.out / 'init.json', {
        'schema': 'el-recovery-init-1', 'run_id': os.environ['EL_RUN_ID'], 'object_id': 'set1', 'condition': 'M2',
        'registration': cfg['registration'], 'recipe_sha256': cfg['recipe_sha256'], 'source_parent': cfg['source_parent'],
        'initialization_protocol_sha256': g.PROTOCOL_SHA, 'generator_source_ids': list(ids),
        'source_camera_sha256': cfg['source_camera_sha256'], 'source_rgb_sha256': cfg['source_rgb_sha256'],
        'scene_scale_m': scale, 'shared_endpoints': list(ids), 'actual_method': 'endpoint_stratified_inverse_depth',
        'N': len(points), 'parameter_names': list(g.NAMES), 'parameter_sha256': g.parameter_digest(arrays),
        'init_npz_sha256': g.sha(args.out / 'init.npz'), 'init_npz_bytes': (args.out / 'init.npz').stat().st_size,
        'source_tracks_sha256': g.sha(args.out / 'source-tracks.npz'), 'eligibility': quality, 'matching': matching,
        'parameters': parameter_stats, 'fallback': fallback, 'forced_fallback_fixture': True,
        'point_evidence': '2048 camera-ray inverse-depth hypotheses', 'scientific_sample': False,
        'versions': versions, 'elapsed_seconds': time.perf_counter() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'protocol', 'cameras', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--source', nargs=2, action='append', required=True)
    initialize(parser.parse_args())
