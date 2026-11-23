# Tabletop plane example

![The 128-pixel source view of the authored tabletop](input.png)

The camera file describes one input and six nearby target views. The image is a downsample of the project's first procedural Blender scene. The CPU example adapter in `scripts/example_plane.py` uses this image as a texture on one opaque plane at a hypothesized camera depth of two metres. It produces RGB, camera-Z and accumulated opacity by ray–plane intersection. The adapter reads only the source image and cameras. Its output illustrates the inference interface outside the generator used in the formal study.

The model release supplies the derived seven-view arrays, expected numerical scores and grading commands. The original scene render is `20261123T023609071040Z-9fad7aa54fc84008`; the example derivation is `20261123T034606860802Z-acb5e6955f424043`. The accompanying provenance file identifies the source PNG and camera hashes.
