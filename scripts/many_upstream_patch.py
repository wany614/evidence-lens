"""Apply a content-pinned headless adapter to the official simple_trainer; emit a readable unified patch."""

import ast
import difflib
import hashlib


TRAINER_SHA = "79319e1cd7404e4d1ba0c425634235c39e6054f0643b904a01feea6179462c05"
UTILS_SHA = "44a38b236940706705c9351b365be6bd21d49efc57dd695ba5a7e70055b959e1"


def program(trainer_raw, utils_raw):
    if hashlib.sha256(trainer_raw).hexdigest() != TRAINER_SHA or hashlib.sha256(utils_raw).hexdigest() != UTILS_SHA:
        raise ValueError("official trainer or utilities differ from tagged version")
    original = trainer_raw.decode()
    adapted = "from __future__ import annotations\n" + original
    top = adapted.index("import imageio\n")
    bottom = adapted.index("\n\n@dataclass", top)
    adapted = adapted[:top] + """import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import yaml
from torch import Tensor
from typing import Literal, assert_never
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.optimizers import SelectiveAdam
from upstream_math import rgb_to_sh, set_random_seed
from many_upstream_native import Parser, Dataset, exact_knn as knn, valid_ssim as fused_ssim
""" + adapted[bottom:]
    # Optional evaluation metrics would otherwise load pretrained LPIPS on construction.
    a, b = adapted.index("        # Losses & Metrics."), adapted.index("    def rasterize_splats(")
    adapted = adapted[:a] + adapted[b:]
    changes = (
        ('self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")', 'self.writer = None'),
        ('self.valset = Dataset(self.parser, split="val")', 'self.valset = None'),
        ("num_workers=4,\n            persistent_workers=True,", "num_workers=0,\n            persistent_workers=False,"),
        ("            pbar.set_description(desc)", "            pbar.set_description(desc, refresh=False)\n"
         "            self.trainset.record_step(step, data['image_id'], loss, l1loss, ssimloss, len(self.splats['means']))"),
    )
    for old, new in changes:
        if adapted.count(old) != 1:
            raise ValueError("cannot apply the registered headless trainer edit: " + old)
        adapted = adapted.replace(old, new)
    # CLI, viewer, evaluator and compression methods remain inert: the worker calls only the audited Runner.train branch.
    ast.parse(adapted)
    util = utils_raw.decode()
    nodes = [ast.get_source_segment(util,n) for n in ast.parse(util).body if isinstance(n,ast.FunctionDef) and
             n.name in ("rgb_to_sh","set_random_seed")]
    if len(nodes) != 2:
        raise ValueError("official scalar SH/seed helpers missing")
    helpers = "import random\nimport numpy as np\nimport torch\nfrom torch import Tensor\n\n"+"\n\n".join(nodes)+"\n"
    patch = "".join(difflib.unified_diff(original.splitlines(True), adapted.splitlines(True),
                                          fromfile="official/examples/simple_trainer.py",tofile="runtime/simple_trainer.py"))
    return adapted.encode(), helpers.encode(), patch.encode()
