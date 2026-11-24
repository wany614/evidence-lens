"""Tensor-only DINOv2 ViT-S/14 patch inference and target-pixel risk scoring.

Backbone names and interpolation are compatible with Meta's plain (zero-register)
dinov2_vits14, Apache-2.0. The base verifier compares every loaded tensor and
official patch outputs. Only the projection and risk head are new project layers.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

GRID = 252
PATCH = 14
DIM = 384
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
ARCH = {"schema": "el-dino-risk-architecture-1", "base": "dinov2_vits14", "blocks": 12, "width": DIM,
        "heads": 6, "registers": 0, "patch_size": PATCH, "pretrain_image": 518, "encoder_image": GRID,
        "projection_channels": 32, "head_hidden": 32, "legal_features": 26, "normalization_mean": list(MEAN),
        "normalization_std": list(STD), "mapping": "letterbox_native_pixel_centres_weighted_patch_grid_v1"}


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(DIM, 3 * DIM)
        self.proj = nn.Linear(DIM, DIM)

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, 6, 64).permute(2, 0, 3, 1, 4).unbind(0)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            attended = F.scaled_dot_product_attention(q, k, v)
        return self.proj(attended.transpose(1, 2).reshape(b, n, DIM))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DIM, DIM * 4)
        self.fc2 = nn.Linear(DIM * 4, DIM)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(DIM))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM, eps=1e-6)
        self.attn = Attention()
        self.ls1 = LayerScale()
        self.norm2 = nn.LayerNorm(DIM, eps=1e-6)
        self.mlp = MLP()
        self.ls2 = LayerScale()

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, DIM, PATCH, stride=PATCH)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, 37 * 37 + 1, DIM))
        self.mask_token = nn.Parameter(torch.zeros(1, DIM))
        self.blocks = nn.ModuleList(Block() for _ in range(12))
        self.norm = nn.LayerNorm(DIM, eps=1e-6)

    def prefix(self, image):
        if image.shape[1:] != (3, GRID, GRID):
            raise ValueError("encoder expects normalized [batch,3,252,252]")
        tokens = self.patch_embed(image)
        # The official non-register model uses a historical +0.1 scale offset.
        size = GRID // PATCH
        scale = (size + 0.1) / 37
        patch_positions = self.pos_embed[:, 1:].float().reshape(1, 37, 37, DIM).permute(0, 3, 1, 2)
        positions = F.interpolate(patch_positions, scale_factor=(scale, scale), mode="bicubic",
                                  align_corners=False, antialias=False).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls_token.expand(tokens.shape[0], -1, -1), tokens), 1)
        tokens = tokens + torch.cat((self.pos_embed[:, :1], positions), 1).to(tokens.dtype)
        for block in self.blocks[:10]:
            tokens = block(tokens)
        return tokens

    def suffix(self, tokens):
        for block in self.blocks[10:]:
            tokens = block(tokens)
        return self.norm(tokens)[:, 1:]

    def forward(self, image):
        return self.suffix(self.prefix(image))


@dataclass(frozen=True)
class Frame:
    height: int
    width: int
    scaled_height: int
    scaled_width: int
    top: int
    left: int

    @classmethod
    def from_shape(cls, height, width):
        if height < 1 or width < 1:
            raise ValueError("target must have a nonempty native grid")
        scale = GRID / max(height, width)
        h, w = (max(1, min(GRID, math.floor(n * scale + 0.5))) for n in (height, width))
        return cls(height, width, h, w, (GRID-h)//2, (GRID-w)//2)

    def json(self):
        return dict(asdict(self), encoder_pixels=GRID*GRID, observed_encoder_pixels=self.scaled_height*self.scaled_width,
                    patch_size=PATCH, pixel_coordinates="native centres (x+.5,y+.5)", camera_grid="unchanged")

    def grid(self, indices, device):
        ids = torch.as_tensor(indices, dtype=torch.long, device=device)
        x = self.left + (ids.remainder(self.width).float() + .5) * self.scaled_width / self.width
        y = self.top + (torch.div(ids, self.width, rounding_mode="floor").float() + .5) * self.scaled_height / self.height
        return (torch.stack((x, y), -1) * (2 / GRID) - 1).reshape(1, -1, 1, 2)


def finite_rgb_channels(rgb):
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.float32:
        raise ValueError("expected float32 H×W×3 display-sRGB")
    finite_rgb = np.isfinite(rgb)
    if np.any((rgb[finite_rgb] < 0) | (rgb[finite_rgb] > 1)):
        raise ValueError("finite target RGB outside display-sRGB [0,1]")
    return finite_rgb


def image_tensor(rgb, device):
    finite_rgb = finite_rgb_channels(rgb)
    frame = Frame.from_shape(*rgb.shape[:2])
    observed = np.all(finite_rgb, axis=-1)
    array = np.where(observed[...,None], rgb, 0).astype(np.float32, copy=False)
    native = torch.from_numpy(np.ascontiguousarray(array)).to(device).permute(2,0,1)[None]
    resized = F.interpolate(native, size=(frame.scaled_height,frame.scaled_width),
                            mode="bilinear", align_corners=False, antialias=True)
    mean = native.new_tensor(MEAN).reshape(1,3,1,1)
    std = native.new_tensor(STD).reshape(1,3,1,1)
    encoded = native.new_zeros((1,3,GRID,GRID))
    rectangle = (slice(frame.top,frame.top+frame.scaled_height), slice(frame.left,frame.left+frame.scaled_width))
    encoded[:,:,rectangle[0],rectangle[1]] = (resized-mean) / std
    mask = native.new_zeros((1,1,GRID,GRID))
    mask[:,:,rectangle[0],rectangle[1]] = 1
    return encoded, F.avg_pool2d(mask, PATCH, PATCH), frame, observed


class RiskModel(nn.Module):
    def __init__(self, features=26):
        super().__init__()
        self.backbone = Backbone()
        self.projection = nn.Linear(DIM, 32)
        self.head = nn.Sequential(nn.Linear(32 + features, 32), nn.GELU(), nn.Linear(32, 1))

    def configure(self, branch):
        if branch not in ("L2-frozen","L2-vision-ft"):
            raise ValueError("choose L2-frozen or L2-vision-ft")
        self.requires_grad_(True)
        self.backbone.requires_grad_(False)
        if branch == "L2-vision-ft":
            self.backbone.blocks[10:].requires_grad_(True)
            self.backbone.norm.requires_grad_(True)
        self.backbone.eval()  # fixed inference semantics for both branches; gradients are controlled separately

    def project(self, image):
        with torch.no_grad():
            prefix = self.backbone.prefix(image)
        tokens = self.backbone.suffix(prefix)
        size = GRID//PATCH
        return self.projection(tokens).transpose(1,2).reshape(1,32,size,size)

    def at(self, projected, support, frame, indices, numeric):
        grid = frame.grid(indices, projected.device)
        # Four border-clamped patch neighbours implement align_corners=False sampling.
        # index_select has a deterministic CUDA backward for repeated patch IDs.
        size = GRID // PATCH
        xy = ((grid.reshape(-1,2) + 1) * (size/2) - .5).clamp(0,size-1)
        corner = xy.floor().long()
        hi = (corner+1).clamp_max(size-1)
        dx,dy = (xy-corner).unbind(1)
        x0,y0 = corner.unbind(1)
        x1,y1 = hi.unbind(1)
        ids = torch.stack((y0*size+x0,y0*size+x1,y1*size+x0,y1*size+x1),1)
        coeff = torch.stack(((1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy),1)
        coeff = coeff * support.flatten().index_select(0,ids.flatten()).reshape_as(coeff)
        fraction = coeff.sum(1)
        patches = projected[0].flatten(1).T.index_select(0,ids.flatten()).reshape(len(indices),4,32)
        features = (patches*coeff.unsqueeze(-1)).sum(1) / fraction.clamp_min(1e-8).unsqueeze(-1)
        return self.head(torch.cat((features,numeric),1)).flatten(), fraction > 0


@dataclass(frozen=True)
class RectangularFrame:
    """Native analysis pixels at the upper left of a minimally padded ViT input."""

    height: int
    width: int

    @property
    def patch_shape(self):
        return (self.height + PATCH - 1) // PATCH, (self.width + PATCH - 1) // PATCH

    @property
    def encoded_shape(self):
        rows, columns = self.patch_shape
        return rows * PATCH, columns * PATCH

    def patch_coordinates(self, indices, device):
        ids = torch.as_tensor(indices, dtype=torch.long, device=device)
        if ids.ndim != 1 or torch.any((ids < 0) | (ids >= self.height * self.width)):
            raise ValueError("indices must identify pixels on the unpadded analysis grid")
        x = (ids.remainder(self.width).float() + 0.5) / PATCH - 0.5
        y = (torch.div(ids, self.width, rounding_mode="floor").float() + 0.5) / PATCH - 0.5
        return x, y


def rectangular_image_tensor(rgb, device):
    observed = np.all(finite_rgb_channels(rgb), axis=-1)
    h, w = observed.shape
    if not h or not w:
        raise ValueError("analysis image must have positive height and width")
    frame = RectangularFrame(h, w)
    eh, ew = frame.encoded_shape
    # Mean-color fill gives normalized zero for an unavailable or padded RGB sample.
    filled = np.where(observed[..., None], rgb, np.asarray(MEAN, dtype=np.float32))
    native = torch.from_numpy(np.ascontiguousarray(filled)).to(device).permute(2, 0, 1)[None]
    mean, std = (native.new_tensor(v).reshape(1, 3, 1, 1) for v in (MEAN, STD))
    pad = (0, ew - w, 0, eh - h)
    image = F.pad((native - mean) / std, pad)
    mask = torch.from_numpy(observed).to(device=device, dtype=native.dtype)[None, None]
    support = F.avg_pool2d(F.pad(mask, pad), PATCH, PATCH)
    return image, support, frame, observed


def rectangular_positions(backbone, patch_rows, patch_columns):
    """Ordinary DINOv2 ViT-S/14 uses a 37×37 learned lattice and the +0.1 offset on each axis."""
    table = backbone.pos_embed
    side = math.isqrt(table.shape[1] - 1)
    if side * side != table.shape[1] - 1:
        raise ValueError("expected one CLS position and a square pretrained patch lattice")
    if patch_rows == patch_columns == side:
        return table
    patch = table[:, 1:].float().reshape(1, side, side, table.shape[-1]).permute(0, 3, 1, 2)
    patch = F.interpolate(
        patch, scale_factor=((patch_rows + 0.1) / side, (patch_columns + 0.1) / side),
        mode="bicubic", align_corners=False, antialias=False,
    )
    if patch.shape[-2:] != (patch_rows, patch_columns):
        raise ValueError("positional interpolation yielded a different patch lattice")
    return torch.cat((table[:, :1], patch.flatten(2).transpose(1, 2)), dim=1)


def rectangular_input_tokens(backbone, image):
    if image.ndim != 4 or image.shape[1] != 3 or any(n <= 0 or n % PATCH for n in image.shape[-2:]):
        raise ValueError("expected normalized NCHW RGB with patch-aligned spatial dimensions")
    rows, columns = (n // PATCH for n in image.shape[-2:])
    patches = backbone.patch_embed(image)
    cls = backbone.cls_token.expand(patches.shape[0], -1, -1)
    tokens = torch.cat((cls, patches), dim=1)
    return tokens + rectangular_positions(backbone, rows, columns).to(tokens.dtype)


def rectangular_backbone(backbone, image):
    tokens = rectangular_input_tokens(backbone, image)
    for block in backbone.blocks:
        tokens = block(tokens)
    return backbone.norm(tokens)


def rectangular_project(model, image):
    rows, columns = (n // PATCH for n in image.shape[-2:])
    # The two released models have exactly one CLS and zero register tokens.
    patches = rectangular_backbone(model.backbone, image)[:, 1:]
    projected = model.projection(patches).transpose(1, 2)
    return projected.reshape(image.shape[0], projected.shape[1], rows, columns)


def rectangular_sample(projected, support, frame, indices):
    rows, columns = frame.patch_shape
    if projected.shape[0] != 1 or projected.shape[-2:] != (rows, columns) or support.shape != (1, 1, rows, columns):
        raise ValueError("projection and support must describe the same native patch lattice")
    x, y = frame.patch_coordinates(indices, projected.device)
    x, y = x.clamp(0, columns - 1), y.clamp(0, rows - 1)
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0 + 1).clamp_max(columns - 1), (y0 + 1).clamp_max(rows - 1)
    dx, dy = x - x0, y - y0
    ids = torch.stack((y0 * columns + x0, y0 * columns + x1, y1 * columns + x0, y1 * columns + x1), 1)
    coeff = torch.stack(((1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy), 1)
    weights = coeff * support.flatten()[ids]
    fraction = weights.sum(dim=1)
    neighbors = projected[0].flatten(1).T[ids]
    sampled = (neighbors * weights.unsqueeze(-1)).sum(dim=1) / fraction.clamp_min(1e-8).unsqueeze(-1)
    return sampled, fraction
