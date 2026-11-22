from __future__ import annotations

import io
import re
from typing import Protocol

import numpy as np
from PIL import Image
from witness_client import RenderOptions

YAWS = (0, 90, 180, 270)
MASK_SIZE = 64
FRONT_THETAS = (30.0, 330.0, 0.0, 0.0)
FRONT_PHIS = (0.0, 0.0, 15.0, -30.0)
KEY = (255, 0, 255)
THIN_RATIO = 0.2


class Renderer(Protocol):
    def render(self, source: str, options: RenderOptions) -> bytes: ...


def _normalized(mask: np.ndarray) -> np.ndarray | None:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    crop = mask[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]
    height, width = crop.shape
    side = max(height, width)
    square = np.zeros((side, side), dtype=np.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    square[top: top + height, left: left + width] = crop.astype(np.uint8) * 255
    resized = Image.fromarray(square).resize((MASK_SIZE, MASK_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def reference_mask(image: bytes) -> np.ndarray | None:
    rgb = np.asarray(Image.open(io.BytesIO(image)).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR),
                     dtype=np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    background = np.median(border, axis=0)
    distance = np.sqrt(((rgb - background) ** 2).sum(axis=2))
    mask = distance > 40.0
    if mask.mean() < 0.01 or mask.mean() > 0.95:
        return None
    return _normalized(mask)


def _render_masks(renderer: Renderer, source: str) -> list[np.ndarray | None]:
    options = RenderOptions(thetas=FRONT_THETAS, phis=FRONT_PHIS, img_size=128, gap=0, bg_color="ff00ff",
                            lighting="neutral")
    grid = np.asarray(Image.open(io.BytesIO(renderer.render(source, options))).convert("RGB"), dtype=np.float32)
    masks = []
    for row in range(2):
        for col in range(2):
            tile = grid[row * 128: (row + 1) * 128, col * 128: (col + 1) * 128]
            distance = np.sqrt(((tile - np.array(KEY, dtype=np.float32)) ** 2).sum(axis=2))
            masks.append(_normalized(distance > 60.0))
    return masks


def rotated(source: str, yaw: int) -> str:
    if yaw == 0:
        return source
    call = "root.add(__asset404(THREE));"
    replacement = (f"const __yaw404 = new THREE.Group(); __yaw404.rotation.y = {yaw} * Math.PI / 180; "
                   "__yaw404.add(__asset404(THREE)); root.add(__yaw404);")
    return source.replace(call, replacement, 1)


def stood_up(source: str, pitch: int) -> str:
    call = "root.add(__asset404(THREE));"
    replacement = (f"const __pose404 = new THREE.Group(); __pose404.rotation.x = {pitch} * Math.PI / 180; "
                   "__pose404.add(__asset404(THREE)); root.add(__pose404);")
    return source.replace(call, replacement, 1)


def is_thin(source: str, renderer: Renderer, ratio: float = THIN_RATIO) -> bool:
    options = RenderOptions(thetas=(0.0, 90.0), phis=(0.0, 0.0), img_size=128, gap=0, bg_color="ff00ff",
                            lighting="neutral")
    grid = np.asarray(Image.open(io.BytesIO(renderer.render(source, options))).convert("RGB"), dtype=np.float32)
    distance = np.sqrt(((grid - np.array(KEY, dtype=np.float32)) ** 2).sum(axis=2)) > 60.0
    front, side = distance[:, :128], distance[:, 128:]
    if not front.any() or not side.any():
        return False
    height = np.ptp(np.flatnonzero(front.any(axis=1))) + 1
    width = np.ptp(np.flatnonzero(front.any(axis=0))) + 1
    depth = np.ptp(np.flatnonzero(side.any(axis=0))) + 1
    return bool(height <= min(width, depth) and height < ratio * max(width, depth))


def silhouette_scores(image: bytes, source: str, renderer: Renderer) -> dict[int, float] | None:
    target = reference_mask(image)
    if target is None:
        return None
    scores: dict[int, float] = {}
    for yaw in YAWS:
        values = []
        for mask in _render_masks(renderer, rotated(source, yaw)):
            if mask is None:
                continue
            intersection = np.minimum(mask, target).sum()
            union = np.maximum(mask, target).sum()
            values.append(float(intersection / union) if union else 0.0)
        scores[yaw] = round(sum(values) / len(values), 4) if values else 0.0
    return scores


def best_yaw(image: bytes, source: str, renderer: Renderer, margin: float = 0.04) -> int:
    scores = silhouette_scores(image, source, renderer)
    if not scores:
        return 0
    yaw = max((90, 270), key=lambda y: (scores[y], -y))
    return yaw if scores[yaw] - scores[0] >= margin else 0



_ADD = r"\b{name}\.add\(\s*([A-Za-z_$][\w$]*)\s*\)"
_TURNS = (r"\b{name}\.rotation\.[xyz]\s*=\s*[^;\n]+", r"\b{name}\.rotation\.set\([^;\n]*\)",
          r"\b{name}\.rotate[XYZ]\([^;\n]*\)", r"\b{name}\.(?:quaternion|matrix)\b[^;\n]*")
_GENERATE = re.compile(r"export\s+default\s+function\s*(?:[A-Za-z_$][\w$]*\s*)?\([^)]*\)\s*\{")


def _generate_body(raw: str) -> str:
    found = _GENERATE.search(raw)
    if found is None:
        return ""
    depth, start = 1, found.end()
    for position in range(start, len(raw)):
        if raw[position] == "{":
            depth += 1
        elif raw[position] == "}":
            depth -= 1
            if depth == 0:
                return raw[start:position]
    return raw[start:]


def _top_returns(body: str) -> list[str]:
    names, depth = [], 0
    for token in re.finditer(r"[{}]|\breturn\s+([A-Za-z_$][\w$]*)\s*;", body):
        if token.group(0) == "{":
            depth += 1
        elif token.group(0) == "}":
            depth -= 1
        elif depth == 0:
            names.append(token.group(1))
    return names


def pose_signature(raw: str) -> tuple[str, ...]:
    body = _generate_body(raw)
    names = _top_returns(body)
    if not names:
        return ()
    root = names[-1]
    raw = body
    children = re.findall(_ADD.format(name=re.escape(root)), raw)
    targets = [("root", root)] + ([("outer", children[0])] if len(set(children)) == 1 else [])
    found = []
    for role, name in targets:
        for pattern in _TURNS:
            for match in re.finditer(pattern.format(name=re.escape(name)), raw):
                found.append(role + ":" + re.sub(r"\s+", "", match.group(0).split(".", 1)[1]))
    return tuple(sorted(found))
