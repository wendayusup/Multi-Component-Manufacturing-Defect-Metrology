"""
Defect Localization Module using Pure-NumPy Grad-CAM.
Identifies and localizes surface defects without external object detection frameworks.
Generates bounding boxes, defect hotspot center dots, perimeter points,
annotated overlay images, and 128x128 Surface Anomaly Matrix heatmaps using Pillow + NumPy.
"""
import io
import base64
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cnn.model import Sequential


# ---------------------------------------------------------------------------
# Jet Colormap (blue → cyan → green → yellow → red)
# ---------------------------------------------------------------------------

def _jet_colormap(t: float) -> Tuple[int, int, int]:
    """Maps a scalar t in [0, 1] to jet RGB (0-255 each)."""
    r = min(1.0, max(0.0, 1.5 - abs(4.0 * t - 3.0)))
    g = min(1.0, max(0.0, 1.5 - abs(4.0 * t - 2.0)))
    b = min(1.0, max(0.0, 1.5 - abs(4.0 * t - 1.0)))
    return (int(r * 255), int(g * 255), int(b * 255))


def _apply_jet_to_array(arr_normalized: np.ndarray) -> np.ndarray:
    """Apply jet colormap to a 2D [0,1] array → (H, W, 3) uint8 RGB."""
    h, w = arr_normalized.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for iy in range(h):
        for ix in range(w):
            rgb[iy, ix] = _jet_colormap(float(arr_normalized[iy, ix]))
    return rgb


# ---------------------------------------------------------------------------
# BFS Connected Component Discovery
# ---------------------------------------------------------------------------

def get_connected_components(mask: np.ndarray, min_area: int = 50) -> List[List[Tuple[int, int]]]:
    """
    Discovers connected positive regions in a binary mask using 8-connectivity.
    Pure NumPy and Python BFS without OpenCV.
    """
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if mask[r, c] and not visited[r, c]:
                comp = []
                queue = [(r, c)]
                visited[r, c] = True

                while queue:
                    curr_r, curr_c = queue.pop(0)
                    comp.append((curr_r, curr_c))

                    # 8-connectivity neighbors
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                queue.append((nr, nc))

                if len(comp) >= min_area:
                    components.append(comp)

    return components


# ---------------------------------------------------------------------------
# Grad-CAM Core
# ---------------------------------------------------------------------------

def _compute_gradcam_32(model: Sequential) -> Optional[np.ndarray]:
    """
    Performs backward pass targeted at class 1 (Defective) and returns
    normalised 32x32 CAM (or None if unavailable).
    Assumes model already has activations from a forward() call.
    """
    grad = np.array([[0.0, 1.0]])
    grad_conv2 = None
    for i in range(len(model.layers) - 1, -1, -1):
        grad = model.layers[i].backward(grad)
        if i == 5:
            grad_conv2 = grad
            break

    if grad_conv2 is None:
        return None

    # Extract Conv2 feature activations after ReLU (Layer 4)
    act_conv2 = np.maximum(0, model.layers[4].last_x)   # (1, 16, 32, 32)
    weights = np.mean(grad_conv2, axis=(2, 3), keepdims=True)  # (1, 16, 1, 1)

    cam = np.sum(weights * act_conv2, axis=1)[0]   # (32, 32)
    cam = np.maximum(0, cam)
    max_cam = cam.max()
    if max_cam <= 1e-9:
        return None
    return cam / max_cam   # [0, 1], shape (32, 32)


# ---------------------------------------------------------------------------
# Public: Defect Localization (bboxes + dots)
# ---------------------------------------------------------------------------

def locate_defects(
    model: Sequential,
    image: Image.Image,
    threshold_ratio: float = 0.58,
    min_area: int = 40,
    max_components: int = 4
) -> Dict:
    """
    Performs Grad-CAM feature localization on the last convolutional layer.
    Extracts bounding boxes and marker dots where anomalous defect patterns are concentrated.
    """
    orig_w, orig_h = image.size

    # 1. Preprocess to model input
    gray = image.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
    tensor = (np.array(gray, dtype=np.float32) / 255.0).reshape(1, 1, 64, 64)

    # 2. Forward pass to capture feature activations
    logits = model.forward(tensor, training=True)
    probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    def_prob = float(probs[0, 1])

    # If predicted OK with high confidence, no defect to localize
    if def_prob < 0.45:
        return {
            "has_defect": False,
            "defect_count": 0,
            "boxes": [],
            "points": [],
            "confidence": round(def_prob, 4)
        }

    # 3. Grad-CAM backward pass
    cam_32 = _compute_gradcam_32(model)
    if cam_32 is None:
        return {"has_defect": False, "defect_count": 0, "boxes": [], "points": []}

    # 4. Upscale CAM heatmap to original image resolution
    cam_pil = Image.fromarray((cam_32 * 255).astype(np.uint8)).resize(
        (orig_w, orig_h), Image.Resampling.BILINEAR
    )
    cam_arr = np.array(cam_pil, dtype=np.float32) / 255.0

    # 5. Threshold active defect regions
    mask = cam_arr >= threshold_ratio
    comps = get_connected_components(mask, min_area=min_area)

    # Sort components by peak intensity * area
    comps_sorted = sorted(
        comps,
        key=lambda c: max(cam_arr[p[0], p[1]] for p in c) * np.sqrt(len(c)),
        reverse=True
    )[:max_components]

    boxes = []
    points = []
    margin = 8

    for idx, comp in enumerate(comps_sorted, 1):
        ys = [p[0] for p in comp]
        xs = [p[1] for p in comp]

        x_min = max(0, min(xs) - margin)
        y_min = max(0, min(ys) - margin)
        x_max = min(orig_w, max(xs) + margin)
        y_max = min(orig_h, max(ys) + margin)

        # Peak coordinate (hottest spot of defect)
        peak_y, peak_x = max(comp, key=lambda p: cam_arr[p[0], p[1]])
        peak_intensity = float(cam_arr[peak_y, peak_x])

        box = {
            "id": idx,
            "x": int(x_min),
            "y": int(y_min),
            "width": int(x_max - x_min),
            "height": int(y_max - y_min),
            "intensity": round(peak_intensity, 3),
            "confidence": round(def_prob, 4),
            "label": f"Defect #{idx}"
        }
        boxes.append(box)

        # Hotspot center point
        points.append({
            "box_id": idx,
            "x": int(peak_x),
            "y": int(peak_y),
            "type": "peak",
            "label": f"Pusat Defect #{idx}"
        })

        # Sample perimeter dots along the edge of the defect area
        step = max(1, len(comp) // 6)
        for k in range(0, len(comp), step):
            points.append({
                "box_id": idx,
                "x": int(comp[k][1]),
                "y": int(comp[k][0]),
                "type": "perimeter"
            })

    return {
        "has_defect": len(boxes) > 0,
        "defect_count": len(boxes),
        "boxes": boxes,
        "points": points,
        "confidence": round(def_prob, 4)
    }


# ---------------------------------------------------------------------------
# Public: Annotated Image (Bounding Boxes + Dots)
# ---------------------------------------------------------------------------

def generate_annotated_image_base64(image: Image.Image, localization: Dict) -> Optional[str]:
    """
    Renders visual bounding boxes, center marker dots, and defect tags
    directly onto the image, returning a Base64-encoded JPEG data URL.
    """
    if not localization.get("has_defect"):
        return None

    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    boxes = localization.get("boxes", [])
    points = localization.get("points", [])

    # 1. Draw Bounding Boxes
    for box in boxes:
        x0, y0 = box["x"], box["y"]
        x1, y1 = x0 + box["width"], y0 + box["height"]

        # Outer bold red bounding box
        draw.rectangle([x0, y0, x1, y1], outline=(239, 68, 68), width=3)

        # Label banner
        label_text = f"{box['label']} ({box['confidence']*100:.1f}%)"
        tag_w = len(label_text) * 7 + 8
        tag_h = 18
        tag_y0 = max(0, y0 - tag_h - 2)
        draw.rectangle([x0, tag_y0, x0 + tag_w, tag_y0 + tag_h], fill=(239, 68, 68))
        draw.text((x0 + 4, tag_y0 + 2), label_text, fill=(255, 255, 255))

    # 2. Draw Marker Dots
    for pt in points:
        px, py = pt["x"], pt["y"]
        if pt.get("type") == "peak":
            # Primary peak dot: yellow center with red rim
            r = 5
            draw.ellipse([px - r, py - r, px + r, py + r], fill=(245, 158, 11), outline=(220, 38, 38), width=2)
        else:
            # Perimeter dots: small red markers
            r = 2
            draw.ellipse([px - r, py - r, px + r, py + r], fill=(239, 68, 68))

    # Export to base64 JPEG
    buffer = io.BytesIO()
    annotated.save(buffer, format="JPEG", quality=90)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


# ---------------------------------------------------------------------------
# Public: 128x128 Surface Anomaly Matrix (Jet Heatmap)
# ---------------------------------------------------------------------------

def compute_anomaly_matrix(
    model: Sequential,
    image: Image.Image
) -> Dict:
    """
    Computes a 128x128 Surface Anomaly Matrix by fusing Grad-CAM activation
    with gradient disparity (edge magnitude map) of the input image.

    Returns dict with:
        matrix       : np.ndarray (128, 128) float32 in [0, 1]
        peak_spot    : (row, col) of the highest anomaly value
        def_prob     : float defect probability from the model
    """
    # 1. Build 128x128 grayscale array of image
    gray_128 = image.convert("L").resize((128, 128), Image.Resampling.BILINEAR)
    arr_128 = np.array(gray_128, dtype=np.float32) / 255.0  # (128, 128)

    # 2. Gradient disparity map (pure NumPy, no OpenCV)
    gx = np.zeros_like(arr_128)
    gy = np.zeros_like(arr_128)
    gx[:, 1:-1] = np.abs(arr_128[:, 2:] - arr_128[:, :-2])
    gy[1:-1, :] = np.abs(arr_128[2:, :] - arr_128[:-2, :])
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    gm_max = grad_mag.max()
    if gm_max > 1e-9:
        grad_mag = grad_mag / gm_max

    # 3. Forward pass on 64x64 input for Grad-CAM
    gray_64 = image.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
    tensor = (np.array(gray_64, dtype=np.float32) / 255.0).reshape(1, 1, 64, 64)
    logits = model.forward(tensor, training=True)
    probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    def_prob = float(probs[0, 1])

    # 4. Grad-CAM backward
    cam_32 = _compute_gradcam_32(model)

    if cam_32 is None:
        # Fallback: gradient disparity only
        matrix = grad_mag
    else:
        # 5. Upscale CAM 32->128
        cam_pil = Image.fromarray((cam_32 * 255).astype(np.uint8)).resize(
            (128, 128), Image.Resampling.BILINEAR
        )
        cam_128 = np.array(cam_pil, dtype=np.float32) / 255.0

        # 6. Fuse: 75% CAM + 25% (CAM × grad disparity)
        fused = cam_128 * 0.75 + (cam_128 * grad_mag) * 0.25
        fused_max = fused.max()
        if fused_max > 1e-9:
            matrix = fused / fused_max
        else:
            matrix = fused

    # 7. Find peak location
    peak_flat = int(np.argmax(matrix))
    peak_spot = (peak_flat // 128, peak_flat % 128)

    return {
        "matrix": matrix.astype(np.float32),
        "peak_spot": peak_spot,
        "def_prob": def_prob
    }


def render_surface_anomaly_matrix(
    matrix_128: np.ndarray,
    peak_spot: Optional[Tuple[int, int]] = None,
    def_prob: float = 0.0
) -> Image.Image:
    """
    Renders a 128x128 anomaly matrix as a jet-colormap heatmap with
    axis ticks, colorbar, title, and optional peak marker dot.

    Returns a PIL Image (340 x 310 px white canvas).
    """
    CANVAS_W, CANVAS_H = 340, 310
    HMAP_X, HMAP_Y = 45, 38       # top-left of heatmap on canvas
    HMAP_SIZE = 220                # heatmap display size (square)
    CBAR_X = 280                   # colorbar left-x
    CBAR_W = 14
    CBAR_H = HMAP_SIZE

    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # --- Title ---
    draw.text((CANVAS_W // 2 - 90, 6), "128x128 Surface Anomaly Matrix", fill=(30, 30, 30))
    draw.text((CANVAS_W // 2 - 100, 19), "[Hotspots = High Anomaly / Surface Defect]", fill=(180, 20, 20))

    # --- Apply jet colormap → 220x220 heatmap ---
    # Downsample/upscale matrix to HMAP_SIZE x HMAP_SIZE via PIL
    mat_uint8 = (np.clip(matrix_128, 0, 1) * 255).astype(np.uint8)
    mat_pil_gray = Image.fromarray(mat_uint8, mode="L").resize(
        (HMAP_SIZE, HMAP_SIZE), Image.Resampling.BILINEAR
    )
    mat_arr_scaled = np.array(mat_pil_gray, dtype=np.float32) / 255.0

    # Build jet RGB array
    jet_rgb = _apply_jet_to_array(mat_arr_scaled)   # (HMAP_SIZE, HMAP_SIZE, 3)
    heatmap_img = Image.fromarray(jet_rgb, mode="RGB")
    canvas.paste(heatmap_img, (HMAP_X, HMAP_Y))

    # --- Axis ticks ---
    ticks = [0, 32, 64, 96, 128]
    tick_color = (80, 80, 80)
    for t in ticks:
        px = HMAP_X + int(t * HMAP_SIZE / 128)
        # X axis (bottom)
        draw.line([(px, HMAP_Y + HMAP_SIZE), (px, HMAP_Y + HMAP_SIZE + 4)], fill=tick_color)
        draw.text((px - 6, HMAP_Y + HMAP_SIZE + 6), str(t), fill=tick_color)
        # Y axis (left)
        py = HMAP_Y + int(t * HMAP_SIZE / 128)
        draw.line([(HMAP_X - 4, py), (HMAP_X, py)], fill=tick_color)
        draw.text((2, py - 5), str(t), fill=tick_color)

    # --- Colorbar (vertical gradient, jet) ---
    for cbar_row in range(CBAR_H):
        t = 1.0 - (cbar_row / (CBAR_H - 1))
        r, g, b = _jet_colormap(t)
        for cx in range(CBAR_W):
            canvas.putpixel((CBAR_X + cx, HMAP_Y + cbar_row), (r, g, b))

    # Colorbar border
    draw.rectangle([CBAR_X, HMAP_Y, CBAR_X + CBAR_W, HMAP_Y + CBAR_H], outline=(150, 150, 150))

    # Colorbar scale labels
    for label_pct in [0.0, 0.25, 0.5, 0.75, 1.0]:
        label_y = HMAP_Y + int((1.0 - label_pct) * CBAR_H)
        draw.text((CBAR_X + CBAR_W + 2, label_y - 5), f"{label_pct:.1f}", fill=(60, 60, 60))

    # --- Peak marker dot (yellow) ---
    if peak_spot is not None:
        pr, pc = peak_spot
        dot_x = HMAP_X + int(pc * HMAP_SIZE / 128)
        dot_y = HMAP_Y + int(pr * HMAP_SIZE / 128)
        rr = 5
        draw.ellipse([dot_x - rr, dot_y - rr, dot_x + rr, dot_y + rr],
                     fill=(255, 230, 0), outline=(220, 50, 50), width=2)

    return canvas


def compute_anomaly_matrix_base64(
    model: Sequential,
    image: Image.Image
) -> str:
    """
    Full pipeline: compute anomaly matrix → render jet heatmap → encode base64 JPEG.
    Returns data URL string suitable for <img src="...">.
    Always returns a valid image even for OK parts (low activation = blue matrix).
    """
    result = compute_anomaly_matrix(model, image)
    rendered = render_surface_anomaly_matrix(
        result["matrix"],
        peak_spot=result["peak_spot"],
        def_prob=result["def_prob"]
    )
    buffer = io.BytesIO()
    rendered.save(buffer, format="JPEG", quality=92)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"
