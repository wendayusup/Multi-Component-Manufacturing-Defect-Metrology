
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import torch
from torch import nn
from torchvision.ops import nms

STRIDE = 4

def block(a, b, stride=1):
    return nn.Sequential(nn.Conv2d(a, b, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(b), nn.ReLU(inplace=True))

class PlasticDetector(nn.Module):
    """CNN satu kelas. Setiap titik memprediksi pusat, offset pusat dan ukuran kotak."""
    def __init__(self):
        super().__init__()
        self.fine = nn.Sequential(block(3, 16, 2), block(16, 32, 2), block(32, 32))
        self.middle = nn.Sequential(block(32, 48, 2), block(48, 48))
        self.deep = nn.Sequential(block(48, 64, 2), block(64, 64), block(64, 64))
        self.fuse = block(32 + 48 + 64, 48)
        self.center = nn.Conv2d(48, 1, 1)
        self.geometry = nn.Conv2d(48, 4, 1)
        nn.init.constant_(self.center.bias, -4.6)

    def forward(self, x):
        a = self.fine(x)
        b = self.middle(a)
        c = self.deep(b)
        features = torch.cat([a, nn.functional.interpolate(b, size=a.shape[-2:]),
                              nn.functional.interpolate(c, size=a.shape[-2:])], dim=1)
        h = self.fuse(features)
        return self.center(h), self.geometry(h)

def prepare(image, side):
    w, h = image.size
    scale = side / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    left, top = (side - nw) // 2, (side - nh) // 2
    canvas = Image.new('RGB', (side, side), (114, 114, 114))
    canvas.paste(image.convert('RGB').resize((nw, nh), Image.Resampling.BILINEAR), (left, top))
    tensor = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float().div_(255)
    return tensor, (nw / w, nh / h, left, top)

def decode(center, geometry, mapping, size, cutoff=0.01):
    scores = center.sigmoid().flatten()
    scores, ids = scores.topk(min(1000, scores.numel()))
    keep = scores >= cutoff
    scores, ids = scores[keep], ids[keep]
    grid_w = geometry.shape[-1]
    y, x = ids // grid_w, ids % grid_w
    values = geometry[:, y, x].T
    xy = (torch.stack((x, y), 1) + values[:, :2].sigmoid()) * STRIDE
    wh = values[:, 2:].clamp(-4, 8).exp() * STRIDE
    boxes = torch.cat([xy - wh / 2, xy + wh / 2], dim=1)
    sx, sy, left, top = mapping
    boxes[:, 0::2] = ((boxes[:, 0::2] - left) / sx).clamp(0, size[0])
    boxes[:, 1::2] = ((boxes[:, 1::2] - top) / sy).clamp(0, size[1])
    valid = (boxes[:, 2:] > boxes[:, :2]).all(dim=1)
    boxes, scores = boxes[valid], scores[valid]
    indices = nms(boxes, scores, 0.45)[:100]
    return boxes[indices].cpu(), scores[indices].cpu()

def load_detector(path, device='cpu'):
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    model = PlasticDetector()
    model.load_state_dict(checkpoint['weights'])
    return model.to(device).eval(), checkpoint['side'], checkpoint['threshold']

@torch.inference_mode()
def inspect_photo(photo, bundle):
    model, side, threshold = bundle
    image = ImageOps.exif_transpose(photo).convert('RGB')
    tensor, mapping = prepare(image, side)
    # Supply reflected context outside the input, then restore the original grid.
    # A multiple of 16 keeps all three feature levels aligned.
    margin = 64
    batch = tensor[None].to(next(model.parameters()).device)
    expanded = nn.functional.pad(batch, (margin,) * 4, mode='reflect')
    heat, geometry = model(expanded)
    trim = margin // STRIDE
    heat = heat[:, :, trim:-trim, trim:-trim]
    geometry = geometry[:, :, trim:-trim, trim:-trim]
    boxes, scores = decode(heat[0], geometry[0], mapping, image.size, threshold)
    draw = ImageDraw.Draw(image)
    detections = []
    for box, score in zip(boxes.tolist(), scores.tolist()):
        draw.rectangle(box, outline='red', width=max(2, image.width // 500))
        draw.text((box[0], max(0, box[1]-12)), f'defect {score:.2f}', fill='red')
        detections.append(dict(label='defect', xyxy=box, confidence=score))
    return image, detections



