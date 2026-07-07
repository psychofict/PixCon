"""PixCon interactive demo (Hugging Face Space) — Pascal VOC / Cityscapes / ADE20K.

Layout expected in the Space repo:
    app.py, labels.py, requirements.txt
    model/   (segmentor.py, dinov2.py, __init__.py)
    core/    (inference.py, __init__.py)
    examples/ (a few demo images, prefixed pascal_/cityscapes_/ade20k_)

Weights are pulled per dataset from the matching HF model repo (see DATASETS). Runs on CPU
(free tier); uses CUDA if present. Set PIXCON_LOCAL_DIR to load weights from a local folder
(filenames must match DATASETS[*]['file']) and PIXCON_NO_WARM=1 to skip background warming — both
used for offline testing.
"""

import os
import threading
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
import gradio as gr

from model.segmentor import PixConSegmentor
from core.inference import whole_inference
from labels import (VOC_CLASSES, VOC_PALETTE, CS_CLASSES, CS_PALETTE,
                    ADE_CLASSES, ADE_PALETTE)

MAX_SIDE = 1024                       # cap long side for inference; overlay is drawn at input res
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = {
    "Pascal VOC (21 classes)": dict(
        repo="psychofict/pixcon-pascal", file="pixcon_pascal_1_8.pth",
        nclass=21, classes=VOC_CLASSES, palette=VOC_PALETTE, bg=0),
    "Cityscapes (19 classes)": dict(
        repo="psychofict/pixcon-cityscapes", file="pixcon_cityscapes_1_8.pth",
        nclass=19, classes=CS_CLASSES, palette=CS_PALETTE, bg=None),
    "ADE20K (150 classes)": dict(
        repo="psychofict/pixcon-ade20k", file="pixcon_ade20k_1_8.pth",
        nclass=150, classes=ADE_CLASSES, palette=ADE_PALETTE, bg=None),
}
DEFAULT_DATASET = "Pascal VOC (21 classes)"

_normalize = T.Compose([
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
_MODELS = {}
_LOCK = threading.Lock()


def _load_state_dict(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "teacher" in ckpt and isinstance(ckpt["teacher"], dict) \
                and "model_state_dict" in ckpt["teacher"]:
            return ckpt["teacher"]["model_state_dict"]
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "student" in ckpt:
            return ckpt["student"]
    return ckpt


def _weights_path(cfg):
    local_dir = os.environ.get("PIXCON_LOCAL_DIR")
    if local_dir:
        p = os.path.join(local_dir, cfg["file"])
        if os.path.exists(p):
            return p
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=cfg["repo"], filename=cfg["file"])


def get_model(dataset):
    """Lazy, thread-safe per-dataset singleton. Keeps import fast (Space binds its port before
    any DINOv2/weights download) and shares one load between the warmer and requests."""
    with _LOCK:
        if dataset not in _MODELS:
            cfg = DATASETS[dataset]
            model = PixConSegmentor(backbone="dinov2_vitb14", nclass=cfg["nclass"],
                                    pretrained=False).eval()
            model.load_state_dict(_load_state_dict(_weights_path(cfg)), strict=False)
            _MODELS[dataset] = model.to(DEVICE)
        return _MODELS[dataset]


def _resize_long_side(img, max_side):
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    return img


def _legend_html(pairs, cfg):
    """pairs: list of (class_idx, pixel_count), sorted by count desc."""
    total = sum(c for _, c in pairs) or 1
    chips = []
    for idx, cnt in pairs:
        if idx == cfg["bg"] or idx >= len(cfg["classes"]):
            continue
        r, g, b = cfg["palette"][idx]
        pct = 100.0 * cnt / total
        chips.append(
            f"<span class='chip'><span class='sw' style='background:rgb({r},{g},{b})'></span>"
            f"{cfg['classes'][idx]}<span class='count'>{pct:.0f}%</span></span>")
    if not chips:
        chips = ["<span class='muted'>Background only.</span>"]
    return f"<div class='legend'>{''.join(chips)}</div>"


_EMPTY_LEGEND = "<div class='legend'><span class='muted'>Run a segmentation to see detected classes.</span></div>"


@torch.no_grad()
def segment(image, dataset=DEFAULT_DATASET, alpha=0.55):
    if image is None:
        return None, _EMPTY_LEGEND
    cfg = DATASETS[dataset]
    palette = np.asarray(cfg["palette"], dtype=np.uint8)

    orig = image.convert("RGB")
    small = _resize_long_side(orig, MAX_SIDE)
    x = _normalize(small).unsqueeze(0).to(DEVICE)
    logits = whole_inference(get_model(dataset), x)          # [1, nclass, h, w]
    pred_small = logits.argmax(1)[0].to(torch.uint8).cpu().numpy()

    # upsample the label map to the ORIGINAL input resolution so the overlay matches the input
    pred = np.array(Image.fromarray(pred_small).resize(orig.size, Image.NEAREST))
    seg_rgb = palette[pred]
    base = np.asarray(orig, dtype=np.float32)
    overlay = (alpha * seg_rgb + (1 - alpha) * base).clip(0, 255).astype(np.uint8)

    vals, cnts = np.unique(pred, return_counts=True)
    pairs = sorted(zip(vals.tolist(), cnts.tolist()), key=lambda t: -t[1])
    return Image.fromarray(overlay), _legend_html(pairs, cfg)


def _build_examples():
    ex_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
    prefix = {"pascal": "Pascal VOC (21 classes)", "cityscapes": "Cityscapes (19 classes)",
              "ade20k": "ADE20K (150 classes)"}
    rows = []
    if os.path.isdir(ex_dir):
        for fn in sorted(os.listdir(ex_dir)):
            ds = prefix.get(fn.split("_")[0])
            if ds:
                rows.append([os.path.join(ex_dir, fn), ds, 0.55])
    return rows


def _warm_all():
    for ds in DATASETS:
        try:
            get_model(ds)
        except Exception:
            pass


CSS = """
.gradio-container {max-width: 1120px !important; margin: 0 auto !important;}
#hero {text-align:center; padding: 10px 0 2px;}
#hero h1 {font-size: 2rem; font-weight: 800; margin: 0 0 2px;
  background: linear-gradient(90deg,#6366f1,#a855f7); -webkit-background-clip:text;
  background-clip:text; color:transparent;}
#hero .sub {opacity:.85; margin: 2px 0 8px; font-size:1rem;}
#hero .links a {margin:0 4px; padding:3px 12px; border-radius:999px; text-decoration:none;
  border:1px solid rgba(128,128,128,.35); font-size:.85rem; white-space:nowrap;}
#hero .hint {font-size:.82rem; opacity:.65; margin-top:8px;}
#classhead {margin:6px 0 0; font-weight:600;}
.legend {display:flex; flex-wrap:wrap; gap:7px; max-height:170px; overflow-y:auto;
  padding:10px; border:1px solid rgba(128,128,128,.25); border-radius:12px;}
.chip {display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border-radius:999px;
  border:1px solid rgba(128,128,128,.3); font-size:.86rem; line-height:1.4;}
.chip .sw {width:13px; height:13px; border-radius:3px; display:inline-block;
  border:1px solid rgba(0,0,0,.18);}
.chip .count {opacity:.55; font-size:.78rem;}
.legend .muted {opacity:.6;}
#foot {text-align:center; opacity:.6; font-size:.82rem; padding:10px 0 4px;}
"""

HERO = """
<div id="hero">
  <h1>PixCon</h1>
  <div class="sub">Clean-Positive Contrastive Learning for Foundation-Model Semi-Supervised Segmentation</div>
  <div class="links">
    <a href="https://arxiv.org/abs/2607.03068" target="_blank">📄 arXiv:2607.03068</a>
    <a href="https://huggingface.co/psychofict/pixcon-pascal" target="_blank">Pascal VOC</a>
    <a href="https://huggingface.co/psychofict/pixcon-cityscapes" target="_blank">Cityscapes</a>
    <a href="https://huggingface.co/psychofict/pixcon-ade20k" target="_blank">ADE20K</a>
  </div>
  <div class="hint">DINOv2-Base encoder. Upload an image and pick a label set, or click an example below.
  The first run per model warms up (~1 min on CPU); after that it is fast.</div>
</div>
"""

FOOT = "<div id='foot'>PixCon · single DINOv2-Base backbone · no inference-time parameters · arXiv:2607.03068</div>"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple"),
               css=CSS, title="PixCon — Semi-Supervised Segmentation Demo") as demo:
    gr.HTML(HERO)
    with gr.Row():
        dataset = gr.Dropdown(choices=list(DATASETS.keys()), value=DEFAULT_DATASET,
                              label="Label set / model", scale=3)
        alpha = gr.Slider(0.0, 1.0, value=0.55, step=0.05, label="Overlay opacity", scale=2)
    with gr.Row(equal_height=True):
        inp = gr.Image(type="pil", label="Input", height=430)
        out = gr.Image(type="pil", label="Segmentation overlay", height=430)
    with gr.Row():
        run = gr.Button("Segment", variant="primary", scale=3)
        clear = gr.Button("Clear", scale=1)
    gr.HTML("<div id='classhead'>Detected classes</div>")
    legend = gr.HTML(_EMPTY_LEGEND)

    gr.Examples(
        examples=_build_examples(),
        inputs=[inp, dataset, alpha], outputs=[out, legend], fn=segment,
        cache_examples=False, examples_per_page=6, label="Examples (click to run)",
    )
    gr.HTML(FOOT)

    run.click(segment, [inp, dataset, alpha], [out, legend])
    dataset.change(segment, [inp, dataset, alpha], [out, legend])
    clear.click(lambda: (None, None, _EMPTY_LEGEND), None, [inp, out, legend])

if os.environ.get("PIXCON_NO_WARM") != "1":
    threading.Thread(target=_warm_all, daemon=True).start()

if __name__ == "__main__":
    demo.launch()
