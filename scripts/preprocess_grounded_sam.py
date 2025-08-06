#!/usr/bin/env python3
"""
Grounded‑SAM batch processor (single GPU)

usage:
    python grounded_sam_dir.py  /path/to/images  /path/to/out  /path/to/Grounded-Segment-Anything
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as TS
from PIL import Image
from tqdm import tqdm
import clip
import matplotlib.pyplot as plt
import os
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BASE_DIR = f"{_BASE_DIR}/Grounded-Segment-Anything"
sys.path.insert(0, _BASE_DIR)

from automatic_label_ram_demo import (
    load_model, get_grounding_output, load_image, show_mask, show_box
)
from ram.models import ram
from ram import inference_ram
from segment_anything import build_sam, SamPredictor
# ─────────────────────────────────────────────────────────────────


# ──────────────────────────── helpers ────────────────────────────
def resize_bboxes(boxes, orig_sz, tgt_sz):
    """Scale (x1,y1,x2,y2) coords from orig_sz (W,H) → tgt_sz (W,H)."""
    ow, oh = orig_sz
    tw, th = tgt_sz
    scale_x, scale_y = tw / ow, th / oh
    out = boxes.clone()
    out[:, [0, 2]] *= scale_x
    out[:, [1, 3]] *= scale_y
    return out


def resize_sam_masks(masks, h, w):
    return (
        F.interpolate(masks.float(), size=(h, w), mode="nearest")
        .squeeze(1)
        .bool()
    )


def labels_to_clip(labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    with torch.no_grad():
        feats = model.encode_text(clip.tokenize(labels).to(device))
    return feats.cpu().numpy()
# ─────────────────────────────────────────────────────────────────


def process_dir(
    in_dir: Path,
    out_dir: Path,
    ckpt_dir: Path,
    device: torch.device,
    box_thr=0.25,
    text_thr=0.2,
    iou_thr=0.5,
):
    # ───── checkpoint paths ─────
    gdino_cfg  = ckpt_dir / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    gdino_ckpt = ckpt_dir / "groundingdino_swint_ogc.pth"
    ram_ckpt   = ckpt_dir / "ram_swin_large_14m.pth"
    sam_ckpt   = ckpt_dir / "sam_vit_h_4b8939.pth"

    # ───── heavy models (loaded ONCE) ─────
    print("[INFO] Loading models …")
    ram_model   = ram(pretrained=str(ram_ckpt), image_size=384, vit="swin_l").to(device).eval()
    gdino_model = load_model(str(gdino_cfg), str(gdino_ckpt), device=device)
    sam_predict = SamPredictor(build_sam(checkpoint=str(sam_ckpt)).to(device))

    ram_norm = TS.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ram_pre  = TS.Compose([TS.Resize((384, 384)), TS.ToTensor(), ram_norm])

    # ───── image list ─────
    imgs = sorted(p for p in in_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not imgs:
        print("[WARN] No images found – nothing to do.")
        return
    print(f"[INFO] Found {len(imgs)} images")

    for img_path in tqdm(imgs, desc="Images"):
        stub     = img_path.stem
        out_file = out_dir / f"{stub}_groundedsam.npz"
        if out_file.exists():
            continue  # skip already processed

        pil_img, torch_img = load_image(str(img_path))

        # ───── RAM tags ─────
        ram_in = ram_pre(pil_img).unsqueeze(0).to(device)
        tags, _ = inference_ram(ram_in, ram_model)

        # ───── GroundingDINO ─────
        boxes, scores, phrases = get_grounding_output(
            gdino_model,
            torch_img.to(device),
            tags,
            box_thr,
            text_thr,
            device=device,
        )
        boxes = boxes.cpu()
        W, H = pil_img.size
        boxes[:, [0, 2]] *= W
        boxes[:, [1, 3]] *= H
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

        if boxes.size(0):
            keep = torchvision.ops.nms(boxes, scores, iou_thr)
            boxes = boxes[keep]
            phrases = [phrases[i] for i in keep.tolist()]

        # ───── SAM ─────
        img_np = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        if boxes.size(0):
            sam_predict.set_image(img_np)
            t_boxes = sam_predict.transform.apply_boxes_torch(boxes, img_np.shape[:2]).to(device)
            masks, m_scores, _ = sam_predict.predict_torch(
                point_coords=None, point_labels=None,
                boxes=t_boxes, multimask_output=False
            )
        else:
            masks = torch.zeros((0, img_np.shape[0], img_np.shape[1]), dtype=torch.bool)
            m_scores = torch.empty(0)

        # ───── write results ─────
        np.savez_compressed(
            out_file,
            masks=resize_sam_masks(masks, H, W).cpu().numpy(),
            mask_scores=m_scores.cpu().numpy(),
            boxes=boxes.cpu().numpy(),
            labels=np.array(phrases),
            tags=tags,
            clip=labels_to_clip(phrases),
        )

        # quick viz every 100 images
        if len(imgs) > 1 and imgs.index(img_path) % 1 == 0:
            plt.figure(figsize=(10, 10))
            plt.imshow(img_np)
            for m in masks:
                show_mask(m.cpu().numpy(), plt.gca(), random_color=True)
            for b, l in zip(boxes, phrases):
                show_box(b.numpy(), plt.gca(), l)
            plt.axis("off")
            (out_dir / "plots").mkdir(exist_ok=True)
            plt.savefig(out_dir / "plots" / f"{stub}.jpg",
                        dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close()

    print("[✓] Done.")


def main():
    parser = argparse.ArgumentParser(description="Batch Grounded‑SAM on a directory")
    parser.add_argument("in_dir",  help="Directory with input images")
    parser.add_argument("out_dir", help="Where to store .npz outputs")
    parser.add_argument("ckpt_dir", help="Directory that contains Grounded‑SAM checkpoints")
    args = parser.parse_args()

    in_dir  = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()

    if not in_dir.is_dir():
        sys.exit(f"[ERR] in_dir '{in_dir}' is not a directory")
    if not ckpt_dir.is_dir():
        sys.exit(f"[ERR] ckpt_dir '{ckpt_dir}' is not a directory")

    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[WARN] CUDA not found – running on CPU will be very slow.")

    process_dir(in_dir, out_dir, ckpt_dir, device)


if __name__ == "__main__":
    main()
