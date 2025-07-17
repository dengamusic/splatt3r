from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import os
import cv2
import numpy as np
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Segment images using SAM')
    parser.add_argument('--input_dir', required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', required=True, help='Root directory for output files')
    parser.add_argument('--checkpoint', required=True, help='Path to SAM checkpoint file')
    parser.add_argument('--model_type', default='vit_h', help='SAM model type (default: vit_h)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directories
    masks_dir = os.path.join(args.output_dir, "masks")
    vis_dir = os.path.join(args.output_dir, "vis")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    print(f"device: {device}")

    for fname in os.listdir(args.input_dir):
        img_path = os.path.join(args.input_dir, fname)
        img_bgr = cv2.imread(img_path)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(img_rgb)

        vis = img_bgr.copy()
        for i, m in enumerate(masks):
            mask_bin = (m["segmentation"]).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

            mask_filename = os.path.splitext(fname)[0] + f"_mask_{i:03d}.png"
            cv2.imwrite(os.path.join(masks_dir, mask_filename), mask_bin)

            vis_filename = fname
            cv2.imwrite(os.path.join(vis_dir, vis_filename), vis)


if __name__ == "__main__":
    main()