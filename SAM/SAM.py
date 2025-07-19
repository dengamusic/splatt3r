from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import os
import cv2
import numpy as np
import torch
import argparse
import clip
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Segment images using SAM')
    parser.add_argument('--input_dir', required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', required=True, help='Root directory for output files')
    parser.add_argument('--checkpoint', required=True, help='Path to SAM checkpoint file')
    parser.add_argument('--model_type', default='vit_h', help='SAM model type (default: vit_h)')
    parser.add_argument('--clip_model_type', default="ViT-B/32", help='CLIP model type (default: ViT-B/32)')
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
    clip_model, clip_preprocess = clip.load(args.clip_model_type, device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    print(f"device: {device}")

    for fname in os.listdir(args.input_dir):
        img_path = os.path.join(args.input_dir, fname)
        img_bgr = cv2.imread(img_path)

        #img_bgr = cv2.resize(img_bgr, (512, 512))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(img_rgb)

        vis = img_bgr.copy()
        for i, m in enumerate(masks):
            mask_bin = (m["segmentation"]).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

            mask_filename = os.path.splitext(fname)[0] + f"_mask_{i:03d}.png"
            cv2.imwrite(os.path.join(masks_dir, mask_filename), mask_bin)

            
            #CLIP embedding
            mask = m["segmentation"]
            ys, xs = np.where(mask)
            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()

            cropped_img = img_rgb[ymin:ymax+1, xmin:xmax+1, :]
            cropped_pil = Image.fromarray(cropped_img)

            input_clip = clip_preprocess(cropped_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(input_clip)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # normalize

            npz_filename = os.path.splitext(fname)[0] + f"_mask_{i:03d}.npz"
            np.savez_compressed(
                os.path.join(masks_dir, npz_filename),
                mask=mask_bin,
                embedding=image_features.cpu().numpy()
            )
            
        vis_filename = fname
        cv2.imwrite(os.path.join(vis_dir, vis_filename), vis)

if __name__ == "__main__":
    main()