import os
import cv2
import numpy as np
import torch
import argparse
from PIL import Image
import clip
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def parse_args():
    parser = argparse.ArgumentParser(description='SAM + CLIP Semantic Segmentation')
    parser.add_argument('--input_dir', required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', required=True, help='Directory to save outputs')
    parser.add_argument('--checkpoint', required=True, help='Path to SAM checkpoint file')
    parser.add_argument('--model_type', default='vit_h', help='SAM model type (default: vit_h)')
    parser.add_argument('--clip_model_type', default='ViT-B/32', help='CLIP model type')
    #parser.add_argument('--labels', required=True, help='Path to text file with candidate labels (one per line)')
    return parser.parse_args()

def load_clip_labels(label_path, device, clip_model):
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    text_tokens = clip.tokenize(labels).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return labels,  text_features

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    masks_dir = os.path.join(args.output_dir, "masks")
    vis_dir = os.path.join(args.output_dir, "vis")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load SAM
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Load CLIP
    clip_model, clip_preprocess = clip.load(args.clip_model_type, device=device)
    
    # label_list, text_features = load_clip_labels(args.labels, device, clip_model)

    # Process each image
    for fname in os.listdir(args.input_dir):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(args.input_dir, fname)
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(img_rgb)
        masks = mask_generator.generate(img_rgb)
        vis = img_bgr.copy()
        base_name = os.path.splitext(fname)[0]
        results = []
        
        image_area = img_rgb.shape[0] * img_rgb.shape[1]
        min_mask_area = 0.01 * image_area

        for i, m in enumerate(masks):
            mask = m["segmentation"]
            '''
            if mask.sum() < min_mask_area:
                continue
            '''
            mask_bin = (mask.astype(np.uint8)) * 255

            # Apply mask to image
            
            masked_img = img_rgb.copy()
            blurred = cv2.GaussianBlur(img_rgb, (25, 25), 0)
            masked_img[~mask] = blurred[~mask]
            masked_crop = Image.fromarray(masked_img).convert("RGB")
            '''

            ys, xs = np.where(mask)
            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()
            masked_img = img_rgb[ymin:ymax+1, xmin:xmax+1, :]
            masked_crop = Image.fromarray(masked_img)
            '''
            # CLIP preprocess and inference
            img_input = clip_preprocess(masked_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                image_feat = clip_model.encode_image(img_input)
                image_feat /= image_feat.norm(dim=-1, keepdim=True)
                image_feat_np = image_feat.cpu().numpy()
                '''
                sims = (image_feat @ text_features.T).squeeze(0)
                best_idx = sims.argmax().item()
                label = label_list[best_idx]
                score = float(sims[best_idx].item())
                '''

                '''
                sims = (image_feat @ text_features.T).squeeze(0)  # shape: [num_prompts]
                sims_per_label = sims.view(len(labels), len(prompt_templates))  # reshape
                avg_sims = sims_per_label.mean(dim=1)  # shape: [num_labels]
                best_idx = avg_sims.argmax().item()
                label = labels[best_idx]
                score = float(avg_sims[best_idx].item())
                '''

            # Save mask
            npy_path = os.path.join(masks_dir, f"{base_name}_mask_{i:03d}.npy")
            png_path = os.path.join(masks_dir, f"{base_name}_mask_{i:03d}.png")
            np.save(npy_path, mask)
            cv2.imwrite(png_path, cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))
            feat_path = os.path.join(masks_dir, f"{base_name}_feat_{i:03d}.npy")
            np.save(feat_path, image_feat_np)
            
            results.append({
                "image": fname,
                "mask": os.path.basename(npy_path),
                "feature": os.path.basename(feat_path)
            })
            '''
            # Visualization overlay
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
            ys, xs = np.where(mask)
            if len(xs) > 0 and len(ys) > 0:
                
                cv2.putText(vis, label, (int(xs.min()), max(0, int(ys.min()) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
            
            '''
        # Save visualization
        # cv2.imwrite(os.path.join(vis_dir, fname), vis)
        
        # Save results
        
        result_txt = os.path.join(masks_dir, f"{base_name}_results.txt")
        with open(result_txt, "w", encoding="utf-8") as f:
            for r in results:
                f.write(f"Image: {r['image']}, Mask: {r['mask']}, Feature: {r['feature']}\n")
        #print(f"Processed {fname}: {len(results)} masks with semantic labels.")
        
        # Save image features
        feat_path = os.path.join(masks_dir, f"{base_name}_feat_{i:03d}.npy")
        np.save(feat_path, image_feat_np)

if __name__ == "__main__":
    main()
