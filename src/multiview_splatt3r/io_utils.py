
import os
import json

def ensure_outdirs(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    imgs_outdir = os.path.join(outdir, "imgs")
    os.makedirs(imgs_outdir, exist_ok=True)
    return imgs_outdir

def copy_images_and_write_list(images, imgs_outdir: str, list_path: str) -> None:
    import shutil
    with open(list_path, "w") as f:
        for i, src in enumerate(images):
            dst = os.path.join(imgs_outdir, os.path.basename(src))
            shutil.copy2(src, dst)
            f.write(f"{i} {os.path.basename(src)}
")
    print(f"✓ copied {len(images)} images to {imgs_outdir}")

def write_poses_txt(path: str, poses, im_order, image_paths) -> None:
    import numpy as np
    with open(path, "w") as f:
        for row, img_idx in enumerate(im_order):
            M = poses[img_idx]
            fname = os.path.basename(image_paths[img_idx])
            f.write(f"# {fname}
")
            np.savetxt(f, M.detach().cpu().numpy(), fmt="%.8f")
            f.write("
")
    print("✓ wrote", path)

def write_metrics_json(path: str, metrics) -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("✓ wrote", path)

def write_metrics_csv(path: str, per_view_rows) -> None:
    import csv
    with open(path, "w", newline="") as f:
        if per_view_rows:
            fields = sorted({k for r in per_view_rows for k in r.keys()})
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(per_view_rows)
    print("✓ wrote", path)
