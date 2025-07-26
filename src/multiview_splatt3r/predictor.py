
import torch

@torch.no_grad()
def run_pair(model, imgs, i: int, j: int):
    v_i, v_j = imgs[i], imgs[j]
    p_i, p_j = model(v_i, v_j)

    for p in (p_i, p_j):
        if "log_scales" not in p and "scales" in p:
            p["log_scales"] = p["scales"].log()
        if "logit_opacities" not in p and "opacities" in p:
            a = p["opacities"].clamp(1e-4, 1 - 1e-4)
            p["logit_opacities"] = (a / (1 - a)).log()

    d_i = {"pts3d": p_i["pts3d"], "conf": p_i["conf"]}
    d_j = {
        "pts3d_in_other_view": p_j["pts3d_in_other_view"],
        "conf": p_j["conf"],
        "pts3d": p_j["pts3d_in_other_view"],
    }
    for k in ("means", "rotations", "sh", "log_scales", "logit_opacities"):
        if k in p_j:
            d_j["means_in_other_view" if k == "means" else k] = p_j[k]
    return v_i, v_j, d_i, d_j, p_i, p_j
