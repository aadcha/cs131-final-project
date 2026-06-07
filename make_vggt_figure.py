import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/aadichauhan/Developer/CS131Final")
GT_DIR = ROOT / "data" / "gt_poses"
OUT = Path("/Users/aadichauhan/Developer/cs131-final-report/figures/figure2_vggt_runs.png")

def w2c_to_c2w(extr3x4):
    R = extr3x4[:3, :3]
    t = extr3x4[:3, 3]
    c2w = np.eye(4); c2w[:3,:3] = R.T; c2w[:3,3] = -R.T @ t
    return c2w

def read_cam0(path):
    poses = {}
    with open(path) as f:
        for line in f:
            v = list(map(float, line.split()))
            if len(v) < 17: continue
            poses[int(v[0])] = np.array(v[1:17]).reshape(4,4)
    return poses

def umeyama(pred, gt):
    n = pred.shape[0]
    mu_p = pred.mean(0); mu_g = gt.mean(0)
    p = pred - mu_p; g = gt - mu_g
    cov = g.T @ p / n
    u, s, vh = np.linalg.svd(cov)
    r = u @ vh
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1; r = u @ vh
    var_p = (p**2).sum() / n
    scale = s.sum() / var_p
    return scale * (p @ r.T) + mu_g

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
COLORS = plt.cm.viridis(np.linspace(0.1, 0.9, 5))

for ax, clip in zip(axes, ["static", "dynamic"]):
    gt_full = read_cam0(GT_DIR / f"{clip}_cam0_to_world.txt")
    fids_sorted = sorted(gt_full.keys())
    gt_pts = np.array([gt_full[f][:3, 3] for f in fids_sorted])
    ax.plot(gt_pts[:, 0], gt_pts[:, 1], "k-", lw=2, alpha=0.7, label="GT", zorder=1)

    run_dir = ROOT / "vggt_out" / f"{clip}_5runs"
    for r in range(5):
        npz = run_dir / f"run{r}.npz"
        if not npz.exists():
            continue
        data = np.load(npz)
        extr = data["extrinsics"]
        fids = data["frame_ids"]
        pred_centers = np.array([w2c_to_c2w(extr[i])[:3, 3] for i in range(len(fids))])
        gt_centers = np.array([gt_full[int(f)][:3, 3] for f in fids if int(f) in gt_full])
        if len(gt_centers) != len(pred_centers):
            continue
        aligned = umeyama(pred_centers, gt_centers)
        ax.plot(aligned[:, 0], aligned[:, 1], "-", color=COLORS[r], lw=1.2,
                alpha=0.85, label=f"run {r}", zorder=2)
        ax.scatter(aligned[:, 0], aligned[:, 1], color=COLORS[r], s=10, alpha=0.6, zorder=3)

    ax.set_title(f"{clip.capitalize()} clip", fontsize=12)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"wrote {OUT}")
