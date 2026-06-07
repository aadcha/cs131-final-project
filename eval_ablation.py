import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_metrics import (
    read_cam0_to_world,
    read_colmap_poses,
    camera_center,
    compute_ate,
    w2c_to_c2w,
    frame_id_from_name,
)

ROOT = Path("/Users/aadichauhan/Developer/CS131Final")
GT_DIR = ROOT / "data" / "gt_poses"


def load_gt(clip):
    per_clip = GT_DIR / f"{clip}_cam0_to_world.txt"
    return read_cam0_to_world(per_clip)


def colmap_ate(images_txt, gt_poses):
    pred = read_colmap_poses(images_txt)
    common = sorted(set(pred) & set(gt_poses))
    pred_c = np.array([camera_center(pred[f]) for f in common])
    gt_c   = np.array([camera_center(gt_poses[f]) for f in common])
    return compute_ate(pred_c, gt_c), len(common)


print("\ncolmap overlap ablation")
ablation = {}
for clip in ("static", "dynamic"):
    gt = load_gt(clip)
    ablation[clip] = {}
    for overlap in (5, 10, 20):
        if overlap == 10:
            images_txt = ROOT / "colmap_out" / clip / "sparse" / "0" / "images.txt"
        else:
            images_txt = ROOT / "colmap_out" / f"{clip}_overlap{overlap}" / "sparse" / "0" / "images.txt"
        if not images_txt.exists():
            print(f"  missing: {images_txt}")
            continue
        ate, n = colmap_ate(images_txt, gt)
        if overlap == 10:
            timing_path = ROOT / "colmap_out" / clip / "timing.json"
        else:
            timing_path = ROOT / "colmap_out" / f"{clip}_overlap{overlap}" / "timing.json"
        wall = json.loads(timing_path.read_text())["wall_seconds"] if timing_path.exists() else None
        ablation[clip][overlap] = {"ATE_m": ate, "n_frames": n, "wall_s": wall}
        print(f"  {clip:8s} overlap={overlap:2d}: ate={ate:.4f}m  ({n} frames, {wall:.1f}s)")

(ROOT / "ablation_results.json").write_text(json.dumps(ablation, indent=2))
print(f"\nwrote ablation_results.json")


print("\nvggt 5-run ate std")
vggt_results = {}
for clip in ("static", "dynamic"):
    run_dir = ROOT / "vggt_out" / f"{clip}_5runs"
    if not run_dir.exists():
        print(f"  skip {clip}: no {run_dir}")
        continue
    gt = load_gt(clip)
    ates = []
    for r in range(5):
        npz = run_dir / f"run{r}.npz"
        if not npz.exists():
            print(f"  missing: {npz}")
            continue
        data = np.load(npz)
        extr = data["extrinsics"]
        fids = data["frame_ids"]
        pred = {int(fid): w2c_to_c2w(extr[i]) for i, fid in enumerate(fids) if fid >= 0}
        common = sorted(set(pred) & set(gt))
        pred_c = np.array([camera_center(pred[f]) for f in common])
        gt_c   = np.array([camera_center(gt[f]) for f in common])
        ate = compute_ate(pred_c, gt_c)
        ates.append(ate)
        print(f"  {clip:8s} run{r}: ate={ate:.4f}m  ({len(common)} frames)")
    if ates:
        ates = np.array(ates)
        vggt_results[clip] = {
            "runs": ates.tolist(),
            "mean": float(ates.mean()),
            "std":  float(ates.std(ddof=1)),
        }
        print(f"  {clip:8s} mean={ates.mean():.3f}m  std={ates.std(ddof=1):.3f}m")

if vggt_results:
    (ROOT / "vggt_5runs_results.json").write_text(json.dumps(vggt_results, indent=2))
    print(f"\nwrote vggt_5runs_results.json")
