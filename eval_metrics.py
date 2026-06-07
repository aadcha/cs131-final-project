import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

try:
    import lpips
except ImportError:
    lpips = None


def load_image_np(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def compute_psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(psnr(gt, pred, data_range=1.0))


def compute_ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(ssim(gt, pred, data_range=1.0, channel_axis=-1))


def compute_lpips(gt: np.ndarray, pred: np.ndarray, lpips_fn) -> float:
    def to_tensor(img):
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        return t * 2 - 1

    with torch.no_grad():
        score = lpips_fn(to_tensor(gt), to_tensor(pred))
    return float(score.item())


def eval_nvs(gt_dir: Path, pred_dir: Path, method_name: str) -> dict:
    if lpips is None:
        raise ImportError("lpips is required for NVS metrics: pip install lpips")

    lpips_fn = lpips.LPIPS(net="alex")

    gt_paths = sorted(gt_dir.glob("*.png"))
    pred_paths = sorted(pred_dir.glob("*.png"))

    if not gt_paths:
        print(f"  no gt in {gt_dir}")
        return {}
    if not pred_paths:
        print(f"  no renders in {pred_dir}")
        return {}

    gt_names = {p.name: p for p in gt_paths}
    pred_names = {p.name: p for p in pred_paths}
    common = sorted(set(gt_names) & set(pred_names))

    if not common:
        print(f"  no matching filenames in {gt_dir} vs {pred_dir}")
        return {}

    print(f"  nvs {method_name}: {len(common)} frames")

    psnr_scores, ssim_scores, lpips_scores = [], [], []

    for name in common:
        gt = load_image_np(gt_names[name])
        pred = load_image_np(pred_names[name])

        if gt.shape != pred.shape:
            pred_pil = Image.fromarray((pred * 255).astype(np.uint8))
            pred_pil = pred_pil.resize((gt.shape[1], gt.shape[0]), Image.LANCZOS)
            pred = np.array(pred_pil, dtype=np.float32) / 255.0

        psnr_scores.append(compute_psnr(gt, pred))
        ssim_scores.append(compute_ssim(gt, pred))
        lpips_scores.append(compute_lpips(gt, pred, lpips_fn))

    return {
        "method": method_name,
        "n_frames": len(common),
        "PSNR": float(np.mean(psnr_scores)),
        "SSIM": float(np.mean(ssim_scores)),
        "LPIPS": float(np.mean(lpips_scores)),
    }


def frame_id_from_name(name: str) -> int:
    m = re.search(r"(\d+)\.png$", name)
    return int(m.group(1)) if m else -1


def read_cam0_to_world(cam0_txt: Path) -> dict:
    # frame_id + 16 floats (4x4 cam-to-world)
    poses = {}
    with open(cam0_txt) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) < 17:
                continue
            fid = int(vals[0])
            poses[fid] = np.array(vals[1:17], dtype=np.float64).reshape(4, 4)
    return poses


def quat_to_rotmat(qw, qx, qy, qz) -> np.ndarray:
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
    ])


def w2c_to_c2w(extrinsic_3x4: np.ndarray) -> np.ndarray:
    R = extrinsic_3x4[:3, :3]
    t = extrinsic_3x4[:3, 3]
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w


def camera_center(c2w: np.ndarray) -> np.ndarray:
    return c2w[:3, 3].copy()


def umeyama_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    # Sim(3) alignment, Nx3 to Nx3
    assert pred.shape == gt.shape and pred.shape[1] == 3
    n = pred.shape[0]

    mu_p = pred.mean(axis=0)
    mu_g = gt.mean(axis=0)
    p_c = pred - mu_p
    g_c = gt - mu_g

    cov = g_c.T @ p_c / n
    u, s, vh = np.linalg.svd(cov)
    r = u @ vh
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vh

    var_p = (p_c**2).sum() / n
    scale = np.trace(np.diag(s)) / var_p if var_p > 1e-12 else 1.0
    aligned = scale * (p_c @ r.T) + mu_g
    return aligned


def compute_ate(pred_centers: np.ndarray, gt_centers: np.ndarray) -> float:
    aligned = umeyama_align(pred_centers, gt_centers)
    errors = np.linalg.norm(aligned - gt_centers, axis=1)
    return float(np.sqrt(np.mean(errors**2)))


def read_colmap_poses(images_txt: Path) -> dict:
    poses = {}
    with open(images_txt) as f:
        lines = [ln for ln in f if not ln.startswith("#") and ln.strip()]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9]
        R = quat_to_rotmat(qw, qx, qy, qz)
        w2c = np.hstack([R, np.array([[tx], [ty], [tz]])])
        fid = frame_id_from_name(name)
        if fid >= 0:
            poses[fid] = w2c_to_c2w(w2c)
        i += 2
    return poses


def eval_ate(pred_poses: dict, gt_poses: dict, method_name: str) -> dict:
    common = sorted(set(pred_poses) & set(gt_poses))
    if len(common) < 2:
        print(f"  ate: need >=2 matching frames for {method_name}, got {len(common)}")
        return {}

    pred_c = np.array([camera_center(pred_poses[f]) for f in common])
    gt_c = np.array([camera_center(gt_poses[f]) for f in common])
    ate = compute_ate(pred_c, gt_c)

    return {
        "method": method_name,
        "n_frames": len(common),
        "ATE_m": ate,
    }


def eval_colmap_ate(root: Path, gt_poses: dict, clip: str) -> dict:
    images_txt = root / "colmap_out" / clip / "sparse" / "0" / "images.txt"
    if not images_txt.exists():
        print(f"  missing colmap poses: {images_txt}")
        return {}
    pred = read_colmap_poses(images_txt)
    return eval_ate(pred, gt_poses, f"colmap/{clip}")


def eval_vggt_ate(root: Path, gt_poses: dict, clip: str) -> dict:
    npz_path = root / "vggt_out" / clip / "cameras.npz"
    if not npz_path.exists():
        print(f"  missing vggt output: {npz_path}")
        return {}

    data = np.load(npz_path)
    extrinsics = data["extrinsics"]  # [N, 3, 4]
    if "frame_ids" in data:
        frame_ids = data["frame_ids"].astype(int)
    else:
        frame_ids = np.array(
            [frame_id_from_name(str(p)) for p in data["image_paths"]]
        )

    pred = {}
    for i, fid in enumerate(frame_ids):
        if fid >= 0:
            pred[int(fid)] = w2c_to_c2w(extrinsics[i])
    return eval_ate(pred, gt_poses, f"vggt/{clip}")


def load_gt_for_clip(root: Path, clip: str) -> dict:
    per_clip = root / "data" / "gt_poses" / f"{clip}_cam0_to_world.txt"
    combined = root / "data" / "gt_poses" / "cam0_to_world.txt"
    cam0_txt = per_clip if per_clip.exists() else combined
    if not cam0_txt.exists():
        raise FileNotFoundError(f"no gt poses at {per_clip} or {combined}")
    return read_cam0_to_world(cam0_txt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cs131_root", default="/home/shreyas/CS131Final")
    parser.add_argument("--ate_only", action="store_true")
    args = parser.parse_args()

    root = Path(args.cs131_root)
    results = {"nvs": [], "ate": []}

    for clip in ("static", "dynamic"):
        try:
            gt_poses = load_gt_for_clip(root, clip)
        except FileNotFoundError as e:
            print(e)
            continue

        print(f"\n{clip}  ({len(gt_poses)} gt poses)")

        ate_colmap = eval_colmap_ate(root, gt_poses, clip)
        if ate_colmap:
            print(f"  colmap ate {ate_colmap['ATE_m']:.4f}m  ({ate_colmap['n_frames']} frames)")
            results["ate"].append(ate_colmap)

        ate_vggt = eval_vggt_ate(root, gt_poses, clip)
        if ate_vggt:
            print(f"  vggt ate   {ate_vggt['ATE_m']:.4f}m  ({ate_vggt['n_frames']} frames)")
            results["ate"].append(ate_vggt)

        if args.ate_only:
            continue

        gt_dir = root / "clips" / clip
        renders_root = root / "renders"
        if renders_root.exists():
            for method_dir in sorted(renders_root.iterdir()):
                pred_dir = method_dir / clip
                if pred_dir.is_dir():
                    nvs = eval_nvs(gt_dir, pred_dir, f"{method_dir.name}/{clip}")
                    if nvs:
                        results["nvs"].append(nvs)
                        print(
                            f"  {method_dir.name} psnr={nvs['PSNR']:.2f} "
                            f"ssim={nvs['SSIM']:.3f} lpips={nvs['LPIPS']:.3f}"
                        )

    out_path = root / "metrics_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
