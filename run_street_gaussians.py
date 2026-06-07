import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


def read_colmap_cameras_txt(cameras_txt: Path) -> dict:
    cameras = {}
    with open(cameras_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            cam_id = int(parts[0])
            w, h = int(parts[2]), int(parts[3])
            params = list(map(float, parts[4:]))
            cameras[cam_id] = {"w": w, "h": h, "params": params}
    return cameras


def read_colmap_images_txt(images_txt: Path) -> list:
    images = []
    with open(images_txt) as f:
        lines = [ln for ln in f if not ln.startswith("#") and ln.strip()]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        images.append({
            "qw": float(parts[1]), "qx": float(parts[2]),
            "qy": float(parts[3]), "qz": float(parts[4]),
            "tx": float(parts[5]), "ty": float(parts[6]), "tz": float(parts[7]),
            "name": parts[9],
        })
        i += 2
    return images


def quat_to_rotmat(qw, qx, qy, qz) -> np.ndarray:
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
    ])


def build_sg_data_dir(colmap_dir: Path, clip_dir: Path, sg_data_dir: Path):
    sg_data_dir.mkdir(parents=True, exist_ok=True)

    sparse_src = colmap_dir / "sparse" / "0"
    sparse_dst = sg_data_dir / "sparse" / "0"
    sparse_dst.mkdir(parents=True, exist_ok=True)

    for fname in ("cameras.txt", "images.txt", "points3D.txt"):
        src = sparse_src / fname
        if src.exists():
            shutil.copy2(src, sparse_dst / fname)
        else:
            print(f"missing {src}")

    img_dst = sg_data_dir / "images"
    if img_dst.exists():
        shutil.rmtree(img_dst)
    img_dst.mkdir()
    for png in sorted(clip_dir.glob("*.png")):
        (img_dst / png.name).symlink_to(png.resolve())

    cameras = read_colmap_cameras_txt(sparse_dst / "cameras.txt")
    images = read_colmap_images_txt(sparse_dst / "images.txt")
    cam = list(cameras.values())[0]
    fx, fy, cx, cy = cam["params"][:4]
    w, h = cam["w"], cam["h"]

    frames = []
    for img in sorted(images, key=lambda x: x["name"]):
        R = quat_to_rotmat(img["qw"], img["qx"], img["qy"], img["qz"])
        t = np.array([img["tx"], img["ty"], img["tz"]])
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        frames.append({
            "file_path": f"images/{img['name']}",
            "transform_matrix": c2w.tolist(),
        })

    with open(sg_data_dir / "transforms.json", "w") as f:
        json.dump({
            "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
            "w": w, "h": h, "frames": frames,
        }, f, indent=2)

    print(f"built {sg_data_dir} ({len(frames)} cameras)")


def write_sg_config(
    sg_data_dir: Path,
    sg_out: Path,
    config_path: Path,
    clip_name: str,
):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""task: cs131_{clip_name}
source_path: {sg_data_dir.resolve()}
model_path: {sg_out.resolve()}
exp_name: {clip_name}_sfm
mode: train
to_cuda: true

data:
  split_test: 8
  split_train: 1
  use_colmap: true
  filter_colmap: false
  white_background: false
  extent: 50

model:
  gaussian:
    sh_degree: 3

train:
  iterations: 30000
  test_iterations: [7000, 30000]
  save_iterations: [30000]

optim:
  iterations: 30000
  position_lr_init: 0.00016
  densify_until_iter: 15000
  densify_grad_threshold: 0.0002
"""
    config_path.write_text(text)
    print(f"config: {config_path}")


def launch_training(sg_repo: Path, config_path: Path):
    train_script = sg_repo / "train.py"
    if not train_script.exists():
        print(f"clone street_gaussians first, no {train_script}")
        return False

    cmd = ["python", str(train_script), "--config", str(config_path)]
    print("launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cs131_root", default="/home/shreyas/CS131Final")
    parser.add_argument("--sg_repo", default="/content/street_gaussians")
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()

    root = Path(args.cs131_root)
    sg_repo = Path(args.sg_repo)

    for clip_name in ("static", "dynamic"):
        print(f"\nsg prep: {clip_name}")

        colmap_dir = root / "colmap_out" / clip_name
        clip_dir = root / "clips" / clip_name
        sg_data = root / "sg_data" / clip_name
        sg_out = root / "sg_out" / clip_name
        config = root / "sg_configs" / f"{clip_name}.yaml"

        sparse_cameras = colmap_dir / "sparse" / "0" / "cameras.txt"
        if not sparse_cameras.exists():
            print(f"skip {clip_name}, run run_colmap.py first ({sparse_cameras})")
            continue

        build_sg_data_dir(colmap_dir, clip_dir, sg_data)
        write_sg_config(sg_data, sg_out, config, clip_name)

        if not args.skip_train:
            launch_training(sg_repo, config)

    print("done")


if __name__ == "__main__":
    main()
