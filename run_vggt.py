import argparse
import json
import re
import time
import numpy as np
from pathlib import Path
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.bfloat16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32
else:
    DEVICE = "cpu"
    DTYPE = torch.float32
DEFAULT_MAX_FRAMES = 80


def subsample_frames(image_paths: list, max_frames: int) -> list:
    if len(image_paths) <= max_frames:
        return image_paths
    indices = np.linspace(0, len(image_paths) - 1, max_frames, dtype=int)
    return [image_paths[i] for i in indices]


def frame_id_from_path(path: str) -> int:
    m = re.search(r"(\d+)\.png$", Path(path).name)
    return int(m.group(1)) if m else -1


def run_vggt_on_clip(clip_dir: Path, out_dir: Path, model, max_frames: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(str(p) for p in clip_dir.glob("*.png"))
    print(f"\n{clip_dir.name}: {len(image_paths)} frames")

    sampled = subsample_frames(image_paths, max_frames)
    frame_ids = np.array([frame_id_from_path(p) for p in sampled], dtype=np.int64)
    print(f"using {len(sampled)} frames")

    images = load_and_preprocess_images(sampled)
    images = images.to(DEVICE)
    if DEVICE == "cuda":
        images = images.to(dtype=DTYPE)
    images = images.unsqueeze(0)

    print(f"running on {DEVICE}")
    t0 = time.perf_counter()
    with torch.no_grad():
        if DEVICE == "cuda":
            with torch.cuda.amp.autocast(dtype=DTYPE):
                predictions = model(images)
        else:
            predictions = model(images.float())
    elapsed = time.perf_counter() - t0
    print(f"forward: {elapsed:.2f}s")

    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    extrinsics = extrinsics.squeeze(0).cpu().numpy()
    intrinsics = intrinsics.squeeze(0).cpu().numpy()

    world_points = predictions["world_points"].squeeze(0).cpu().numpy()
    world_conf = predictions["world_points_conf"].squeeze(0).cpu().numpy()

    pts = world_points.reshape(-1, 3)
    conf = world_conf.reshape(-1)
    pts_filtered = pts[conf > 0.5]
    print(f"pts: {len(pts_filtered)} (conf > 0.5)")

    np.savez(
        out_dir / "cameras.npz",
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_paths=np.array(sampled),
        frame_ids=frame_ids,
    )
    np.savez(
        out_dir / "points3d.npz",
        points=pts_filtered,
        all_points=pts,
        confidence=conf,
    )
    (out_dir / "timing.json").write_text(
        json.dumps({
            "wall_seconds": elapsed,
            "device": DEVICE,
            "n_frames": len(sampled),
            "n_points": int(len(pts_filtered)),
        }, indent=2)
    )

    print(f"saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cs131_root", default="/content/CS131Final")
    parser.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES)
    args = parser.parse_args()

    root = Path(args.cs131_root)

    print("loading VGGT-1B")
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model.eval().to(DEVICE)
    print(f"loaded on {DEVICE}")

    clips = {
        "static": root / "clips" / "static",
        "dynamic": root / "clips" / "dynamic",
    }

    for name, clip_dir in clips.items():
        if not clip_dir.exists():
            print(f"skip {name}: no {clip_dir}")
            continue
        run_vggt_on_clip(
            clip_dir,
            root / "vggt_out" / name,
            model,
            args.max_frames,
        )

    print("done")


if __name__ == "__main__":
    main()
