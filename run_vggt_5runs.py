# Five VGGT runs per clip, with slightly perturbed (+/- 2 frame) subsamples so each
# run sees a different 80-frame window. The model itself is deterministic at inference;
# the perturbation is what would happen across re-runs with a different sampling
# decision. Pass --identical to force the same subsample every time.

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def frame_id_from_path(p):
    m = re.search(r"(\d+)\.png$", Path(p).name)
    return int(m.group(1)) if m else -1


def subsample(image_paths, n, seed):
    if seed is None:
        idx = np.linspace(0, len(image_paths) - 1, n, dtype=int)
    else:
        rng = np.random.default_rng(seed)
        idx = np.linspace(0, len(image_paths) - 1, n).astype(int)
        jitter = rng.integers(-2, 3, size=n)
        idx = np.clip(idx + jitter, 0, len(image_paths) - 1)
        idx = np.sort(np.unique(idx))
        while len(idx) < n:
            extra = rng.integers(0, len(image_paths))
            if extra not in idx:
                idx = np.sort(np.append(idx, extra))
    return [image_paths[i] for i in idx]


def run_once(model, clip_dir, n_frames, seed, run_idx):
    image_paths = sorted(str(p) for p in clip_dir.glob("*.png"))
    sampled = subsample(image_paths, n_frames, seed)
    frame_ids = np.array([frame_id_from_path(p) for p in sampled], dtype=np.int64)

    images = load_and_preprocess_images(sampled).to(DEVICE).unsqueeze(0)
    if DEVICE == "cuda":
        images = images.to(dtype=torch.bfloat16)

    print(f"  run {run_idx}: {len(sampled)} frames, seed={seed}")
    t0 = time.perf_counter()
    with torch.no_grad():
        if DEVICE == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred = model(images)
        else:
            pred = model(images.float())
    elapsed = time.perf_counter() - t0
    print(f"  run {run_idx}: {elapsed:.1f}s")

    extrinsics, _ = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    extrinsics = extrinsics.squeeze(0).cpu().numpy()
    return frame_ids, extrinsics, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cs131_root", default="/Users/aadichauhan/Developer/CS131Final")
    ap.add_argument("--n_frames", type=int, default=80)
    ap.add_argument("--n_runs", type=int, default=5)
    ap.add_argument("--clip", choices=("static", "dynamic", "both"), default="both")
    ap.add_argument("--identical", action="store_true")
    args = ap.parse_args()

    print(f"device: {DEVICE}")
    print(f"loading VGGT-1B")
    t0 = time.perf_counter()
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model.eval().to(DEVICE)
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    root = Path(args.cs131_root)
    clips = ("static", "dynamic") if args.clip == "both" else (args.clip,)

    for name in clips:
        clip_dir = root / "clips" / name
        out_dir = root / "vggt_out" / f"{name}_5runs"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{name}")
        timings = []
        for r in range(args.n_runs):
            seed = None if args.identical else (1000 + r)
            frame_ids, extrinsics, t = run_once(model, clip_dir, args.n_frames, seed, r)
            timings.append(t)
            np.savez(
                out_dir / f"run{r}.npz",
                extrinsics=extrinsics,
                frame_ids=frame_ids,
                seed=seed if seed is not None else -1,
            )
        (out_dir / "timings.json").write_text(json.dumps({
            "wall_seconds_per_run": timings,
            "device": DEVICE,
            "n_frames": args.n_frames,
            "n_runs": args.n_runs,
            "identical_input": args.identical,
        }, indent=2))
        print(f"  mean {np.mean(timings):.1f}s  std {np.std(timings):.1f}s")


if __name__ == "__main__":
    main()
