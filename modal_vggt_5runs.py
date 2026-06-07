import json
import shutil
import tempfile
import time
from pathlib import Path

import modal

LOCAL_ROOT = Path(__file__).resolve().parent
CLIPS = ("static", "dynamic")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "numpy<2",
        "Pillow",
        "huggingface_hub",
        "transformers",
        "accelerate",
        "safetensors",
    )
    .pip_install("git+https://github.com/facebookresearch/vggt.git")
)

app = modal.App("cs131-vggt-5runs")
vol = modal.Volume.from_name("cs131-vggt-vol", create_if_missing=True)
VOL = "/vol"
N_FRAMES = 80
N_RUNS = 5


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 60,
    volumes={VOL: vol},
)
def run_clip(clip: str) -> dict:
    import numpy as np
    import torch
    import re
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    clip_dir = Path(f"{VOL}/data/{clip}")
    out_dir = Path(f"{VOL}/out/{clip}")
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(str(p) for p in clip_dir.glob("*.png"))
    print(f"{clip}: {len(image_paths)} frames")

    def frame_id(p):
        m = re.search(r"(\d+)\.png$", Path(p).name)
        return int(m.group(1)) if m else -1

    def subsample(seed):
        idx = np.linspace(0, len(image_paths) - 1, N_FRAMES).astype(int)
        if seed is None:
            return [image_paths[i] for i in idx]
        rng = np.random.default_rng(seed)
        jitter = rng.integers(-2, 3, size=N_FRAMES)
        idx = np.clip(idx + jitter, 0, len(image_paths) - 1)
        idx = np.sort(np.unique(idx))
        while len(idx) < N_FRAMES:
            extra = rng.integers(0, len(image_paths))
            if extra not in idx:
                idx = np.sort(np.append(idx, extra))
        return [image_paths[i] for i in idx]

    print("loading VGGT-1B")
    t0 = time.perf_counter()
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model.eval().to("cuda")
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    timings = []
    for r in range(N_RUNS):
        seed = 1000 + r
        sampled = subsample(seed)
        frame_ids = np.array([frame_id(p) for p in sampled], dtype=np.int64)
        images = load_and_preprocess_images(sampled).to("cuda").to(torch.bfloat16).unsqueeze(0)
        print(f"  run {r}: {len(sampled)} frames, seed={seed}")
        t_run = time.perf_counter()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            pred = model(images)
        extr, _ = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
        extr = extr.squeeze(0).cpu().numpy()
        t_elapsed = time.perf_counter() - t_run
        timings.append(t_elapsed)
        print(f"  run {r}: {t_elapsed:.1f}s")
        np.savez(out_dir / f"run{r}.npz", extrinsics=extr, frame_ids=frame_ids, seed=seed)

    (out_dir / "timings.json").write_text(json.dumps({
        "wall_seconds_per_run": timings,
        "device": "cuda-T4",
        "n_frames": N_FRAMES,
        "n_runs": N_RUNS,
    }, indent=2))
    vol.commit()
    print(f"{clip} done, mean {sum(timings)/len(timings):.1f}s/run")
    return {"clip": clip, "timings": timings}


def _stage_local():
    tmp = Path(tempfile.mkdtemp(prefix="cs131_vggt_"))
    for clip in CLIPS:
        src = LOCAL_ROOT / "clips" / clip
        dst = tmp / "data" / clip
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.glob("*.png"):
            shutil.copy2(p, dst / p.name)
    return tmp


@app.local_entrypoint()
def main(skip_upload: bool = False):
    if not skip_upload:
        staged = _stage_local()
        print(f"uploading from {staged}")
        with vol.batch_upload(force=True) as batch:
            batch.put_directory(str(staged / "data"), "/data")
        print("upload done")
        shutil.rmtree(staged, ignore_errors=True)

    summary = {}
    for c in CLIPS:
        print(f"\n{c}")
        summary[c] = run_clip.remote(c)
    print(json.dumps(summary, indent=2, default=str))
