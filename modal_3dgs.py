from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import modal

LOCAL_ROOT = Path(__file__).resolve().parent
CLIPS = ("static", "dynamic")
CUDA_TAG = "12.1.1-devel-ubuntu22.04"  # needs nvcc to build 3dgs submodules

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.10")
    .apt_install(
        "git", "build-essential", "g++", "cmake", "ninja-build",
        "libglm-dev", "clang",
        # opencv runtime
        "libgl1", "libglib2.0-0",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64",
            "CC": "gcc",
            "CXX": "g++",
            "TORCH_CUDA_ARCH_LIST": "8.6;8.9;9.0",
        }
    )
    .pip_install(
        "torch==2.1.0",
        "torchvision==0.16.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "plyfile",
        "tqdm",
        "opencv-python-headless",
        "joblib",
        "scikit-image",
        "lpips",
        "numpy<2",
        "Pillow",
        "setuptools",
        "wheel",
        "ninja",
    )
    .run_commands(
        # plain clone; the upstream .gitmodules points at gitlab.inria.fr which is flaky
        # from modal, so we pull the three submodules we need from github mirrors and
        # skip sibr_viewers
        "git clone https://github.com/graphdeco-inria/gaussian-splatting.git /gs",
        "git clone --branch dr_aa https://github.com/graphdeco-inria/diff-gaussian-rasterization.git "
        "/gs/submodules/diff-gaussian-rasterization",
        "cd /gs/submodules/diff-gaussian-rasterization && git submodule update --init --recursive",
        "git clone https://github.com/mjkaufer/simple-knn.git /gs/submodules/simple-knn",
        "git clone https://github.com/rahul-goel/fused-ssim.git /gs/submodules/fused-ssim",
        # --no-build-isolation so torch is visible at compile time
        "cd /gs && pip install --no-build-isolation ./submodules/diff-gaussian-rasterization",
        "cd /gs && pip install --no-build-isolation ./submodules/simple-knn",
        "cd /gs && pip install --no-build-isolation ./submodules/fused-ssim",
    )
)

app = modal.App("cs131-3dgs")

data_vol = modal.Volume.from_name("cs131-3dgs-vol", create_if_missing=True)
VOL = "/vol"


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 3,
    volumes={VOL: data_vol},
)
def train_one_clip(clip: str, iters: int = 30000) -> dict:
    import subprocess
    import os

    src = f"{VOL}/data/{clip}"
    out = f"{VOL}/out/{clip}"
    Path(out).mkdir(parents=True, exist_ok=True)

    n_imgs = len(list((Path(src) / "images").glob("*.png")))
    sparse_files = sorted(p.name for p in (Path(src) / "sparse" / "0").iterdir())
    print(f"\n{clip}: {n_imgs} images, sparse={sparse_files}")

    t0 = time.perf_counter()
    subprocess.run(
        [
            "python", "/gs/train.py",
            "-s", src,
            "-m", out,
            "--eval",
            "--iterations", str(iters),
            "--test_iterations", str(iters),
            "--save_iterations", str(iters),
            "--resolution", "1",
        ],
        check=True,
    )
    train_seconds = time.perf_counter() - t0

    subprocess.run(
        ["python", "/gs/render.py", "-m", out, "--skip_train"],
        check=True,
    )
    subprocess.run(["python", "/gs/metrics.py", "-m", out], check=True)

    metrics_path = Path(out) / "results.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    (Path(out) / "timing.json").write_text(
        json.dumps(
            {"clip": clip, "iterations": iters, "train_seconds": train_seconds, "n_images": n_imgs},
            indent=2,
        )
    )
    data_vol.commit()

    print(f"\n{clip} done in {train_seconds:.1f}s")
    print(json.dumps(metrics, indent=2))
    return {"clip": clip, "metrics": metrics, "train_seconds": train_seconds}


def _stage_local() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="cs131_modal_"))
    print(f"staging into {tmp}")
    for clip in CLIPS:
        src_imgs = LOCAL_ROOT / "clips" / clip
        src_sparse = LOCAL_ROOT / "colmap_out" / clip / "sparse" / "0"
        dst_imgs = tmp / "data" / clip / "images"
        dst_sparse = tmp / "data" / clip / "sparse" / "0"
        dst_imgs.mkdir(parents=True, exist_ok=True)
        dst_sparse.mkdir(parents=True, exist_ok=True)
        for p in src_imgs.glob("*.png"):
            shutil.copy2(p, dst_imgs / p.name)
        for fname in ("cameras.bin", "images.bin", "points3D.bin"):
            shutil.copy2(src_sparse / fname, dst_sparse / fname)
    return tmp


@app.local_entrypoint()
def main(clip: str = "both", iters: int = 30000, skip_upload: bool = False):
    clips_to_run = CLIPS if clip == "both" else (clip,)

    if not skip_upload:
        staged = _stage_local()
        print("uploading to cs131-3dgs-vol")
        with data_vol.batch_upload(force=True) as batch:
            batch.put_directory(str(staged / "data"), "/data")
        print("upload done")
        shutil.rmtree(staged, ignore_errors=True)

    summary = {}
    for c in clips_to_run:
        print(f"\ntraining {c} for {iters} iters")
        summary[c] = train_one_clip.remote(c, iters)

    print("\ndone")
    print(json.dumps(summary, indent=2, default=str))
    print("\npull outputs:")
    print(f"  modal volume get cs131-3dgs-vol /out ./gs_out --recursive")
