import argparse
import time
from pathlib import Path

import pycolmap

# KITTI-360 rectified perspective intrinsics (camera 00)
FX = 552.554261
FY = 552.554261
CX = 682.049453
CY = 238.769549


def run_colmap_on_clip(clip_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "database.db"
    sparse_dir = out_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    if db_path.exists():
        db_path.unlink()

    n_images = len(list(clip_dir.glob("*.png")))
    print(f"\nclip: {clip_dir}")
    print(f"out:  {out_dir}")
    print(f"n:    {n_images}")

    t0 = time.perf_counter()

    print("extracting features")
    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "PINHOLE"
    reader_options.camera_params = f"{FX},{FY},{CX},{CY}"
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(clip_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
    )

    print("matching (overlap=10)")
    pycolmap.match_sequential(
        database_path=str(db_path),
        pairing_options=pycolmap.SequentialPairingOptions(
            overlap=10,
            loop_detection=False,
        ),
    )

    print("mapping")
    maps = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(clip_dir),
        output_path=str(sparse_dir),
        options=pycolmap.IncrementalPipelineOptions(
            min_num_matches=15,
        ),
    )

    elapsed = time.perf_counter() - t0

    if not maps:
        print("no reconstruction")
        return None

    best = max(maps.values(), key=lambda r: r.num_reg_images())
    print(f"{best.num_reg_images()}/{n_images} images, {best.num_points3D()} pts, {elapsed:.1f}s")

    txt_dir = out_dir / "sparse" / "0"
    txt_dir.mkdir(parents=True, exist_ok=True)
    best.write_text(str(txt_dir))

    timing_path = out_dir / "timing.json"
    timing_path.write_text(
        f'{{"wall_seconds": {elapsed:.3f}, "registered_images": {best.num_reg_images()}, '
        f'"total_images": {n_images}, "num_points3d": {best.num_points3D()}}}\n'
    )

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cs131_root", default="/home/shreyas/CS131Final")
    args = parser.parse_args()

    root = Path(args.cs131_root)
    clips = {
        "static": root / "clips" / "static",
        "dynamic": root / "clips" / "dynamic",
    }
    outputs = {
        "static": root / "colmap_out" / "static",
        "dynamic": root / "colmap_out" / "dynamic",
    }

    for name, clip_dir in clips.items():
        if not clip_dir.exists():
            print(f"missing: {clip_dir}")
            continue
        run_colmap_on_clip(clip_dir, outputs[name])

    print("done")


if __name__ == "__main__":
    main()
