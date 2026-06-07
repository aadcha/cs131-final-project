import argparse
import time
from pathlib import Path

import pycolmap

FX = 552.554261
FY = 552.554261
CX = 682.049453
CY = 238.769549


def run_colmap_on_clip(clip_dir: Path, out_dir: Path, overlap: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "database.db"
    sparse_dir = out_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    n_images = len(list(clip_dir.glob("*.png")))
    print(f"\n{clip_dir.name}  overlap={overlap}  n={n_images}")

    t0 = time.perf_counter()

    print("features")
    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "PINHOLE"
    reader_options.camera_params = f"{FX},{FY},{CX},{CY}"
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(clip_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
    )

    print(f"matching overlap={overlap}")
    pycolmap.match_sequential(
        database_path=str(db_path),
        pairing_options=pycolmap.SequentialPairingOptions(
            overlap=overlap,
            loop_detection=False,
        ),
    )

    print("mapping")
    maps = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(clip_dir),
        output_path=str(sparse_dir),
        options=pycolmap.IncrementalPipelineOptions(min_num_matches=15),
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

    (out_dir / "timing.json").write_text(
        f'{{"wall_seconds": {elapsed:.3f}, '
        f'"registered_images": {best.num_reg_images()}, '
        f'"total_images": {n_images}, '
        f'"num_points3d": {best.num_points3D()}, '
        f'"overlap": {overlap}}}\n'
    )
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cs131_root", default="/Users/aadichauhan/Developer/CS131Final")
    parser.add_argument("--overlap", type=int, required=True)
    parser.add_argument("--clip", choices=("static", "dynamic", "both"), default="both")
    args = parser.parse_args()

    root = Path(args.cs131_root)
    clips = ("static", "dynamic") if args.clip == "both" else (args.clip,)

    for name in clips:
        clip_dir = root / "clips" / name
        out_dir = root / "colmap_out" / f"{name}_overlap{args.overlap}"
        if not clip_dir.exists():
            print(f"skip {name}, no {clip_dir}")
            continue
        run_colmap_on_clip(clip_dir, out_dir, args.overlap)


if __name__ == "__main__":
    main()
