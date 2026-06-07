# CS131 Final: COLMAP vs VGGT vs Street Gaussians

Comparative study on KITTI-360 driving clips (static vs dynamic scenes).

**Group:** Shreyas Anand, Aadi Chauhan, Arin Parsa  

## Data layout 

```
CS131Final/
  clips/static/          # 201 frames (KITTI 4650-4850)
  clips/dynamic/         # 201 frames (KITTI 9700-9900)
  data/clip_info.json    # frame ranges + metadata
  data/gt_poses/         # cam0_to_world GT for those 402 frames
  calibration/           # KITTI-360 intrinsics (for COLMAP)
```

Full KITTI-360 sequences are **not** stored in this repo (~9GB+). Re-download from [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) only if you need to re-pick clips (`select_clips.py --xml ...`).

## Scripts

| Script | Purpose |
|--------|---------|
| `select_clips.py` | Show chosen clip ranges (`data/clip_info.json`) |
| `run_colmap.py` | COLMAP SfM on `clips/*` |
| `run_vggt.py` | VGGT (GPU / Colab) |
| `run_street_gaussians.py` | COLMAP -> Street Gaussians data prep |
| `eval_metrics.py` | ATE vs `data/gt_poses/`, NVS from `renders/` |

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
export ROOT=/home/shreyas/CS131Final

python run_colmap.py --cs131_root $ROOT
python run_vggt.py --cs131_root $ROOT          # GPU
python run_street_gaussians.py --cs131_root $ROOT --skip_train
python eval_metrics.py --cs131_root $ROOT --ate_only
```

Put NVS renders in `renders/<method>/<static|dynamic>/*.png` (filenames must match `clips/`).
