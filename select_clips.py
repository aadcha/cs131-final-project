import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

CLIP_LEN = 200
TOTAL_FRAMES = 11518
DYNAMIC_LABELS = ("car", "truck", "bus", "person", "bicycle", "motorcycle")


def find_clips(xml_path: Path, clip_len: int, total_frames: int):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    frame_counts = defaultdict(int)
    for obj in root:
        dynamic = obj.findtext("dynamic")
        label = obj.findtext("label")
        start = obj.findtext("start_frame")
        end = obj.findtext("end_frame")
        if None in (dynamic, start, end):
            continue
        if dynamic == "1" or label in DYNAMIC_LABELS:
            for f in range(int(start), int(end) + 1):
                frame_counts[f] += 1

    best_static = (0, 0, 999999)
    best_dynamic = (0, 0, 0)

    for start in range(0, total_frames - clip_len, 10):
        end = start + clip_len
        score = sum(frame_counts[f] for f in range(start, end))
        if score < best_static[2]:
            best_static = (start, end, score)
        if score > best_dynamic[2]:
            best_dynamic = (start, end, score)

    return best_static, best_dynamic


def main():
    parser = argparse.ArgumentParser(description="Pick static/dynamic KITTI-360 clips")
    parser.add_argument("--cs131_root", type=Path, default=Path("/home/shreyas/CS131Final"))
    parser.add_argument("--xml", type=Path, default=None)
    args = parser.parse_args()

    root = args.cs131_root
    clip_info_path = root / "data" / "clip_info.json"

    if args.xml is None and clip_info_path.exists():
        import json

        info = json.loads(clip_info_path.read_text())
        for name in ("static", "dynamic"):
            c = info["clips"][name]
            print(
                f"{name.upper():7} clip: frames {c['start_frame']:05d} - {c['end_frame']:05d}  "
                f"({c['n_frames']} frames in clips/{name}/)"
            )
        return

    xml_path = args.xml or (
        root / "data_3d_bboxes" / "train" / "2013_05_28_drive_0000_sync.xml"
    )
    if not xml_path.exists():
        raise FileNotFoundError(f"Annotations not found: {xml_path}")

    best_static, best_dynamic = find_clips(xml_path, CLIP_LEN, TOTAL_FRAMES)

    print(f"STATIC  clip: frames {best_static[0]:05d} - {best_static[1]:05d}  (score: {best_static[2]})")
    print(f"DYNAMIC clip: frames {best_dynamic[0]:05d} - {best_dynamic[1]:05d}  (score: {best_dynamic[2]})")


if __name__ == "__main__":
    main()
