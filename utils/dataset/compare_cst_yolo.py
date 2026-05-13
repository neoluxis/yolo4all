#!/usr/bin/env python3
"""
Compare bounding boxes between original CST format and converted YOLO format.
This script reads the same image from both sources and draws boxes using the same logic
as visualize_dataset.py for CST and verify_yolo_labels.py for YOLO.

Usage:
    python compare_cst_yolo.py --cst_dir datasets/CST_AntiUAV/CST-AntiUAV \
                               --yolo_dir datasets/CST_AntiUAV/yolo \
                               --split train --scene_id 1 --output compare_result.jpg
"""
import argparse
import json
from pathlib import Path
import os

import cv2
import numpy as np


def parse_cst_gt(gt_path: Path):
    """Parse CST gt.txt - same logic as visualize_dataset.py"""
    if not gt_path.exists():
        return []
    out = []
    with gt_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 4:
                out.append(None)
                continue
            try:
                nums = [float(p) for p in parts[:4]]
            except Exception:
                out.append(None)
                continue
            # CST: x, y, w, h where x,y are top-left
            if nums[0] == 0 and nums[1] == 0 and nums[2] == 0 and nums[3] == 0:
                out.append(None)
            else:
                x, y, w, h = nums
                x1 = int(x)
                y1 = int(y)
                x2 = int(x + w)
                y2 = int(y + h)
                out.append((x1, y1, x2, y2))
    return out


def parse_cst_exist(json_path: Path):
    """Parse exist.txt - same logic as visualize_dataset.py"""
    if not json_path.exists():
        return []
    try:
        j = json.loads(json_path.read_text())
        return j.get('exist', [])
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description="Compare CST and YOLO bounding boxes")
    parser.add_argument(
        "--cst_dir",
        type=str,
        default="datasets/CST_AntiUAV/CST-AntiUAV",
        help="Original CST dataset root",
    )
    parser.add_argument(
        "--yolo_dir",
        type=str,
        default="datasets/CST_AntiUAV/yolo",
        help="Converted YOLO dataset root",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to compare",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default="1",
        help="If set and matches a filename in YOLO images, use it; otherwise first YOLO image is used",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="compare_result.jpg",
        help="Output comparison image path",
    )
    args = parser.parse_args()

    cst_dir = Path(args.cst_dir)
    yolo_dir = Path(args.yolo_dir)

    yolo_split = yolo_dir / args.split
    if not yolo_split.exists():
        print(f"YOLO split not found: {yolo_split}")
        return

    yolo_images_dir = yolo_split / 'images'
    yolo_labels_dir = yolo_split / 'labels'
    if not yolo_images_dir.exists():
        print(f"YOLO images dir not found: {yolo_images_dir}")
        return

    yolo_images = sorted([p for p in yolo_images_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    if not yolo_images:
        print(f"No images found in YOLO split: {yolo_images_dir}")
        return

    # choose image by provided name if it exists in YOLO images, else use first image
    if args.scene_id and (yolo_images_dir / args.scene_id).exists():
        img_name = args.scene_id
    else:
        img_name = yolo_images[0].name

    yolo_img_path = yolo_images_dir / img_name

    # try label file: prefer exact stem, but also try scene-prefixed variants
    label_stem = Path(img_name).stem
    yolo_label_path = yolo_labels_dir / f"{label_stem}.txt"
    # we'll determine cst_scene later; try a few fallbacks

    # try to locate original CST image: resolve symlink target or search CST tree
    cst_img_path = None
    try:
        resolved = Path(os.path.realpath(yolo_img_path))
        if resolved.exists():
            cst_img_path = resolved
    except Exception:
        cst_img_path = None

    if cst_img_path is None:
        matches = list(cst_dir.rglob(img_name))
        if matches:
            cst_img_path = matches[0]

    if cst_img_path is None:
        print(f"Could not locate original CST image for {img_name} under {cst_dir}")
        return

    cst_scene = cst_img_path.parent

    # now that we have cst_scene, try scene-prefixed label name
    if not yolo_label_path.exists():
        candidate = yolo_labels_dir / f"{os.path.basename(cst_scene)}_{label_stem}.txt"
        if candidate.exists():
            yolo_label_path = candidate
        else:
            candidate2 = yolo_labels_dir / f"{img_name}.txt"
            if candidate2.exists():
                yolo_label_path = candidate2

    # Find images in CST scene
    cst_imgs = sorted([p for p in cst_scene.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    if not cst_imgs:
        print(f"No images found in {cst_scene}")
        return

    # Read CST image
    cst_img = cv2.imread(str(cst_img_path))
    if cst_img is None:
        print(f"Failed to read CST image: {cst_img_path}")
        return
    cst_h, cst_w = cst_img.shape[:2]

    # Parse CST annotations
    cst_gt = parse_cst_gt(cst_scene / "gt.txt")
    cst_exist = parse_cst_exist(cst_scene / "IR_label.json")

    # Draw CST boxes for this image index
    try:
        img_index = cst_imgs.index(cst_img_path)
    except ValueError:
        img_index = 0

    print(f"Using image: {img_name} (CST index {img_index})")
    print("\n=== CST Original Boxes ===")
    expected_yolo = None
    if img_index < len(cst_gt) and cst_gt[img_index] is not None:
        x1, y1, x2, y2 = cst_gt[img_index]
        cv2.rectangle(cst_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(cst_img, f"CST: ({x1},{y1})-({x2},{y2})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print(f"  Pixel box ({x1},{y1})-({x2},{y2})")
        # compute expected YOLO normalized values from CST bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = x2 - x1
        bh = y2 - y1
        expected_yolo = (cx / float(cst_w), cy / float(cst_h), bw / float(cst_w), bh / float(cst_h))
        print(f"  Expected YOLO (normalized): cx={expected_yolo[0]:.6f} cy={expected_yolo[1]:.6f} w={expected_yolo[2]:.6f} h={expected_yolo[3]:.6f}")
    else:
        print("  No CST bbox for this image")

    # Read YOLO image
    yolo_img = cv2.imread(str(yolo_img_path))
    if yolo_img is None:
        print(f"Failed to read YOLO image: {yolo_img_path}")
        return
    yolo_h, yolo_w = yolo_img.shape[:2]

    # Parse YOLO annotations and draw
    print("\n=== YOLO Converted Boxes ===")
    print(f"Using YOLO label file: {yolo_label_path}")
    if yolo_label_path.exists():
        with open(yolo_label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    parts = [float(p) for p in line.split()]
                    if len(parts) < 5:
                        continue
                    class_id = int(parts[0])
                    cx_n, cy_n, bw_n, bh_n = parts[1:5]

                    # Denormalize
                    cx = cx_n * yolo_w
                    cy = cy_n * yolo_h
                    bw = bw_n * yolo_w
                    bh = bh_n * yolo_h

                    # Top-left corner
                    x1 = int(cx - bw / 2)
                    y1 = int(cy - bh / 2)
                    x2 = int(cx + bw / 2)
                    y2 = int(cy + bh / 2)

                    cv2.rectangle(yolo_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv2.putText(yolo_img, f"YOLO: ({x1},{y1})-({x2},{y2})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    print(f"  Normalized (from label): cx={cx_n:.6f} cy={cy_n:.6f} w={bw_n:.6f} h={bh_n:.6f}")
                    print(f"  Pixel (from label): ({x1},{y1})-({x2},{y2})")
                    if expected_yolo is not None:
                        print(f"  Delta (normalized): d_cx={(cx_n-expected_yolo[0]):.6f} d_cy={(cy_n-expected_yolo[1]):.6f} d_w={(bw_n-expected_yolo[2]):.6f} d_h={(bh_n-expected_yolo[3]):.6f}")
                except Exception as e:
                    print(f"  Failed to parse line: {line} -> {e}")
    else:
        print(f"  No YOLO label file: {yolo_label_path}")

    # Create comparison image
    max_h = max(cst_h, yolo_h)
    max_w = max(cst_w, yolo_w)
    comparison = np.zeros((max_h + 50, max_w * 2 + 30, 3), dtype=np.uint8)
    comparison.fill(255)

    # Place images side by side
    comparison[20:20+cst_h, 10:10+cst_w] = cst_img
    comparison[20:20+yolo_h, 20+max_w:20+max_w+yolo_w] = yolo_img

    # Add labels
    cv2.putText(comparison, "CST Original", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(comparison, "YOLO Converted", (20+max_w, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Save
    out_path = Path(args.output)
    cv2.imwrite(str(out_path), comparison)
    print(f"\n✓ Comparison saved to: {out_path}")


if __name__ == "__main__":
    main()
