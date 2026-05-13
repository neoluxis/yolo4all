#!/usr/bin/env python3
"""
Verify YOLO dataset labels by reading them and drawing back onto images.
This helps diagnose coordinate conversion issues.

Usage:
    python verify_yolo_labels.py --yolo_dir datasets/CST_AntiUAV/yolo --type train --num 5
"""
import argparse
import cv2
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="验证 YOLO 数据集标签是否正确")
    parser.add_argument(
        "--yolo_dir",
        type=str,
        default="datasets/CST_AntiUAV/yolo",
        help="YOLO 数据集根目录",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="要验证的数据集划分",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=5,
        help="要验证的样本数量",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="verify_output",
        help="输出验证图片的目录",
    )
    args = parser.parse_args()

    yolo_dir = Path(args.yolo_dir)
    images_dir = yolo_dir / args.type / "images"
    labels_dir = yolo_dir / args.type / "labels"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.is_dir():
        print(f"错误：图片目录不存在: {images_dir}")
        return
    if not labels_dir.is_dir():
        print(f"错误：标签目录不存在: {labels_dir}")
        return

    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not image_files:
        print(f"错误：{images_dir} 中没有找到图片")
        return

    image_files = image_files[:args.num]

    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"跳过（缺少标签）: {img_path.name}")
            continue

        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"跳过（无法读取图片）: {img_path.name}")
            continue

        h, w = img.shape[:2]

        # 读取标签
        try:
            with open(label_path) as f:
                lines = [l.strip() for l in f if l.strip()]
        except Exception as e:
            print(f"跳过（无法读取标签）: {img_path.name} -> {e}")
            continue

        # 处理每个 bbox
        for line_idx, line in enumerate(lines):
            try:
                parts = [float(p) for p in line.split()]
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                cx_n, cy_n, bw_n, bh_n = parts[1:5]

                # 反向归一化
                cx = cx_n * w
                cy = cy_n * h
                bw = bw_n * w
                bh = bh_n * h

                # 计算左上角
                x1 = int(cx - bw / 2)
                y1 = int(cy - bh / 2)
                x2 = int(cx + bw / 2)
                y2 = int(cy + bh / 2)

                # 画框
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"c{class_id}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                print(
                    f"  Box {line_idx}: class={class_id}, "
                    f"norm=(cx={cx_n:.4f}, cy={cy_n:.4f}, w={bw_n:.4f}, h={bh_n:.4f}) -> "
                    f"pixel=({x1}, {y1}, {x2}, {y2})"
                )
            except Exception as e:
                print(f"  跳过行: {line} -> {e}")

        # 保存验证图片
        out_path = output_dir / f"{img_path.stem}_verified.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"✓ 已保存: {out_path}")
        print()


if __name__ == "__main__":
    main()
