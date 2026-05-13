#!/usr/bin/env python3
"""
Convert a multiclass YOLO dataset into a single-class dataset.

All bounding boxes are preserved and every class id is remapped to 0.

Usage:
	python yolo_mc2sc.py --source datasets/CST_AntiUAV/yolo --output datasets/CST_AntiUAV/yolo_sc
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SPLITS = ("", "train", "val", "test")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Convert a multiclass YOLO dataset to single class")
	parser.add_argument(
		"--source",
		type=str,
		default="datasets/YOLO-20260411-QZ_Avia_CST_Hybrid",
		help="Source YOLO dataset root or a single split directory",
	)
	parser.add_argument(
		"--output",
		type=str,
		default="datasets/YOLO-20260411-QZ_Avia_CST_Hybrid_sc",
		help="Output YOLO dataset root",
	)
	parser.add_argument(
		"--class-name",
		type=str,
		default="target",
		help="Name of the merged single class written to data.yaml",
	)
	parser.add_argument(
		"--splits",
		nargs="+",
		default=list(DEFAULT_SPLITS),
		help="Dataset splits to process when source is a dataset root",
	)
	return parser.parse_args()


def detect_splits(source_dir: Path, requested_splits: list[str]) -> list[str]:
	if (source_dir / "images").is_dir() and (source_dir / "labels").is_dir():
		return [""]
	return [split for split in requested_splits if (source_dir / split / "images").is_dir()]


def remap_label_line(line: str) -> str | None:
	parts = line.strip().split()
	if len(parts) < 5:
		return None
	parts[0] = "0"
	return " ".join(parts[:5])


def convert_label_file(source_path: Path, destination_path: Path) -> tuple[int, int]:
	destination_path.parent.mkdir(parents=True, exist_ok=True)
	if not source_path.exists():
		destination_path.write_text("")
		return 0, 0

	converted_lines: list[str] = []
	kept = 0
	with source_path.open() as handle:
		for raw_line in handle:
			raw_line = raw_line.strip()
			if not raw_line:
				continue
			converted = remap_label_line(raw_line)
			if converted is None:
				continue
			converted_lines.append(converted)
			kept += 1

	destination_path.write_text("\n".join(converted_lines) + ("\n" if converted_lines else ""))
	return kept, len(converted_lines)


def copy_image(source_path: Path, destination_path: Path) -> None:
	destination_path.parent.mkdir(parents=True, exist_ok=True)
	shutil.copy2(source_path, destination_path)


def process_split(source_split: Path, output_split: Path) -> tuple[int, int, int]:
	source_images_dir = source_split / "images"
	source_labels_dir = source_split / "labels"
	output_images_dir = output_split / "images"
	output_labels_dir = output_split / "labels"
	output_images_dir.mkdir(parents=True, exist_ok=True)
	output_labels_dir.mkdir(parents=True, exist_ok=True)

	image_files = sorted(p for p in source_images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
	if not image_files:
		return 0, 0, 0

	converted_images = 0
	converted_labels = 0
	total_boxes = 0

	for image_path in image_files:
		destination_image = output_images_dir / image_path.name
		copy_image(image_path, destination_image)
		converted_images += 1

		source_label = source_labels_dir / f"{image_path.stem}.txt"
		destination_label = output_labels_dir / f"{image_path.stem}.txt"
		kept, _ = convert_label_file(source_label, destination_label)
		converted_labels += 1
		total_boxes += kept

	return converted_images, converted_labels, total_boxes


def write_data_yaml(output_dir: Path, class_name: str, split_mode: bool) -> None:
	names_line = f"names:\n  0: {class_name}\n"
	if split_mode:
		content = (
			"path: .\n"
			"train: train/images\n"
			"val: val/images\n"
			"test: test/images\n"
			"nc: 1\n"
			f"{names_line}"
		)
	else:
		content = f"path: .\nnc: 1\n{names_line}"
	(output_dir / "data.yaml").write_text(content)


def main() -> None:
	args = parse_args()
	source_dir = Path(args.source)
	output_dir = Path(args.output)
	output_dir.mkdir(parents=True, exist_ok=True)

	splits = detect_splits(source_dir, args.splits)
	if not splits:
		print(f"No YOLO splits found under: {source_dir}")
		return

	total_images = 0
	total_labels = 0
	total_boxes = 0

	split_mode = splits != [""]
	if not split_mode:
		converted_images, converted_labels, converted_boxes = process_split(source_dir, output_dir)
		total_images += converted_images
		total_labels += converted_labels
		total_boxes += converted_boxes
		print(f"Processed dataset: images={converted_images}, labels={converted_labels}, boxes={converted_boxes}")
	else:
		for split in splits:
			converted_images, converted_labels, converted_boxes = process_split(source_dir / split, output_dir / split)
			total_images += converted_images
			total_labels += converted_labels
			total_boxes += converted_boxes
			print(
				f"Processed split '{split}': images={converted_images}, labels={converted_labels}, boxes={converted_boxes}"
			)

	write_data_yaml(output_dir, args.class_name, split_mode=split_mode)
	print(
		f"Done. Output written to {output_dir} | images={total_images}, labels={total_labels}, boxes={total_boxes}"
	)


if __name__ == "__main__":
	main()
