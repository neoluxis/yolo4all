import argparse
import os
import random
import shutil
from pathlib import Path


def parse_args():
	parser = argparse.ArgumentParser(
		description="从 YOLO 数据集中按划分抽样指定数量的图片和标签。"
	)
	parser.add_argument(
		"--input_dir",
		type=str,
		default="datasets/CST_AntiUAV/yolo",
		help="输入 YOLO 数据集根目录，包含 train、val、test 目录。",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="datasets/CST_AntiUAV/yolo_sampled",
		help="输出采样后 YOLO 数据集根目录。",
	)
	parser.add_argument(
		"--type",
		type=str,
		default="train",
		choices=["train", "val", "test"],
		help="要抽样的数据集划分。",
	)
	parser.add_argument(
		"--number",
		type=int,
		default=700,
		help="要抽取的图片数量，0 表示输出全部。",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="随机种子。",
	)
	args = parser.parse_args()

	if not os.path.isdir(args.input_dir):
		raise FileNotFoundError(f"输入 YOLO 数据集目录不存在: {args.input_dir}")
	if args.number < 0:
		raise ValueError("number 必须为非负整数。")

	return args


def resolve_image_target(image_path: Path) -> Path:
	if image_path.is_symlink():
		target = Path(os.path.realpath(image_path))
		if target.exists():
			return target
	return image_path


def link_or_copy_image(src_image: Path, dst_image: Path) -> None:
	target = resolve_image_target(src_image)
	if dst_image.exists() or dst_image.is_symlink():
		dst_image.unlink()

	try:
		os.symlink(str(target), str(dst_image))
	except OSError:
		shutil.copy2(target, dst_image)


def copy_label(src_label: Path, dst_label: Path) -> None:
	if dst_label.exists() or dst_label.is_symlink():
		dst_label.unlink()

	if src_label.exists():
		shutil.copy2(src_label, dst_label)
	else:
		dst_label.write_text("", encoding="utf-8")


def collect_candidates(images_dir: Path, labels_dir: Path):
	candidates = []
	for image_path in sorted(images_dir.iterdir()):
		if image_path.is_dir():
			continue
		if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
			continue
		label_path = labels_dir / f"{image_path.stem}.txt"
		candidates.append((image_path, label_path))
	return candidates


def main():
	args = parse_args()
	random.seed(args.seed)

	input_split_dir = Path(args.input_dir) / args.type
	input_images_dir = input_split_dir / "images"
	input_labels_dir = input_split_dir / "labels"

	if not input_images_dir.is_dir():
		raise FileNotFoundError(f"输入图片目录不存在: {input_images_dir}")
	if not input_labels_dir.is_dir():
		raise FileNotFoundError(f"输入标签目录不存在: {input_labels_dir}")

	candidates = collect_candidates(input_images_dir, input_labels_dir)
	if not candidates:
		raise SystemExit("没有找到可抽样的图片。")

	total = len(candidates)
	if args.number == 0 or args.number >= total:
		selected = candidates
	else:
		selected = random.sample(candidates, args.number)

	output_split_dir = Path(args.output_dir) / args.type
	output_images_dir = output_split_dir / "images"
	output_labels_dir = output_split_dir / "labels"
	output_images_dir.mkdir(parents=True, exist_ok=True)
	output_labels_dir.mkdir(parents=True, exist_ok=True)

	for image_path, label_path in selected:
		dst_image = output_images_dir / image_path.name
		dst_label = output_labels_dir / label_path.name
		link_or_copy_image(image_path, dst_image)
		copy_label(label_path, dst_label)

	print(
		f"完成：从 {input_split_dir} 抽取 {len(selected)}/{total} 张图片到 {output_split_dir}"
	)


if __name__ == "__main__":
	main()
