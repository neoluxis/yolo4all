"""
invert_images.py

用法:
	python invert_images.py --source ./images/
	python invert_images.py --source ./images/ --output ./images_inverted/
	python invert_images.py --source ./images/ --overwrite

说明:
	- 使用 OpenCV 将文件夹中所有图片反色
	- 支持 jpg, jpeg, png, bmp, tif, tiff, webp 格式
	- 默认输出到 {source_folder}_inverted 目录
	- 可选: 直接覆盖原图片 (--overwrite)
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np


def is_image_file(p: Path):
	return p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


def get_image_files(folder: Path):
	return [p for p in sorted(folder.iterdir()) if p.is_file() and is_image_file(p)]


def invert_image(img):
	"""Invert image colors: output = 255 - input"""
	return cv2.bitwise_not(img)


def invert_folder(src_folder: Path, dst_folder: Path, overwrite: bool = False):
	"""Process all images in src_folder and save inverted versions to dst_folder"""
	imgs = get_image_files(src_folder)
	if not imgs:
		print(f"目录中没有图片: {src_folder}")
		return
	
	# Create output directory if it doesn't exist
	if not overwrite:
		dst_folder.mkdir(parents=True, exist_ok=True)
	
	count = 0
	for src_path in imgs:
		# Read image
		img = cv2.imread(str(src_path))
		if img is None:
			print(f"无法读取图片: {src_path}")
			continue
		
		# Invert colors
		inverted = invert_image(img)
		
		# Determine output path
		if overwrite:
			dst_path = src_path
		else:
			dst_path = dst_folder / src_path.name
		
		# Save inverted image
		cv2.imwrite(str(dst_path), inverted)
		print(f"已处理: {src_path.name} -> {dst_path}")
		count += 1
	
	print(f"\n完成! 共处理 {count} 张图片")


def main():
	parser = argparse.ArgumentParser(description="使用 OpenCV 反色图片文件夹")
	parser.add_argument('--source', '-s', default="datasets/CST_AntiUAV/yolo_sampled/train/images", help='源图片文件夹路径')
	parser.add_argument('--output', '-o', help='输出文件夹 (默认: {source}_inverted)')
	parser.add_argument('--overwrite', action='store_true', help='直接覆盖原图片')
	args = parser.parse_args()
	
	src_folder = Path(args.source)
	if not src_folder.is_dir():
		print(f"错误: {src_folder} 不是一个有效的文件夹")
		return
	
	if args.overwrite:
		dst_folder = src_folder
	else:
		dst_folder = Path(args.output) if args.output else src_folder.parent / (src_folder.name + '_inverted')
	
	print(f"源文件夹: {src_folder}")
	print(f"输出文件夹: {dst_folder}")
	print(f"覆盖原图片: {args.overwrite}")
	print()
	
	invert_folder(src_folder, dst_folder, args.overwrite)


if __name__ == '__main__':
	main()
