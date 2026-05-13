#!/usr/bin/env python3
"""
infer_sahi.py

用法:
        python infer_sahi.py --weights yolov8s.pt --source /path/to/image.jpg
        python infer_sahi.py --weights yolov8s.pt --source /path/to/images_dir/
	python infer_sahi.py --weights yolov8s.pt --source /path/to/video.mp4

说明:
        - 使用 SAHI + Ultralytics YOLO 做画窗(切片)推理。
	- 支持单张图片、图片文件夹和视频文件。
        - 推理后的可视化结果会保存到 --save-dir 指定目录。
"""

from __future__ import annotations

import argparse
import threading
import sys
import time
from queue import Queue
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
VIDEO_FRAME_SENTINEL = object()
FAST_RENDER_COLORS = [
	(0, 255, 0),
	(0, 255, 255),
	(255, 0, 0),
	(255, 255, 0),
	(255, 0, 255),
	(0, 128, 255),
]


def is_image_file(path: Path) -> bool:
	return path.suffix.lower() in IMAGE_EXTS


def is_video_file(path: Path) -> bool:
	return path.suffix.lower() in VIDEO_EXTS


def get_image_files(folder: Path) -> list[Path]:
	return [p for p in sorted(folder.iterdir()) if p.is_file() and is_image_file(p)]


def mkdir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="使用 SAHI 对 YOLO 做画窗推理")
	parser.add_argument(
		"--weights",
		"-w",
		# default="runs/detect/runs/train/QZ_Avia_CST_Hybrid_sc-v8s/weights/best.pt", # sc
        default="runs/detect/runs/train/qz_avia_cst-yolov8s/weights/best.pt", # mc
		help="模型权重路径 (.pt)",
	)
	parser.add_argument(
		"--source",
		"-s",
		default="datasets/111无人机测试.MP4",
		help="图片文件或图片文件夹路径",
	)
	parser.add_argument("--device", default="cuda", help="设备: cpu 或 cuda:0")
	parser.add_argument("--imgsz", type=int, default=640, help="模型输入尺寸")
	parser.add_argument("--conf", type=float, default=0.2, help="置信度阈值")
	parser.add_argument("--slice-height", type=int, default=480, help="切片高度")
	parser.add_argument("--slice-width", type=int, default=640, help="切片宽度")
	parser.add_argument("--overlap-height-ratio", type=float, default=0.2, help="切片高度重叠比例")
	parser.add_argument("--overlap-width-ratio", type=float, default=0.2, help="切片宽度重叠比例")
	parser.add_argument("--batch-size", type=int, default=8, help="单次送入模型的切片数量，增大可提升吞吐")
	parser.add_argument("--prefetch-size", type=int, default=16, help="视频预取队列大小，配合线程模式使用")
	parser.add_argument("--postprocess-type", default="GREEDYNMM", help="切片后处理类型: NMS/NMM/GREEDYNMM")
	parser.add_argument("--postprocess-match-metric", default="IOS", help="后处理匹配度量: IOU/IOS")
	parser.add_argument("--postprocess-match-threshold", type=float, default=0.5, help="后处理匹配阈值")
	parser.add_argument("--postprocess-class-agnostic", action="store_true", help="后处理是否类别无关")
	parser.add_argument("--no-standard-pred", action="store_true", help="仅切片推理，不叠加整图预测")
	parser.add_argument(
		"--render-mode",
		choices=("fast", "sahi", "defer", "threaded"),
		default="threaded",
		help="结果渲染方式: fast 用 OpenCV 轻量绘制, sahi 用 SAHI 原生可视化, defer 先推理后统一绘制, threaded 先预取再推理",
	)
	parser.add_argument("--line-width", type=int, default=1, help="可视化框线宽")
	parser.add_argument("--font-size", type=float, default=0.3, help="可视化字体缩放")
	parser.add_argument("--hide-labels", action="store_true", help="隐藏类别标签")
	parser.add_argument("--hide-conf", action="store_true", help="隐藏置信度")
	parser.add_argument("--save-dir", default="runs/infer_sahi", help="结果保存目录")
	return parser.parse_args()


def render_predictions_fast(
	image,
	object_prediction_list,
	line_width: int,
	font_size: float,
	hide_labels: bool,
	hide_conf: bool,
):
	annotated = image.copy()
	thickness = max(1, line_width)
	font_scale = max(0.35, float(font_size))
	for prediction in object_prediction_list:
		bbox = prediction.bbox.to_xyxy()
		x1, y1, x2, y2 = map(int, bbox)
		category_id = int(getattr(prediction.category, "id", 0) or 0)
		color = FAST_RENDER_COLORS[category_id % len(FAST_RENDER_COLORS)]
		cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=thickness)
		if hide_labels:
			continue
		label = str(getattr(prediction.category, "name", category_id))
		if not hide_conf:
			label = f"{label} {prediction.score.value:.2f}"
		(text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
		text_y1 = max(0, y1 - text_h - baseline - 4)
		text_y2 = max(0, y1)
		cv2.rectangle(annotated, (x1, text_y1), (x1 + text_w + 4, text_y2), color, thickness=-1)
		cv2.putText(
			annotated,
			label,
			(x1 + 2, max(12, y1 - 3)),
			cv2.FONT_HERSHEY_SIMPLEX,
			font_scale,
			(255, 255, 255),
			thickness=1,
			lineType=cv2.LINE_AA,
		)
	return annotated


def render_predictions_with_mode(
	image,
	object_prediction_list,
	args: argparse.Namespace,
):
	from sahi.utils.cv import visualize_object_predictions

	if args.render_mode in ("fast", "threaded"):
		return render_predictions_fast(
			image,
			object_prediction_list,
			line_width=args.line_width,
			font_size=args.font_size,
			hide_labels=args.hide_labels,
			hide_conf=args.hide_conf,
		)

	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	vis = visualize_object_predictions(
		image=rgb,
		object_prediction_list=object_prediction_list,
		rect_th=max(1, args.line_width),
		text_size=args.font_size,
		hide_labels=args.hide_labels,
		hide_conf=args.hide_conf,
	)
	return cv2.cvtColor(vis["image"], cv2.COLOR_RGB2BGR)


def _video_frame_reader(video_path: Path, frame_queue: Queue, prefetch_size: int) -> None:
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		frame_queue.put(VIDEO_FRAME_SENTINEL)
		return

	frame_idx = 0
	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				break
			frame_queue.put((frame_idx, frame))
			frame_idx += 1
	finally:
		cap.release()
		frame_queue.put(VIDEO_FRAME_SENTINEL)


def infer_video_threaded(detection_model, video_path: Path, save_path: Path, args: argparse.Namespace) -> None:
	from sahi.predict import get_sliced_prediction

	probe = cv2.VideoCapture(str(video_path))
	if not probe.isOpened():
		print(f"无法打开视频: {video_path}")
		return
	fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
	width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
	probe.release()

	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	mkdir(save_path.parent)
	out = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))

	frame_queue: Queue = Queue(maxsize=max(1, args.prefetch_size))
	reader = threading.Thread(target=_video_frame_reader, args=(video_path, frame_queue, args.prefetch_size), daemon=True)
	reader.start()

	processed_frames = 0
	total_detections = 0
	predictions_by_frame: list[list] = []
	inference_start = time.perf_counter()
	print(f"开始线程化视频推理: {video_path} -> {save_path}")

	while True:
		item = frame_queue.get()
		if item is VIDEO_FRAME_SENTINEL:
			break

		_, frame = item
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		result = get_sliced_prediction(
			rgb,
			detection_model=detection_model,
			slice_height=args.slice_height,
			slice_width=args.slice_width,
			overlap_height_ratio=args.overlap_height_ratio,
			overlap_width_ratio=args.overlap_width_ratio,
			perform_standard_pred=not args.no_standard_pred,
			postprocess_type=args.postprocess_type,
			postprocess_match_metric=args.postprocess_match_metric,
			postprocess_match_threshold=args.postprocess_match_threshold,
			postprocess_class_agnostic=args.postprocess_class_agnostic,
			batch_size=args.batch_size,
			verbose=0,
		)
		predictions_by_frame.append(result.object_prediction_list)
		total_detections += len(result.object_prediction_list)
		processed_frames += 1
		if processed_frames % 20 == 0:
			print(f"已处理 {processed_frames} 帧")

	reader.join()
	inference_elapsed = time.perf_counter() - inference_start
	inference_fps = processed_frames / inference_elapsed if inference_elapsed > 0 else 0.0

	render_start = time.perf_counter()
	replay = cv2.VideoCapture(str(video_path))
	if not replay.isOpened():
		print(f"无法重新打开视频用于绘制: {video_path}")
		out.release()
		return

	frame_cursor = 0
	while frame_cursor < len(predictions_by_frame):
		ret, frame = replay.read()
		if not ret:
			break
		annotated_bgr = render_predictions_fast(
			frame,
			predictions_by_frame[frame_cursor],
			line_width=args.line_width,
			font_size=args.font_size,
			hide_labels=args.hide_labels,
			hide_conf=args.hide_conf,
		)
		out.write(annotated_bgr)
		frame_cursor += 1

	replay.release()
	out.release()
	render_elapsed = time.perf_counter() - render_start
	print(
		f"视频推理完成: frames={processed_frames}, detections={total_detections}, "
		f"infer_time={inference_elapsed:.2f}s, infer_fps={inference_fps:.2f}, "
		f"render_time={render_elapsed:.2f}s"
	)


def build_sahi_model(weights: str, device: str, conf: float, imgsz: int):
	from sahi import AutoDetectionModel

	return AutoDetectionModel.from_pretrained(
		model_type="ultralytics",
		model_path=weights,
		device=device,
		confidence_threshold=conf,
		image_size=imgsz,
	)


def infer_one_image(detection_model, image_path: Path, save_path: Path, args: argparse.Namespace) -> None:
	from sahi.predict import get_sliced_prediction

	image = cv2.imread(str(image_path))
	if image is None:
		print(f"无法读取图片: {image_path}")
		return

	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	result = get_sliced_prediction(
		rgb,
		detection_model=detection_model,
		slice_height=args.slice_height,
		slice_width=args.slice_width,
		overlap_height_ratio=args.overlap_height_ratio,
		overlap_width_ratio=args.overlap_width_ratio,
		perform_standard_pred=not args.no_standard_pred,
		postprocess_type=args.postprocess_type,
		postprocess_match_metric=args.postprocess_match_metric,
		postprocess_match_threshold=args.postprocess_match_threshold,
		postprocess_class_agnostic=args.postprocess_class_agnostic,
		batch_size=args.batch_size,
		verbose=0,
	)

	annotated_bgr = render_predictions_with_mode(image, result.object_prediction_list, args)

	mkdir(save_path.parent)
	cv2.imwrite(str(save_path), annotated_bgr)
	print(f"推理图片: {image_path} -> {save_path} | detections={len(result.object_prediction_list)}")


def infer_video(detection_model, video_path: Path, save_path: Path, args: argparse.Namespace) -> None:
	from sahi.predict import get_sliced_prediction

	if args.render_mode == "threaded":
		infer_video_threaded(detection_model, video_path, save_path, args)
		return

	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		print(f"无法打开视频: {video_path}")
		return

	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	mkdir(save_path.parent)
	out = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))

	frame_idx = 0
	total_detections = 0
	predictions_by_frame: list[list] = []
	inference_start = time.perf_counter()
	print(f"开始视频推理: {video_path} -> {save_path}")
	while True:
		ret, frame = cap.read()
		if not ret:
			break

		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		result = get_sliced_prediction(
			rgb,
			detection_model=detection_model,
			slice_height=args.slice_height,
			slice_width=args.slice_width,
			overlap_height_ratio=args.overlap_height_ratio,
			overlap_width_ratio=args.overlap_width_ratio,
			perform_standard_pred=not args.no_standard_pred,
			postprocess_type=args.postprocess_type,
			postprocess_match_metric=args.postprocess_match_metric,
			postprocess_match_threshold=args.postprocess_match_threshold,
			postprocess_class_agnostic=args.postprocess_class_agnostic,
			batch_size=args.batch_size,
			verbose=0,
		)

		predictions_by_frame.append(result.object_prediction_list)
		if args.render_mode == "defer":
			dets = len(result.object_prediction_list)
			total_detections += dets
			frame_idx += 1
			if frame_idx % 20 == 0:
				print(f"已处理 {frame_idx} 帧")
			continue

		annotated_bgr = render_predictions_with_mode(frame, result.object_prediction_list, args)
		out.write(annotated_bgr)

		dets = len(result.object_prediction_list)
		total_detections += dets
		frame_idx += 1
		if frame_idx % 20 == 0:
			print(f"已处理 {frame_idx} 帧")

	cap.release()
	inference_elapsed = time.perf_counter() - inference_start
	inference_fps = frame_idx / inference_elapsed if inference_elapsed > 0 else 0.0

	if args.render_mode == "defer":
		render_start = time.perf_counter()
		replay = cv2.VideoCapture(str(video_path))
		if not replay.isOpened():
			print(f"无法重新打开视频用于绘制: {video_path}")
			out.release()
			return
		frame_cursor = 0
		while True:
			ret, frame = replay.read()
			if not ret:
				break
			annotated_bgr = render_predictions_with_mode(frame, predictions_by_frame[frame_cursor], args)
			out.write(annotated_bgr)
			frame_cursor += 1
		replay.release()
		render_elapsed = time.perf_counter() - render_start
		out.release()
		print(
			f"视频推理完成: frames={frame_idx}, detections={total_detections}, "
			f"infer_time={inference_elapsed:.2f}s, infer_fps={inference_fps:.2f}, "
			f"render_time={render_elapsed:.2f}s"
		)
		return

	out.release()

	print(
		f"视频推理完成: frames={frame_idx}, detections={total_detections}, "
		f"infer_time={inference_elapsed:.2f}s, infer_fps={inference_fps:.2f}"
	)


def main() -> None:
	args = parse_args()

	try:
		# 优先使用已安装的 sahi；若未安装则退回仓库内子模块
		from sahi.predict import get_sliced_prediction  # noqa: F401
	except Exception:
		repo_sahi = Path(__file__).resolve().parent / "sahi-git"
		if repo_sahi.is_dir() and str(repo_sahi) not in sys.path:
			sys.path.insert(0, str(repo_sahi))

	try:
		detection_model = build_sahi_model(args.weights, args.device, args.conf, args.imgsz)
	except Exception as exc:
		print("加载 SAHI/YOLO 模型失败，请先安装依赖: pip install sahi ultralytics")
		raise exc

	source = Path(args.source)
	save_dir = Path(args.save_dir)

	if source.is_file() and is_image_file(source):
		out_path = save_dir / source.name
		infer_one_image(detection_model, source, out_path, args)
		print(f"已保存: {out_path}")
		return

	if source.is_file() and is_video_file(source):
		out_path = save_dir / f"{source.stem}_infer.mp4"
		infer_video(detection_model, source, out_path, args)
		print(f"已保存: {out_path}")
		return

	if source.is_dir():
		image_files = get_image_files(source)
		if not image_files:
			print(f"目录中没有图片: {source}")
			return
		out_dir = save_dir / "images"
		mkdir(out_dir)
		for image_path in image_files:
			infer_one_image(detection_model, image_path, out_dir / image_path.name, args)
		print(f"已保存目录: {out_dir}")
		return

	print(f"未识别的 source 类型: {source}")


if __name__ == "__main__":
	main()
