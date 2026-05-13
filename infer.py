"""
infer.py

用法:
        python infer.py --weights yolov8s.pt --source /path/to/image.jpg
        python infer.py --weights yolov8s.pt --source /path/to/images_dir/
        python infer.py --weights yolov8s.pt --source /path/to/video.mp4

说明:
        - 脚本优先使用 `ultralytics` 的 `YOLO` API 加载并推理 `.pt` 模型。
        - 支持单张图片、图片文件夹和视频文件。推理后的可视化结果会保存到 `--save-dir` 指定目录。
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np


def is_image_file(p: Path):
    return p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def get_image_files(folder: Path):
    return [p for p in sorted(folder.iterdir()) if p.is_file() and is_image_file(p)]


def mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run_on_image(
    model,
    src_path: Path,
    save_path: Path,
    imgsz: int,
    conf: float,
    line_width: int,
    font_size: int,
    draw_scale: float,
):
    img = cv2.imread(str(src_path))
    if img is None:
        print(f"无法读取图片: {src_path}")
        return
    # ultralytics YOLO accepts numpy image directly
    results = model(img, imgsz=imgsz, conf=conf)
    # results may be a list-like; handle first
    res = results[0]
    try:
        # compute effective drawing sizes using draw_scale to allow smaller rendering
        h, w = img.shape[:2]
        scale = float(draw_scale) if draw_scale is not None else 1.0
        # effective sizes (may be zero to indicate hide)
        eff_line = int(line_width * scale) if line_width is not None else 1
        eff_font = int(font_size * scale) if font_size is not None else 0
        # if both zero -> no drawing
        if eff_line <= 0 and eff_font <= 0:
            annotated = img
        # only labels (no boxes)
        elif eff_line <= 0 and eff_font > 0:
            annotated = img.copy()
            boxes = getattr(res, "boxes", None)
            names = getattr(model, "names", None)
            if boxes is not None and len(boxes) > 0:
                xyxy = None
                try:
                    xyxy = boxes.xyxy.cpu().numpy()
                except Exception:
                    try:
                        xyxy = np.array(boxes.xyxy)
                    except Exception:
                        xyxy = None
                cls_ids = None
                confs = None
                try:
                    cls_ids = boxes.cls.cpu().numpy()
                except Exception:
                    try:
                        cls_ids = np.array(boxes.cls)
                    except Exception:
                        cls_ids = None
                try:
                    confs = boxes.conf.cpu().numpy()
                except Exception:
                    try:
                        confs = np.array(boxes.conf)
                    except Exception:
                        confs = None
                font_scale = max(0.3, eff_font * 0.08)
                for i, b in enumerate(xyxy if xyxy is not None else []):
                    x1, y1 = int(b[0]), int(b[1])
                    cls_name = ""
                    if cls_ids is not None and names is not None:
                        idx = int(cls_ids[i]) if i < len(cls_ids) else None
                        if idx is not None and idx in names:
                            cls_name = names[idx]
                    conf = confs[i] if confs is not None and i < len(confs) else None
                    label = cls_name if cls_name else ""
                    if conf is not None:
                        label = f"{label} {conf:.2f}".strip()
                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )
            # else use built-in plot
        else:
            # ensure at least 1 for visibility of boxes when requested
            eff_line = max(1, eff_line)
            annotated = res.plot(line_width=eff_line, font_size=eff_font)
    except Exception:
        # fallback: use original image
        annotated = img
    cv2.imwrite(str(save_path), annotated)


def run_on_folder(
    model,
    src_folder: Path,
    save_dir: Path,
    imgsz: int,
    conf: float,
    line_width: int,
    font_size: int,
    draw_scale: float,
):
    imgs = get_image_files(src_folder)
    if not imgs:
        print(f"目录中没有图片: {src_folder}")
        return
    mkdir(save_dir)
    for p in imgs:
        outp = save_dir / p.name
        print(f"推理图片: {p} -> {outp}")
        run_on_image(model, p, outp, imgsz, conf, line_width, font_size, draw_scale)


def run_on_video(
    model,
    src_video: Path,
    save_path: Path,
    imgsz: int,
    conf: float,
    device: str,
    line_width: int,
    font_size: int,
    draw_scale: float,
):
    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        print(f"无法打开视频: {src_video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    mkdir(save_path.parent)
    out = cv2.VideoWriter(str(save_path), fourcc, fps, (w, h))
    frame_idx = 0
    print(f"开始视频推理, 输出: {save_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, imgsz=imgsz, conf=conf, device=device)
        res = results[0]
        try:
            # apply draw_scale per-frame (keeps sizes reasonable for different resolutions)
            scale = float(draw_scale) if draw_scale is not None else 1.0
            eff_line = max(1, int(line_width * scale)) if line_width is not None else 1
            eff_font = max(1, int(font_size * scale)) if font_size is not None else 1
            annotated = res.plot(line_width=eff_line, font_size=eff_font)
        except Exception:
            annotated = frame
        out.write(annotated)
        frame_idx += 1
    cap.release()
    out.release()
    print(f"视频推理完成, 帧数: {frame_idx}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        "-w",
        default="runs/detect/runs/train/QZ_Avia_CST_Hybrid_sc-v8s/weights/best.pt",
        help="模型权重路径 (.pt)",
    )
    parser.add_argument(
        "--source",
        "-s",
        default="datasets/avia/L20260105093255107.MP4",
        help="图片/文件夹/视频路径",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--conf", type=float, default=0.2, help="置信度阈值")
    parser.add_argument("--iou_thres", type=float, default=0.1, help="NMS IoU 阈值")
    parser.add_argument("--device", default="cuda", help="设备: cpu 或 cuda:0")
    parser.add_argument("--line-width", type=int, default=1, help="目标框线宽")
    parser.add_argument("--font-size", type=int, default=10, help="字体大小")
    parser.add_argument(
        "--draw-scale",
        type=float,
        default=0.5,
        help="绘制缩放因子 (<1: 更细/更小, 1: 原始, >1: 放大)",
    )
    parser.add_argument("--save-dir", default="runs/infer", help="保存结果目录")
    args = parser.parse_args()

    weights = args.weights
    src = Path(args.source)
    savedir = Path(args.save_dir)

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("需要安装 ultralytics 才能使用此脚本: pip install ultralytics")
        raise e

    print(f"加载模型: {weights}")
    model = YOLO(weights)

    if src.is_file() and is_image_file(src):
        out_dir = savedir
        mkdir(out_dir)
        out_path = out_dir / src.name
        run_on_image(
            model,
            src,
            out_path,
            args.imgsz,
            args.conf,
            args.line_width,
            args.font_size,
            args.draw_scale,
        )
        print(f"已保存: {out_path}")
    elif src.is_dir():
        out_dir = savedir / "images"
        run_on_folder(
            model,
            src,
            out_dir,
            args.imgsz,
            args.conf,
            args.line_width,
            args.font_size,
            args.draw_scale,
        )
        print(f"已保存目录: {out_dir}")
    elif src.is_file():
        # assume video
        out_path = savedir / (src.stem + "_infer.mp4")
        run_on_video(
            model,
            src,
            out_path,
            args.imgsz,
            args.conf,
            args.device,
            args.line_width,
            args.font_size,
            args.draw_scale,
        )
    else:
        print(f"未识别的 source 类型: {src}")


if __name__ == "__main__":
    main()
