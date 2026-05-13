#!/usr/bin/env python3
"""
简单的 YOLO 导出脚本，基于 ultralytics 的 `YOLO.export` 接口。

用法示例:
  python export.py --weights yolov8s.pt --formats onnx,torchscript --imgsz 640 --output-dir exports

依赖:
  pip install ultralytics
"""

import argparse
import os
import sys
from typing import List


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export YOLO model to multiple formats"
    )
    parser.add_argument(
        "--weights",
        "-w",
        default="runs/detect/runs/train/QZ_Avia_CST_Hybrid-v8s/weights/best.pt",
        help="Path to model weights (.pt)",
    )
    parser.add_argument(
        "--formats",
        "-f",
        default="rknn,onnx",
        help="Comma-separated export formats, e.g. onnx,torchscript,tflite,openvino,coreml",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Image size for export (square)"
    )
    parser.add_argument(
        "--opset", type=int, default=11, help="ONNX opset version (if applicable)"
    )
    parser.add_argument(
        "--device", default=None, help="Device to run export on (e.g., cpu or 0)"
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Try to simplify ONNX model after export",
    )
    parser.add_argument(
        "--output-dir", "-o", default="exports", help="Output directory"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    return parser.parse_args()


def ensure_output_dir(path: str):
    return
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    formats: List[str] = [s.strip() for s in args.formats.split(",") if s.strip()]
    if "rknn" in formats:
        import os
        os.environ["ULTRALYTICS_EXPORT_FORMAT"] = "rknn"

    try:
        from ultralytics import YOLO
    except Exception as e:
        print(
            "错误: 无法导入 ultralytics。请先安装：pip install ultralytics",
            file=sys.stderr,
        )
        print("导入异常：", e, file=sys.stderr)
        sys.exit(2)

    if not os.path.exists(args.weights):
        print(f"错误: 权重文件不存在: {args.weights}", file=sys.stderr)
        sys.exit(2)

    ensure_output_dir(args.output_dir)

    print(f"加载模型 {args.weights} ...")
    model = YOLO(args.weights)

    for fmt in formats:
        fmt_lower = fmt.lower()
        print(f"导出格式: {fmt_lower} (imgsz={args.imgsz})")
        try:
            # ultralytics 的 export 接口在不同版本上可能接受不同参数。
            # 这里尽量传入通用参数，保留对 opset、device、simplify 的支持。
            export_kwargs = dict(format=fmt_lower, imgsz=args.imgsz, device=args.device)
            if args.opset is not None:
                export_kwargs["opset"] = args.opset
            if args.simplify:
                export_kwargs["simplify"] = True

            # 指定输出目录（新版 ultralytics 支持 save_dir）
            try:
                model.export(**export_kwargs, save_dir=args.output_dir)
            except TypeError:
                # 退回到不带 save_dir 的调用（兼容旧版）
                model.export(**export_kwargs)

            print(f"完成: {fmt_lower} -> {args.output_dir}")
        except Exception as e:
            print(f"导出 {fmt_lower} 失败: {e}", file=sys.stderr)

    print("全部导出尝试完成。请检查输出目录是否包含期望的文件。")


if __name__ == "__main__":
    main()
