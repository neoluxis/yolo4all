import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="使用筛选后的 YOLO 数据集训练 YOLOv8s。")
    parser.add_argument(
        "--data_yaml",
        type=str,
        default="datasets/YOLO-20260411-QZ_Avia_CST_Hybrid/qz_Avia-coco.yaml",
        help="YOLO 数据集 YAML 路径。",
    )
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="yolov8s.yaml",
        help="本地 YOLOv8s 配置文件。",
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default="yolov8s.pt",
        help="预训练权重路径，默认为官方 yolov8s.pt。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="训练轮数。",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="输入图像尺寸。",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=64,
        help="训练 batch size。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="训练设备，例如 0、0,1 或 cpu。留空则使用 Ultralytics 默认设置。",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="训练输出根目录。",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="QZ_Avia_CST_Hybrid-v8s",
        help="训练实验名称。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="数据加载 workers 数。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_cfg = Path(args.model_cfg)
    data_yaml = Path(args.data_yaml)

    if not model_cfg.is_file():
        raise FileNotFoundError(f"YOLOv8s 配置文件不存在: {model_cfg}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"数据集 YAML 不存在: {data_yaml}")

    from ultralytics import YOLO # 延迟导入以避免不必要的依赖问题

    model = YOLO(str(model_cfg))
    if args.pretrained_weights:
        if  args.pretrained_weights.startswith("yolo") and args.pretrained_weights.endswith(".pt"):
            # 官方预训练权重，直接使用名称加载
            model = YOLO(args.pretrained_weights)
        else:
            # 本地权重文件，检查存在后加载
            pretrained_path = Path(args.pretrained_weights)
            if not pretrained_path.is_file():
                raise FileNotFoundError(f"预训练权重文件不存在: {pretrained_path}")
            model.load(str(pretrained_path))
        
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "workers": args.workers,
        "seed": args.seed,
    }
    if args.device:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
