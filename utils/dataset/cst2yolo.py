import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Convert CST format to YOLO format.")

    parser.add_argument(
        "--cst_dir",
        type=str,
        default="datasets/CST_AntiUAV/CST-AntiUAV",
        help="CST 数据集的路径，包含 train、val、test 文件夹。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/CST_AntiUAV/yolo",
        help="输出 YOLO 格式数据集的路径。",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="要转换的数据集类型（默认: train）。",
    )
    parser.add_argument(
        "--number",
        type=int,
        default=600,
        help="要转换的图像数量（默认: 0，表示转换所有图像）。",
    )
    parser.add_argument(
        "--ratio",
        type=int,
        default=100,
        help="转换图像的比例（默认: 100，表示转换所有图像）。",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="global",
        choices=["global", "seqX/Y"],
        help="图像选择方式，'global' 表示全局随机选择，适用于非时序的数据处理；'seq' 表示每个视频随机选取连续的 X 帧，抽 Y 份，适用于需要连续帧进行时序分析的（默认: random）。",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="随机种子，用于随机选择图像（默认: 42）。"
    )
    parser.add_argument(
        "--id", type=int, default=2, help="CST 是单类数据集，其类别的 ID 默认为 0。"
    )

    args = parser.parse_args()

    # 验证输入路径和输出路径
    if not os.path.exists(args.cst_dir):
        raise FileNotFoundError(f"CST 数据集路径不存在: {args.cst_dir}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 处理 number 和 ratio 参数
    if args.number < 0:
        raise ValueError("number 参数必须为非负整数。")
    if not (0 < args.ratio <= 100):
        raise ValueError("ratio 参数必须在 1 到 100 之间。")

    if args.number == 0:
        args.ratio = 100  # 如果 number 为 0，则转换所有图像
    if args.ratio == 100:
        args.number = 0  # 如果 ratio 为 100，则转换所有图像
    if args.number > 0 and args.ratio < 100:
        raise ValueError("不可同时设置 number 和 ratio 参数。请只设置其中一个。")

    return args


if __name__ == "__main__":
    import json
    import random
    from pathlib import Path

    args = parse_args()

    random.seed(args.seed)

    src_type_dir = os.path.join(args.cst_dir, args.type)
    if not os.path.isdir(src_type_dir):
        raise FileNotFoundError(f"CST 子目录不存在: {src_type_dir}")

    out_images_dir = os.path.join(args.output_dir, args.type, "images")
    out_labels_dir = os.path.join(args.output_dir, args.type, "labels")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    def try_load_json(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # fallback: try to read as lines of json
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                if len(lines) == 1:
                    return json.loads(lines[0])
                return [json.loads(l) for l in lines]
            except Exception:
                return None

    def get_image_size(path):
        """Get image size using PIL/cv2, matching visualize_dataset.py approach."""
        try:
            from PIL import Image
            with Image.open(path) as im:
                return im.width, im.height
        except Exception:
            pass
        
        try:
            import cv2
            img = cv2.imread(path)
            if img is None:
                raise RuntimeError("无法读取图像")
            h, w = img.shape[:2]
            return w, h
        except Exception as e:
            raise RuntimeError(f"无法读取图像尺寸: {path} -> {e}")

    # collect candidates: (img_path, bbox)
    candidates = []
    scene_size_cache = {}

    for scene in sorted(os.listdir(src_type_dir)):
        scene_dir = os.path.join(src_type_dir, scene)
        if not os.path.isdir(scene_dir):
            continue

        # image files
        imgs = sorted(
            [
                p
                for p in os.listdir(scene_dir)
                if p.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        if not imgs:
            continue

        exist_path = os.path.join(scene_dir, "exist.txt")
        gt_path = os.path.join(scene_dir, "gt.txt")

        exist_data = try_load_json(exist_path) if os.path.exists(exist_path) else None
        gt_data = try_load_json(gt_path) if os.path.exists(gt_path) else None

        # If JSON load failed, try plain text formats
        if os.path.exists(exist_path) and exist_data is None:
            try:
                with open(exist_path, "r", encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                exist_data = [int(ln) for ln in lines]
            except Exception:
                exist_data = None

        if os.path.exists(gt_path) and gt_data is None:
            try:
                with open(gt_path, "r", encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                parsed = []
                for ln in lines:
                    # support comma or space separated values
                    if "," in ln:
                        parts = [p.strip() for p in ln.split(",") if p.strip()]
                    else:
                        parts = [p.strip() for p in ln.split() if p.strip()]
                    vals = [float(p) for p in parts]
                    parsed.append(vals)
                gt_data = parsed
            except Exception:
                gt_data = None

        # normalize exist and gt arrays
        exist_list = None
        gt_list = None
        if isinstance(exist_data, dict) and "exist" in exist_data:
            exist_list = exist_data["exist"]
        elif isinstance(exist_data, list):
            exist_list = exist_data

        if isinstance(gt_data, dict) and "gt" in gt_data:
            gt_list = gt_data["gt"]
        elif isinstance(gt_data, list):
            gt_list = gt_data

        # If gt_list is a dict mapping frame->bbox, try to extract in order
        if isinstance(gt_list, dict):
            # attempt to order by filename index
            ordered = []
            for img_name in imgs:
                key = img_name
                if key in gt_list:
                    ordered.append(gt_list[key])
                else:
                    # try numeric index
                    idx = os.path.splitext(img_name)[0]
                    if idx in gt_list:
                        ordered.append(gt_list[idx])
                    else:
                        ordered.append([0, 0, 0, 0])
            gt_list = ordered

        # If lengths mismatch, clamp to image count
        num_imgs = len(imgs)
        if gt_list and len(gt_list) != num_imgs:
            # allow if gt has more (maybe 1-based indexing)
            if len(gt_list) > num_imgs:
                gt_list = gt_list[:num_imgs]
            else:
                # pad
                gt_list = gt_list + [[0, 0, 0, 0]] * (num_imgs - len(gt_list))

        if exist_list and len(exist_list) != num_imgs:
            if len(exist_list) > num_imgs:
                exist_list = exist_list[:num_imgs]
            else:
                exist_list = exist_list + [0] * (num_imgs - len(exist_list))

        for i, img_name in enumerate(imgs):
            img_path = os.path.join(scene_dir, img_name)
            exists = None
            bbox = None
            if exist_list is not None:
                try:
                    exists = int(exist_list[i])
                except Exception:
                    exists = 1 if exist_list[i] else 0
            if gt_list is not None:
                try:
                    bbox = gt_list[i]
                except Exception:
                    bbox = None

            # if exist_list provided and is 0 -> skip
            if exists == 0:
                continue

            # if no bbox info, skip
            if not bbox:
                continue

            # bbox expected [x_ltc, y_ltc, w, h]
            if isinstance(bbox, dict):
                # try keys
                bbox = [bbox.get(k, 0) for k in ("x", "y", "w", "h")]

            if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
                continue

            candidates.append((img_path, bbox))

    if not candidates:
        print("没有找到任何候选样本。请检查 CST 数据集结构。")
        raise SystemExit(1)

    print(f"找到 {len(candidates)} 个候选样本。")

    # apply ratio/number selection
    total = len(candidates)
    if args.number > 0:
        k = min(args.number, total)
        selected = random.sample(candidates, k)
    elif args.ratio < 100:
        k = max(1, int(total * args.ratio / 100.0))
        selected = random.sample(candidates, k)
    else:
        selected = candidates

    # write labels and symlink images with progress
    import time, sys

    total_selected = len(selected)
    start_time = time.time()
    for idx, (img_path, bbox) in enumerate(selected, start=1):
        try:
            scene_key = os.path.dirname(img_path)
            if scene_key not in scene_size_cache:
                scene_imgs = sorted(
                    [
                        p
                        for p in os.listdir(scene_key)
                        if p.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                )
                if not scene_imgs:
                    raise RuntimeError(f"场景中没有图片: {scene_key}")
                scene_size_cache[scene_key] = get_image_size(os.path.join(scene_key, scene_imgs[0]))
            w, h = scene_size_cache[scene_key]
        except Exception as e:
            print(f"跳过图像 (无法读取尺寸): {img_path} -> {e}")
            continue

        # Extract bbox as [x, y, w, h] where x,y are top-left (same as visualize_dataset.py)
        x, y, bw, bh = [float(v) for v in bbox[:4]]
        
        # Compute center point and normalize (YOLO format)
        x_center = x + bw / 2.0
        y_center = y + bh / 2.0
        
        x_center_n = x_center / w
        y_center_n = y_center / h
        bw_n = bw / w
        bh_n = bh / h
        
        # Clamp to valid range [0,1]
        x_center_n = max(0.0, min(1.0, x_center_n))
        y_center_n = max(0.0, min(1.0, y_center_n))
        bw_n = max(0.0, min(1.0, bw_n))
        bh_n = max(0.0, min(1.0, bh_n))

        # dest paths — prefix with the source scene to avoid filename collisions across scenes
        img_basename = os.path.basename(img_path)
        prefixed_name = f"{os.path.basename(os.path.dirname(img_path))}_{img_basename}"
        label_name = os.path.splitext(prefixed_name)[0] + ".txt"
        out_img = os.path.join(out_images_dir, prefixed_name)
        out_lbl = os.path.join(out_labels_dir, label_name)

        # symlink image
        try:
            if not os.path.exists(out_img):
                os.symlink(os.path.abspath(img_path), out_img)
        except Exception:
            # fallback to copy if symlink not allowed
            try:
                import shutil

                if not os.path.exists(out_img):
                    shutil.copy2(img_path, out_img)
            except Exception:
                print(f"无法链接或复制图像: {img_path}")
                continue

        # write label
        with open(out_lbl, "w", encoding="utf-8") as f:
            line = f"{args.id} {x_center_n:.6f} {y_center_n:.6f} {bw_n:.6f} {bh_n:.6f}\n"
            f.write(line)

        # progress update every 50 or on last
        if idx % 50 == 0 or idx == total_selected:
            elapsed = time.time() - start_time
            avg = elapsed / idx
            remaining = avg * (total_selected - idx)
            pct = idx / total_selected * 100
            sys.stdout.write(
                f"\r已处理 {idx}/{total_selected} ({pct:.1f}%)，耗时 {elapsed:.1f}s，预计剩余 {remaining:.1f}s"
            )
            sys.stdout.flush()

    print()

    print(
        f"完成：已生成 {len(selected)} 个样本，输出目录: {os.path.join(args.output_dir, args.type)}"
    )
