# CST to YOLO Format Conversion

脚本：`cst2yolo.py`

该脚本将 CST 格式的注释转换为 YOLO 格式。它读取 CST 注释文件，提取目标边界框信息，并将其转换为 YOLO 格式的注释文件。

参数：

| 参数         | 默认值   | 描述                                             |
| ------------ | -------- | ------------------------------------------------ |
| `cst_dir`    | `""`     | CST 数据集的路径，包含 train、val、test 文件夹。 |
| `output_dir` | `""`     | 输出 YOLO 格式数据集的路径。                     |
| `type`       | `train`  | 要转换的数据集类型                               |
| `number`     | `0`      | 要转换的图像数量，0 表示转换所有图像。           |
| `ratio`      | `100`    | 转换图像的比例，100 表示转换所有图像。           |
| `selection`  | `global` | 图像选择方式                                     |
| `seed`       | `42`     | 随机种子，用于随机选择图像。                     |
| `id`         | `0`      | CST 是单类数据集，其类别的 ID 默认为 0。         |


CST 格式：

```
<cst_dir>/
    train/
        scene1/ (每个场景一个文件夹，场景内的图像是连续帧)
            000001.jpg
            ...
            exist.txt (CST annotation files)
            gt.txt (CST annotation files)
            IR_label.json (CST annotation files)
        scene2/
            ...
        ...
    val/
        ...
    test/
        ...
```

其中 exist.txt 包含每个图像是否存在目标的信息，即JSON 文件中的 exist 数组
gt.txt 包含每个图像的目标边界框信息，浮点数，转换为整数即为[x_ltc, y_ltc, w, h]
即JSON 文件中的 gt 数组，exist为0则 gt 数组为 [0, 0, 0, 0]

YOLO 文件夹布局：

```
<output_dir>/
    train/
        images/
            image1.jpg (slinks to original images)
            ...
        labels/
            image1.txt (YOLO annotation files)
            ...
    val/
        ...
    test/
        ...
```


