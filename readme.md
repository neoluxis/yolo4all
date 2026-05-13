# YOLO4All: A Unified, Reusable Framework for YOLO

## Dataset Naming

数据集存放在 `datasets/`，可以使用其他路径的软链接。

数据集命名规范为：
1. 使用有意义的名字
2. `<name>[picnum]_<note>`，其中 `<name>` 是数据集的名字，`[picnum]` 是可选的数据集的图片数量，`<note>` 是对数据集的备注;
   例如：`cst_inversed` 表示对 `cst` 数据集进行了反转处理，`qz900_cst700_hybrid` 表示 900 张来自 `qz` 数据集的图片和 700 张来自 `cst` 数据集的图片混合在一起组成的数据集。

数据集格式：
一个数据集一个文件夹，文件夹内可以划分为 `train/`、`val/`、`test/` 子文件夹，分别存放训练集、验证集和测试集的图片和标签。每个子文件夹内的图片和标签文件名需要一一对应，例如 `img1.jpg` 对应 `img1.txt`。

该数据集的 YAML 文件放在数据集文件夹内，命名为 `dataset.yaml`，内容遵从 COCO 格式。

也可以在数据集文件夹中放一个 `readme.md` 用来更详细地记录数据集的备注。

## RUN Project Naming

文件夹命名格式 `<dataset>...-<model>_<note>-<run idx>`

1. `QZ_CST_Hybrid-yolov8s_relu`: 在备注为 `QZ_CST_Hybrid` 的数据集上训练，使用 `yolov8s` 模型，激活函数为 `relu`。
2. `QZ-yolov8s`: 在备注为 `QZ` 的数据集上训练，使用 `yolov8s` 模型，激活函数默认没有修改，为 `silu`。
3. ...

## Train

训练可以修改 `train.py` 脚本中的参数，或者直接命令行传入参数。

训练部分可以通过环境变量设置的有：

|       env       |                               remarks                               |
| :-------------: | :-----------------------------------------------------------------: |
| `YOLO_USE_RELU` | 设置为 `1` 则使用 `ReLU()` 激活函数，否则使用默认的 `SiLU()` 激活函数。 |

##Export

导出模型可以修改 `export.py` 脚本中的参数，或者直接命令行传入参数。

导出部分可以通过环境变量设置的有：
|             env             |                      remarks                       |
| :-------------------------: | :------------------------------------------------: |
| `ULTRALYTICS_EXPORT_FORMAT` | 使用 RKNN 官方后处理兼容的模型输出头，训练时不可用 |


