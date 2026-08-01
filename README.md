# MNIST 手写数字识别（TensorFlow + OpenCL）

## 前言

本项目使用 TensorFlow 构建与训练卷积神经网络识别 MNIST 手写数字，并提供了基于 OpenCL/FPGA 的推理路径，能实现使用 tensorflow 和 OpenCL 两种方式（会自动选择是否进行GPU版本运行）进行对本地目录下的所有手写数字图片进行识别，并且进行置信度对比。

## 环境准备

### 创建并激活环境

使用以下命令一键创建虚拟环境安装依赖并激活虚拟环境。若需 GPU 支持，可在激活环境后改用 `pip install tensorflow[and-cuda]==2.15.0` 。

```bash
conda env create -f env/environment.yml
conda activate mnist-cnn-tf
```

### GPU检测

如果使用GPU,可以使用以下命令快速自检，判断是否具备GPU对应依赖。

```
python -c "import tensorflow as tf; print(tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

若成功输出版本号且列出 GPU，则说明 TensorFlow 工作正常；否则将自动落到 CPU。

## 项目结构



```
.
├── env/
│   └── environment.yml        # Conda 环境定义
├── src/
│   ├── train.py               # 训练入口，包含 CLI 参数解析
│   ├── predict_ocl.py         # OpenCL 推理入口（CPU/GPU/FPGA）
│   ├── export_weights.py      # 导出权重为 NPZ，供 OCL 管线使用
│   ├── ocl_runtime.py         # OpenCL 内核封装与运行时
│   ├── fold_bn.py             # BN 融合工具（可选）
│   ├── data.py                # MNIST 下载、预处理与增强
│   └── utils.py               # 公共工具函数
├── models/
│   └── cnn.py                 # CNN 架构定义
├── ocl/
│   ├── kernels.cl             # OpenCL 核函数（卷积/池化等）
│   └── README.md              # OCL 内核和部署说明
├── res/
│   └── test_images/           # 示例手写数字
├── outputs/                   # 训练与推理生成的产物
│   ├── logs/                  # TensorBoard 日志
│   └── models/                # 导出的 best_model.h5/.keras/.npz
├── tests/
│   └── compare_ocl_vs_tf.py   # 对比 OCL 与 TF 推理结果
└── README.md
```

## 运行流程

1. **安装依赖**

2. **训练模型**
   
   ```bash
   python -m src.train --epochs 25 --batch-size 128 --lr 1e-3
   ```
   - 默认会下载 MNIST、开启轻量数据增强，并将日志写入 `outputs/logs/`。
   - 回调组合：`EarlyStopping`、`ReduceLROnPlateau`、`ModelCheckpoint`、`TensorBoard`。
   - 训练结束后会保存 `outputs/models/best_model.h5` 与 `best_model.keras`，并在控制台输出测试集精度。
   - 常用参数：`--no-augment` 关闭增强，`--seed` 固定随机种子，`--model-dir` 指定模型输出目录。

1. **导出权重（用于 OpenCL/FPGA）**
   ```bash
   python -m src.export_weights --model-dir outputs/models --out outputs/models/weights_fp32.npz
   ```
   该脚本会读取训练生成的 `.h5` 或 `.keras` 文件，转换卷积核布局，并将 BN 超参数写入 `weights_fp32.npz`。
   

2. **TensorFlow 批量预测**
   ```bash
   python -m src.predict --model-dir outputs/models --img-dir res/test_images --out-csv outputs/predictions.csv
   ```
   - 支持 `png/jpg/jpeg/bmp/tif` 等格式。
   - 会打印每张图的预测类别和置信度；若提供 `--out-csv`，结果也会保存为 CSV。

1. **OpenCL 推理（CPU/GPU/FPGA）**
   ```bash
   python -m src.predict_ocl \
       --img-dir res/test_images \
       --weights outputs/models/weights_fp32.npz \
       --kernel ocl/kernels.cl \
       --platform NVIDIA \
       --device GeForce
   ```
   - `--platform`/`--device` 用于选择具体 OpenCL 平台（示例为 NVIDIA GPU）。
   - FPGA 部署可参考 `docs/fpga_deployment_guide.md`，并通过 `--binary` 指定已编译的 `.aocx`/`.xclbin`。
   - 推理时会逐张图片打印预测结果。

1. **结果对比**
   
   ```bash
   python tests/compare_ocl_vs_tf.py res/test_images
   ```
   脚本会运行上述两条预测命令并统计一致率，便于验证 OpenCL 流水线的正确性。
   
## 数据与输出说明
- MNIST 数据集由 `src/data.py` 自动下载到本地缓存（通常为 `~/.keras/datasets/`），无需手动准备。
- 所有模型、日志、预测结果默认写入 `outputs/`。训练前可清理该目录或指定新的输出路径。
- 示例图片位于 `res/`，可用于验证推理脚本；也可以将自己的 28×28 灰度图放入自定义目录。
