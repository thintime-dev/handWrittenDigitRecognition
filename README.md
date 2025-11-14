# MNIST 手写数字识别（TensorFlow + OpenCL）

使用 TensorFlow 构建卷积神经网络识别 MNIST 手写数字，并提供基于 OpenCL/FPGA 的推理路径。

## 环境准备

### 先决条件
- 推荐使用 Conda（Anaconda 或 Miniconda）。
- Python 3.10（`env/environment.yml` 已固定版本）。
- 需要下载 MNIST 数据集；脚本会自动从网络获取。
- 可选：若希望使用 GPU，请预先安装与硬件匹配的 CUDA 驱动及 GPU 版 TensorFlow。

### 创建并激活环境
```bash
conda env create -f env/environment.yml
```

该环境安装了 TensorFlow 2.15、PyOpenCL、OpenCV、scikit-image 等依赖。若需 GPU 支持，可在激活环境后改用 `pip install tensorflow[and-cuda]==2.15.0` 并按注释调整。

### 快速自检
python -c "import tensorflow as tf; print(tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```
若成功输出版本号且列出 GPU，则说明 TensorFlow 工作正常；否则将自动落到 CPU。

## 项目结构
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

1. **安装依赖（见上节）。**

2. **训练模型**
   ```bash
   python -m src.train --epochs 25 --batch-size 128 --lr 1e-3
   ```
   - 默认会下载 MNIST、开启轻量数据增强，并将日志写入 `outputs/logs/`。
   - 回调组合：`EarlyStopping`、`ReduceLROnPlateau`、`ModelCheckpoint`、`TensorBoard`。
   - 训练结束后会保存 `outputs/models/best_model.h5` 与 `best_model.keras`，并在控制台输出测试集精度。
   - 常用参数：`--no-augment` 关闭增强，`--seed` 固定随机种子，`--model-dir` 指定模型输出目录。
![alt text](./res/readme/train.png)
![alt text](./res/readme/trainx.png)

1. **导出权重（用于 OpenCL/FPGA）**
   ```bash
   python -m src.export_weights --model-dir outputs/models --out outputs/models/weights_fp32.npz
   ```
   该脚本会读取训练生成的 `.h5` 或 `.keras` 文件，转换卷积核布局，并将 BN 超参数写入 `weights_fp32.npz`。
![alt text](./res/readme/export_weight.png)

2. **TensorFlow 批量预测**
   ```bash
   python -m src.predict --model-dir outputs/models --img-dir res/test_images --out-csv outputs/predictions.csv
   ```
   - 支持 `png/jpg/jpeg/bmp/tif` 等格式。
   - 会打印每张图的预测类别和置信度；若提供 `--out-csv`，结果也会保存为 CSV。
![alt text](./res/readme/tf_predict.png)

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
![alt text](./res/readme/ocl_predict.png)

1. **结果对比（可选）**
   ```bash
   python tests/compare_ocl_vs_tf.py res/test_images
   ```
   脚本会运行上述两条预测命令并统计一致率，便于验证 OpenCL 流水线的正确性。
![alt text](./res/readme/compare1.png)
![alt text](./res/readme/compare2.png)
## 数据与输出说明
- MNIST 数据集由 `src/data.py` 自动下载到本地缓存（通常为 `~/.keras/datasets/`），无需手动准备。
- 所有模型、日志、预测结果默认写入 `outputs/`。训练前可清理该目录或指定新的输出路径。
- 示例图片位于 `res/`，可用于验证推理脚本；也可以将自己的 28×28 灰度图放入自定义目录。

## 具体实现原理

### 数据预处理与输入管线

- `src/data.py` 使用 `tf.keras.datasets.mnist` 拉取原始 28×28 灰度图，并借助 `src/utils.normalize_img` 将像素归一化到 `[0,1]` 浮点范围，同时扩展出单通道维度以适配 NHWC 张量格式。
- 训练阶段默认启用 `build_tfrecord_augmenter()` 定义的轻量数据增强：随机平移（±5%）、旋转（±0.06rad）与缩放（±5%），以提升模型对书写体扰动的鲁棒性。
- 经过增强的样本通过 `tf.data.Dataset` 构建流水线完成 `shuffle → map → batch → prefetch`，保证 GPU/CPU 训练阶段的输入吞吐。

### 卷积神经网络结构

- `models/cnn.py` 构建两段卷积块：每段包含两层 `Conv2D(3×3)` + `BatchNormalization` + `ReLU`，随后接 `MaxPooling(2×2)` 与递增的 `Dropout`，在 MNIST 上取得 <1% 的测试错误率。
- 头部使用 `Flatten → Dense(128)` 与 `BatchNormalization`、`Dropout(0.4)`，最后输出 `Dense(10, softmax)`；所有可训练层应用 L2 权重衰减（1e-4）以缓解过拟合。
- `src/train.py` 采用 `Adam`（初始学习率 1e-3）优化，结合 `EarlyStopping`、`ReduceLROnPlateau`、`ModelCheckpoint` 与 `TensorBoard` 监控提升训练稳定性，并默认训练 25 轮批量大小 128。

### 权重导出与 BN 融合

- `src/export_weights.py` 在推理前将 Keras 模型转换成适配 OpenCL 核的权重包：卷积核从 Keras 的 `(Kh, Kw, Cin, Cout)` 置换为 `(Cout, Kh, Kw, Cin)`，全连接权重保持矩阵形式，并序列化为 `weights_fp32.npz`。
- BatchNorm 参数（`gamma/beta/mean/var/epsilon`）同时写入权重包，便于在 OpenCL 主机侧复现推理阶段的仿射变换。
- 如需进一步减少推理阶段的算子数量，可运行 `src/fold_bn.py` 将 BN 融入相邻的卷积或全连接层，生成 `weights_fused.npz`，这会提前吸收缩放与偏移关系，降低设备端访存。

### OpenCL 推理链路

- `src/ocl_runtime.py` 初始化平台/设备并编译 `ocl/kernels.cl` 中的计算核，提供 `conv3x3_nhwc` 与 `maxpool2x2_nhwc` 等封装；推理时主机端负责 BN 与 ReLU，以简化内核逻辑并方便后续定点优化。
- `src/predict_ocl.py` 读取导出的 NPZ 权重，将输入图片通过 `src/utils.load_and_prepare_image` 自动反色、居中、缩放后规整为 NHWC 格式，再依次完成卷积块 → 池化 → 全连接 → Softmax 的手动前向过程，与 TensorFlow 版本完全对齐。
- OpenCL 内核采用逐输出像素的全展开实现，重点关注正确性与跨设备可移植性；若部署在 FPGA，可通过传入编译好的 `.aocx/.xclbin` 在 `OCLRuntime` 中直接加载预编译位流。