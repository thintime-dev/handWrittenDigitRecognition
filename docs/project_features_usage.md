# MNIST 手写数字识别项目功能与使用指南

## 功能总览
- **CNN 模型训练**：基于 TensorFlow/Keras 的 `src/train.py`，提供早停、学习率自适应、TensorBoard 日志与模型检查点。默认启用轻量数据增强提升泛化能力。
- **数据预处理流程**：`src/data.py` 与 `src/utils.py` 自动下载 MNIST 数据集、完成归一化，同时对本地图片执行灰度化、自动反相、定位居中与缩放到 28×28。
- **批量图片预测**：`src/predict.py` 支持加载 `.h5` 或 `.keras` 模型，对指定目录中的图片批量推理，并可导出 CSV。
- **模型权重导出**：`src/export_weights.py` 将 Keras 模型权重转换为 OpenCL 端可读取的 `weights_fp32.npz` 格式。
- **OpenCL 推理链路**：`src/predict_ocl.py` 搭配 `src/ocl_runtime.py` 与 `ocl/kernels.cl`，可在 GPU/FPGA 等 OpenCL 设备上复现 CNN 推理流程。
- **一致性验证脚本**：`tests/compare_ocl_vs_tf.py` 用于对比 TensorFlow 与 OpenCL 推理结果，输出分类一致率与置信度差异。

## 环境准备
```bash
conda env create -f env/environment.yml
conda activate mnist-cnn-tf
```

> 说明：环境文件已包含 TensorFlow、OpenCV、PyOpenCL 等依赖；如需 GPU 版 TensorFlow，可按 `env/environment.yml` 中的注释调整。

## 训练模型
```bash
python -m src.train --epochs 25 --batch-size 128 --lr 1e-3
```

- MNIST 数据集会在首次运行时自动下载并缓存。
- 训练日志写入 `outputs/logs`，最佳模型保存为 `outputs/models/best_model.h5` 与 `best_model.keras`。
- 常用参数：
  - `--no-augment`：关闭训练阶段的数据增强。
  - `--seed`：固定随机种子（默认 42）。
  - `--model-dir`、`--log-dir`：自定义模型与日志输出路径。

## 使用 TensorFlow 模型预测本地图片
```bash
python -m src.predict --img-dir your_images_dir --out-csv outputs/predictions.csv
```

- 支持 `png/jpg/jpeg/bmp/tif` 等常见格式。
- 推理时会逐张打印结果；若指定 `--out-csv`，则在目标路径生成 `filename,pred,confidence` 列的结果文件。
- 图片预处理由 `src/utils.load_and_prepare_image()` 完成：自动灰度化、必要时反相、前景居中裁剪、缩放到 28×28 并归一化。
- 示例测试图片可参考 `res/test_images` 与 `res/test_images_noisy`。

## 导出模型权重（面向 OpenCL 推理）
```bash
python -m src.export_weights --model-dir outputs/models --out outputs/models/weights_fp32.npz
```

- 脚本会优先读取 `best_model.h5`，其次读取 `best_model.keras`。
- 导出的 `weights_fp32.npz` 含卷积、批归一化、全连接层的权重与偏置，并记录网络结构配置和 BN 层的 epsilon。

## 使用 OpenCL 推理
```bash
python -m src.predict_ocl \
  --img-dir your_images_dir \
  --weights outputs/models/weights_fp32.npz \
  --kernel ocl/kernels.cl \
  --platform NVIDIA \
  --device GeForce
```

- `--platform` 与 `--device` 用于筛选具体的 OpenCL 平台与设备关键字，可根据实际硬件调整；若留空则默认选择第一项。
- 如有 FPGA 位流或预编译二进制，可通过 `--binary` 指定文件路径。
- 运行时会打印所选平台/设备，并对目录下图片逐一输出预测结果。

## 验证 OpenCL 与 TensorFlow 推理一致性
```bash
python tests/compare_ocl_vs_tf.py res/test_images
```

- 脚本运行前需要安装对应 GPU 的OpenCl ICD。以 NVIDIA 为例，在 Arch 中需要先安装 `opencl-nvidia` （它依赖 nvidia-utils 并安装 NVIDIA 的 OpenCL ICD）：
  ```bash
  sudo pacman -S opencl-nvidia
  ```
- 脚本会先调用 TensorFlow 推理生成参考 CSV，再运行 OpenCL 推理。
- 最终打印样本数量、预测一致数量及一致率，并报告最大置信度差异，便于快速验证权重导出与 OpenCL 实现是否正确。

## 目录与产物约定
- `outputs/models`：训练后的 `.h5`、`.keras` 模型以及 `weights_fp32.npz`。
- `outputs/logs`：TensorBoard 日志，可通过 `tensorboard --logdir outputs/logs` 可视化。
- `outputs/predictions.csv`：批量预测时可选生成的结果文件。
- `res/test_images*`：内置测试图片，可用于开发阶段自测。
- `docs/`：项目文档目录（本文件与训练日志说明等）。

## 常见问题
- **训练太慢？** 检查是否启用了 GPU 版本 TensorFlow 或合理调整 `--batch-size`。
- **本地图片预测错误？** 确认图片分辨率与背景；项目已自动尝试反相与居中，但极端情况下可手动预处理。
- **OpenCL 推理报错？** 请确认已安装 `pyopencl` 对应的驱动与 ICD，并核对 `--platform`/`--device` 参数是否匹配硬件。

