# 训练日志输出说明（TensorFlow + MNIST CNN）

> 本文档解释运行以下命令时在终端出现的典型输出及其含义：
>
> ```bash
> python -m src.train --epochs 25 --batch-size 128 --lr 1e-3
> ```
>
> 实际数值会因硬件、随机种子、TensorFlow 版本等略有差异，以下示例与解释可直接用于排查与汇报。

---

## 目录

- [训练日志输出说明（TensorFlow + MNIST CNN）](#训练日志输出说明tensorflow--mnist-cnn)
  - [目录](#目录)
  - [一、典型输出（示例）](#一典型输出示例)
  - [二、输出逐段解读](#二输出逐段解读)
    - [1）TensorFlow 启动信息](#1tensorflow-启动信息)
    - [2）模型结构摘要 `model.summary()`](#2模型结构摘要-modelsummary)
    - [3）训练进度（每个 Epoch）](#3训练进度每个-epoch)
    - [4）回调（Callbacks）行为](#4回调callbacks行为)
    - [5）评估与保存](#5评估与保存)
  - [三、字段含义对照表](#三字段含义对照表)
  - [四、实验设置卡片（复现实验条件）](#四实验设置卡片复现实验条件)
  - [五、FAQ](#五faq)
  - [附：Keras 3 保存/加载格式与常见报错](#附keras-3-保存加载格式与常见报错)
    - [报错现象](#报错现象)
    - [成因](#成因)
    - [推荐做法（本项目）](#推荐做法本项目)
    - [代码修改指引](#代码修改指引)

---

## 一、典型输出（示例）

```text
2025-09-30 14:11:41.040020: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
2025-09-30 14:11:41.040258: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results...
2025-09-30 14:11:41.071710: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions...

Model: "mnist_cnn"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 28, 28, 1)]       0
 conv2d (Conv2D)             (None, 28, 28, 32)        320
 batch_normalization (BatchN (None, 28, 28, 32)        128
 re_lu (ReLU)                (None, 28, 28, 32)        0
 conv2d_1 (Conv2D)           (None, 28, 28, 32)        9248
 batch_normalization_1       (None, 28, 28, 32)        128
 re_lu_1 (ReLU)              (None, 28, 28, 32)        0
 max_pooling2d (MaxPooling2D (None, 14, 14, 32)        0
 dropout (Dropout)           (None, 14, 14, 32)        0
 conv2d_2 (Conv2D)           (None, 14, 14, 64)        18496
 batch_normalization_2       (None, 14, 14, 64)        256
 re_lu_2 (ReLU)              (None, 14, 14, 64)        0
 conv2d_3 (Conv2D)           (None, 14, 14, 64)        36928
 batch_normalization_3       (None, 14, 14, 64)        256
 re_lu_3 (ReLU)              (None, 14, 14, 64)        0
 max_pooling2d_1            (None, 7, 7, 64)          0
 dropout_1 (Dropout)         (None, 7, 7, 64)          0
 flatten (Flatten)           (None, 3136)              0
 dense (Dense)               (None, 128)               401536
 batch_normalization_4       (None, 128)               512
 re_lu_4 (ReLU)              (None, 128)               0
 dropout_2 (Dropout)         (None, 128)               0
 dense_1 (Dense)             (None, 10)                1290
=================================================================
Total params: 469,098
Trainable params: 468,458
Non-trainable params: 640
_________________________________________________________________

Epoch 1/25
469/469 - 10s - loss: 0.3124 - accuracy: 0.9135 - val_loss: 0.0812 - val_accuracy: 0.9756
Epoch 2/25
469/469 - 7s - loss: 0.1428 - accuracy: 0.9631 - val_loss: 0.0509 - val_accuracy: 0.9852
Epoch 3/25
469/469 - 7s - loss: 0.1085 - accuracy: 0.9728 - val_loss: 0.0408 - val_accuracy: 0.9883
...
Epoch 25/25
469/469 - 7s - loss: 0.0451 - accuracy: 0.9911 - val_loss: 0.0237 - val_accuracy: 0.9942

Epoch 25: val_accuracy did not improve from 0.9942
[INFO] Test accuracy: 0.9942
[INFO] SavedModel saved to: outputs/models/saved_model
[INFO] Best .h5 checkpoint saved to: outputs/models/best_model.h5
```

> 注：如果某一轮验证精度创历史新高，会看到：
>
> `Epoch X: val_accuracy improved from A to B, saving model to outputs/models/best_model.h5`
>
> 若触发 EarlyStopping，训练会提前结束并回退到最佳权重。

---

## 二、输出逐段解读

### 1）TensorFlow 启动信息

* **`Could not find cuda drivers... GPU will not be used.`**：未检测到可用 CUDA/GPU，改用 CPU；**不是错误**。
* **`oneDNN custom operations are on`**：启用 oneDNN CPU 优化；数值可能有极小差异；**不是错误**。
* **`optimized to use available CPU instructions`**：该 TF 构建针对你 CPU 的 SIMD 指令做了优化；**不是错误**。

> 建议把这一段归类为“环境与硬件检测提示”，无需处理。

### 2）模型结构摘要 `model.summary()`

* **模型名**：`mnist_cnn`（`models/cnn.py` 的 `build_cnn`）。
* **层级结构**：两个卷积模块（`Conv2D + BN + ReLU` ×2 → `MaxPool` → `Dropout`），再接 `Flatten → Dense(128) → Dropout → Dense(10)`。
* **正则化**：BatchNorm（稳定训练）、Dropout（0.25/0.3/0.4，抑制过拟合）、L2（1e-4）。
* **参数量**：`Total params / Trainable / Non-trainable`。BN 的部分统计量为不可训练参数。

### 3）训练进度（每个 Epoch）

* **格式**：`469/469 - 7s - loss: ... - accuracy: ... - val_loss: ... - val_accuracy: ...`

  * `469/469`：该 epoch 的训练步数（≈ 60000/128）。
  * `7s`：该 epoch 用时。
  * `loss/accuracy`：训练集平均损失/精度。
  * `val_loss/val_accuracy`：验证集（此处用测试集）损失/精度。
* **解读**：理想趋势是 `loss` 下降、`accuracy/val_accuracy` 上升并趋稳。由于开启了轻量增强，训练指标可能略慢但泛化更佳。

### 4）回调（Callbacks）行为

1. **EarlyStopping**（`monitor=val_accuracy, patience=8, restore_best_weights=True`）

   * 连续 8 个 epoch 验证精度未创新高 → **提前停止**并**恢复到最佳权重**。
2. **ReduceLROnPlateau**（`monitor=val_loss, patience=3, factor=0.5, min_lr=1e-5`）

   * `val_loss` 3 轮不改善 → **学习率减半**。
3. **ModelCheckpoint**（`monitor=val_accuracy, save_best_only=True`）

   * 验证精度创新高 → **保存** `outputs/models/best_model.h5`。
4. **TensorBoard**（`log_dir=outputs/logs`）

   * 写入训练/验证曲线，可通过 `tensorboard --logdir outputs/logs` 查看。

### 5）评估与保存

* **`[INFO] Test accuracy: 0.9942`**：最终测试集精度（典型 ≥ 0.99）。
* **`SavedModel saved to ...`**：导出 SavedModel 目录格式（便于部署/跨平台加载）。
* **`Best .h5 checkpoint saved to ...`**：最佳权重（H5 文件）。预测脚本优先加载该文件。

---

## 三、字段含义对照表

| 日志片段                                   | 含义                   | 备注                             |
| -------------------------------------- | -------------------- | ------------------------------ |
| `Could not find cuda drivers...`       | 未检测到 CUDA/GPU，改用 CPU | 非错误                            |
| `oneDNN custom operations are on`      | 启用 CPU 高性能内核         | 非错误                            |
| `Model: "mnist_cnn"` & 表格              | 模型结构与参数量             | 来自 `model.summary()`           |
| `Epoch N/25`                           | 第 N 个训练轮次（最多 25）     | 早停则小于 25                       |
| `469/469`                              | 每个 epoch 的训练步数       | ≈ 60000/128                    |
| `loss / accuracy`                      | 训练集平均损失/精度           | 趋势：loss 降、acc 升                |
| `val_loss / val_accuracy`              | 验证集指标                | 代表泛化表现                         |
| `val_accuracy improved from A to B...` | 触发保存最佳模型             | `outputs/models/best_model.h5` |
| `ReduceLROnPlateau...`                 | 学习率衰减提示              | 当 `val_loss` 停滞触发              |
| `EarlyStopping`（无固定文案）                 | 达到耐心阈值后提前停止          | 回退最佳权重                         |
| `[INFO] Test accuracy: X.XXXX`         | 最终测试集精度              | 记录在结果                          |
| `SavedModel saved to...`               | 导出 SavedModel        | 用于部署                           |
| `Best .h5 checkpoint saved to...`      | 最佳权重路径               | 预测优先加载                         |

---

## 四、实验设置卡片（复现实验条件）

* **数据集**：MNIST（训练 60k，测试 10k；28×28 灰度，10 类）
* **模型**：`models/cnn.py` → `build_cnn`（Conv-BN-ReLU ×2 → MaxPool → Dropout）×2 → Flatten → Dense(128) → Dropout → Dense(10)
* **正则化**：Dropout（0.25/0.3/0.4）、BatchNorm、L2(1e-4)
* **数据增强**：RandomTranslation/Rotation/Zoom（轻量，默认开启）
* **优化器**：Adam(lr=1e-3)
* **批大小**：128
* **最大训练轮数**：25（可能 EarlyStopping 提前结束）
* **回调**：EarlyStopping(patience=8)、ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-5)、ModelCheckpoint(save_best_only=True)、TensorBoard
* **环境**：TensorFlow 2.15，Python 3.10（CPU/无 CUDA，也可用 GPU 版）
* **典型结果**：测试精度 99.2%~99.5%（示例 0.9942）

---

## 五、FAQ

**Q1：为什么训练/验证精度不是单调？**
随机性与数据增强会带来波动；关注总体趋势与验证集最佳点。

**Q2：为什么最后一轮不是最高精度？**
最佳往往出现在中间轮次；`ModelCheckpoint` 会保存当时最佳权重；`EarlyStopping` 会恢复到最佳权重。

**Q3：如何查看曲线？**
`tensorboard --logdir outputs/logs`，关注 `loss/val_loss` 与 `accuracy/val_accuracy` 曲线。

**Q4：如何复现实验或对比实验？**
固定种子（`--seed`）、记录环境版本，修改 `--epochs/--batch-size/--lr` 或打开/关闭 `--no-augment`，并保存所有运行日志。

---

## 附：Keras 3 保存/加载格式与常见报错

### 报错现象

```
ValueError: Invalid filepath extension for saving. Please add either a `.keras` extension ... Use `model.export(filepath)` if you want to export a SavedModel ... Received: filepath=outputs/models/saved_model.
```

### 成因

Keras 3（TF 2.15+ 搭配的独立 Keras）已将 `model.save(<目录>)` 行为调整为**仅支持**：

* 原生 Keras 格式：以 **`.keras`** 结尾的文件；
* 旧的 HDF5：以 **`.h5`** 结尾的文件；
* 若要导出 **TensorFlow SavedModel**（用于 TF-Serving / TFLite 转换等），需要调用 **`model.export(<目录>)`**，而不是 `model.save(<目录>)`。

### 推荐做法（本项目）

* 训练结束后保存两份：

  * **`best_model.h5`**（已通过 `ModelCheckpoint` 自动保存，预测脚本优先加载）；
  * **`best_model.keras`**（Keras 3 原生格式，兼容新 API）。
* 如需 TF Serving/TFLite：额外调用 `model.export('outputs/models/saved_model')`。

### 代码修改指引

* 将 `src/train.py` 中 `model.save(saved_model_dir)` 替换为：

  * `model.save(os.path.join(args.model_dir, 'best_model.keras'))`
  * （可选）`model.export(os.path.join(args.model_dir, 'saved_model'))`
* 将 `src/predict.py` 的加载逻辑改为：优先 `.h5` → `.keras`，不再尝试直接 `load_model('saved_model/')`（Keras 3 不支持直接从 SavedModel 目录用 `load_model` 加载）。
