# MNIST 手写数字识别（TensorFlow + CNN）

## 环境准备
```bash
conda env create -f env/environment.yml
conda activate mnist-cnn-tf
```
## 训练模型
```bash
 python -m src.train --epochs 25 --batch-size 128 --lr 1e-3
```
这条命令是在**以模块方式**运行你项目里的训练入口，并显式指定训练超参数。逐项拆解如下：

### 命令分解

* `python -m src.train`

  * `python -m` 表示“**以模块**的方式运行”。它会把 `src` 视为包、把 `src/train.py` 当作模块 `src.train` 来执行，这样**相对导入**（如 `from .data import ...`）才能正常工作。
  * 等价理解：运行 `src/train.py` 里的

    ```python
    if __name__ == "__main__":
        main(args)
    ```

    并使用其中的 `argparse` 解析参数。

* `--epochs 25`

  * 训练最多 **25 个 epoch**（全量训练集被遍历 25 次）。
  * 我们启用了 `EarlyStopping`，如果验证集精度长时间未提升，训练会**提前停止**。

* `--batch-size 128`

  * 每个梯度更新使用 **128 张样本**。
  * MNIST 训练集有 60,000 张图，因此每个 epoch 的步数大约是 `60000 / 128 ≈ 469`（日志里看到的 `469/469`）。

* `--lr 1e-3`

  * 初始**学习率**为 `0.001`（Adam 优化器）。
  * 回调里还有 `ReduceLROnPlateau`，当验证集 `val_loss` 停滞时会把学习率减半，最低降到 `1e-5`。

### 运行这条命令会做什么？

1. **准备数据**：下载/加载 MNIST，归一化到 `[0,1]`，按 `batch_size` 打包；默认带**轻量数据增强**（平移/旋转/缩放）。
2. **构建模型**：用 `models/cnn.py` 的 `build_cnn()` 创建 CNN（Conv+BN+ReLU×2 → MaxPool → Dropout，重复两次；再接 Dense）。
3. **编译**：`Adam(lr=1e-3)`，损失是 `sparse_categorical_crossentropy`，评价指标 `accuracy`。
4. **训练**：最多 25 个 epoch；配合回调：

   * `EarlyStopping(patience=8, restore_best_weights=True)`
   * `ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-5)`
   * `ModelCheckpoint(save_best_only=True)` 保存 `outputs/models/best_model.h5`
   * `TensorBoard` 日志写到 `outputs/logs/`
5. **评估与保存**：打印测试集精度；保存 `best_model.keras`（或 `.h5`）等产物（按你现在的修订）。

### 小贴士

* 省略某个参数就会用 `train.py` 里设置的默认值（如 `epochs=25, batch_size=128, lr=1e-3`）。
* 用 `-m` 的好处是**包内相对导入稳定**；直接 `python src/train.py` 在某些环境下可能导入失败。
* GPU 是否被用到**不由这条命令决定**，而取决于你的环境是否安装了 GPU 版 TensorFlow + 驱动；没装就自动走 CPU（启动日志会提示）。

## 常见变体

* 关掉数据增强：

  ```bash
  python -m src.train --epochs 25 --no-augment
  ```
* 调大学习率、减小 batch：

  ```bash
  python -m src.train --lr 5e-3 --batch-size 64
  ```
* 固定随机种子：

  ```bash
  python -m src.train --seed 123
  ```

一句话总结：
**这条命令以“包模块”方式运行训练脚本，并用你给定的 `epochs=25`、`batch_size=128`、`lr=1e-3` 来训练并保存一个在 MNIST 上高精度的 CNN 模型。**

## 批量预测本地目录
```bash
python -m src.predict --img-dir your_images_dir --out-csv outputs/predictions.csv
```