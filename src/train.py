# src/train.py
import os
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from models.cnn import build_cnn
from .data import make_datasets
from .utils import ensure_dir, set_seed

def main(args):
    set_seed(args.seed)

    ensure_dir(args.model_dir)
    ensure_dir(args.log_dir)

    # 数据
    train_ds, test_ds = make_datasets(batch_size=args.batch_size, augment=not args.no_augment)

    # 模型
    model = build_cnn(input_shape=(28,28,1), num_classes=10)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # 回调
    ckpt_path = os.path.join(args.model_dir, "best_model.h5")
    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5, monitor="val_loss"),
        ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        TensorBoard(log_dir=args.log_dir)
    ]

    # 训练
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2
    )

    # 评估
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"[INFO] Test accuracy: {test_acc:.4f}")

    # 保存为 Keras 原生格式（与 Keras3 兼容）
    keras_path = os.path.join(args.model_dir, "best_model.keras")
    model.save(keras_path)
    print(f"[INFO] Keras model (.keras) saved to: {keras_path}")
    print(f"[INFO] Best .h5 checkpoint saved to: {ckpt_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-dir", type=str, default="outputs/models")
    parser.add_argument("--log-dir", type=str, default="outputs/logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true",
                        help="关闭数据增强（默认开启以提升泛化与上限）")
    args = parser.parse_args()
    main(args)

