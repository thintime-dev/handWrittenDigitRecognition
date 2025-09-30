# src/predict.py
import os
import argparse
import numpy as np
import tensorflow as tf
from glob import glob
from .utils import load_and_prepare_image, ensure_dir

def load_model(model_dir: str):
    # 优先加载 .h5，其次 .keras（Keras3 原生）。不再直接从 SavedModel 目录加载。
    h5_path = os.path.join(model_dir, "best_model.h5")
    keras_path = os.path.join(model_dir, "best_model.keras")
    if os.path.exists(h5_path):
        print(f"[INFO] Loading model from {h5_path}")
        return tf.keras.models.load_model(h5_path, compile=False)
    if os.path.exists(keras_path):
        print(f"[INFO] Loading model from {keras_path}")
        return tf.keras.models.load_model(keras_path, compile=False)
    raise FileNotFoundError(
        f"No model file found. Expected one of: {h5_path} or {keras_path}."
    )

def predict_dir(model, img_dir: str, output_csv: str = None):
    paths = sorted(
        [p for ext in ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff")
         for p in glob(os.path.join(img_dir, ext))]
    )
    if not paths:
        print(f"[WARN] No images found in {img_dir}")
        return []

    results = []
    for p in paths:
        arr = load_and_prepare_image(p)  # (28,28,1), 0-1
        pred = model.predict(arr[np.newaxis, ...], verbose=0)[0]  # (10,)
        cls = int(np.argmax(pred))
        conf = float(np.max(pred))
        results.append((os.path.basename(p), cls, conf))

    # 打印与保存
    for name, cls, conf in results:
        print(f"{name} -> {cls} (conf={conf:.4f})")

    if output_csv:
        ensure_dir(os.path.dirname(output_csv) or ".")
        with open(output_csv, "w", encoding="utf-8") as f:
            f.write("filename,pred,confidence\n")
            for name, cls, conf in results:
                f.write(f"{name},{cls},{conf:.6f}\n")
        print(f"[INFO] Results saved to {output_csv}")

    return results

def main(args):
    model = load_model(args.model_dir)
    predict_dir(model, args.img_dir, args.out_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="outputs/models",
                        help="训练产生的模型目录（含 best_model.h5 / saved_model/）")
    parser.add_argument("--img-dir", type=str, required=True,
                        help="存放手写数字图片的本地目录")
    parser.add_argument("--out-csv", type=str, default="outputs/predictions.csv",
                        help="可选：将预测结果保存为 CSV")
    args = parser.parse_args()
    main(args)

