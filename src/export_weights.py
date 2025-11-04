# src/export_weights.py
import os
import argparse
import numpy as np
import tensorflow as tf

def load_model_any(path_h5, path_keras):
    if os.path.exists(path_h5):
        print(f"[INFO] Loading {path_h5}")
        return tf.keras.models.load_model(path_h5, compile=False)
    if os.path.exists(path_keras):
        print(f"[INFO] Loading {path_keras}")
        return tf.keras.models.load_model(path_keras, compile=False)
    raise FileNotFoundError("No model found (.h5 or .keras).")

def nhwc_kernel_from_keras(w_keras):
    """
    Keras Conv2D kernel layout: (Kh, Kw, Cin, Cout)
    我们的 OpenCL 内核期望 (Cout, Kh, Kw, Cin)
    """
    Kh, Kw, Cin, Cout = w_keras.shape
    return np.transpose(w_keras, (3,0,1,2)).astype(np.float32)

def dense_as_mat(w_keras):
    # Keras Dense: (in_features, out_features)
    # 这里保持该布局，到 CPU 侧做 GEMV
    return w_keras.astype(np.float32)

def export(model, out_npz):
    blobs = {}
    conv_idx = 0
    bn_idx = 0
    dense_idx = 0

    for lyr in model.layers:
        cls = lyr.__class__.__name__
        if cls == "Conv2D":
            w, b = lyr.get_weights()
            W = nhwc_kernel_from_keras(w)  # (Cout,3,3,Cin)
            blobs[f"conv{conv_idx}_W"] = W
            blobs[f"conv{conv_idx}_b"] = b.astype(np.float32)
            conv_idx += 1

        elif cls == "BatchNormalization":
            gamma, beta, moving_mean, moving_var = lyr.get_weights()
            blobs[f"bn{bn_idx}_gamma"] = gamma.astype(np.float32)
            blobs[f"bn{bn_idx}_beta"]  = beta.astype(np.float32)
            blobs[f"bn{bn_idx}_mean"]  = moving_mean.astype(np.float32)
            blobs[f"bn{bn_idx}_var"]   = moving_var.astype(np.float32)
            blobs[f"bn{bn_idx}_eps"]   = np.array([getattr(lyr, 'epsilon', 1e-3)], dtype=np.float32)
            bn_idx += 1

        elif cls == "Dense":
            w, b = lyr.get_weights()
            blobs[f"dense{dense_idx}_W"] = dense_as_mat(w)
            blobs[f"dense{dense_idx}_b"] = b.astype(np.float32)
            dense_idx += 1

    blobs["net_cfg"] = np.array([conv_idx, bn_idx, dense_idx], dtype=np.int32)
    np.savez_compressed(out_npz, **blobs)
    print(f"[INFO] Exported weights to: {out_npz}")
    print(f"[INFO] Summary: conv={conv_idx}, bn={bn_idx}, dense={dense_idx}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="outputs/models")
    ap.add_argument("--out", default="outputs/models/weights_fp32.npz")
    args = ap.parse_args()

    model = load_model_any(
        os.path.join(args.model_dir, "best_model.h5"),
        os.path.join(args.model_dir, "best_model.keras"),
    )
    export(model, args.out)

if __name__ == "__main__":
    main()

