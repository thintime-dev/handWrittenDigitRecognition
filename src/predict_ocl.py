# src/predict_ocl.py
import os
import argparse
import json
import numpy as np
from glob import glob

from .utils import load_and_prepare_image, ensure_dir
from .ocl_runtime import OCLRuntime

def load_weights(npz_path):
    blobs = np.load(npz_path)
    return blobs

# def bn_infer(x, gamma, beta, mean, var, eps=1e-5):
#     # x: NHWC flat
#     # BN 是逐通道，NHWC 的 C 维度最后；这里为了简洁，把 flat 展成 (H*W, C) 做广播
#     # 但我们在每步都知道 H,W,C，所以在调用处 reshape。
#     return gamma * (x - mean) / np.sqrt(var + eps) + beta

def bn_infer(x, gamma, beta, mean, var, eps):
    # x 的最后一维是通道/特征维；gamma/beta/mean/var 形状与该维一致
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def relu_inplace(x):
    x[x < 0] = 0
    return x

def dense_gemv(x, W, b):
    # x: (N,), W: (N, M), b: (M,)
    return x @ W + b

def softmax(x):
    m = x.max()
    e = np.exp(x - m)
    return e / e.sum()

def predict_image(ocl, blobs, img_arr):
    H, Wd, Cin = 28, 28, 1
    x = img_arr.astype(np.float32).reshape(-1)  # flat NHWC

    # Block 1: conv0->bn0->relu, conv1->bn1->relu, pool
    # conv0
    W = blobs["conv0_W"]; b = blobs["conv0_b"]
    out = ocl.conv3x3_nhwc(x, W.reshape(-1), b, H, Wd, Cin, 32)
    out_rs = out.reshape(H*Wd, 32)
    eps = float(blobs["bn0_eps"][0])
    out_bn = bn_infer(out_rs, blobs["bn0_gamma"], blobs["bn0_beta"],
                      blobs["bn0_mean"], blobs["bn0_var"], eps)
    out_bn = relu_inplace(out_bn)
    x = out_bn.reshape(-1)

    # conv1
    W = blobs["conv1_W"]; b = blobs["conv1_b"]
    out = ocl.conv3x3_nhwc(x, W.reshape(-1), b, H, Wd, 32, 32)
    out_rs = out.reshape(H*Wd, 32)
    eps = float(blobs["bn1_eps"][0])
    out_bn = bn_infer(out_rs, blobs["bn1_gamma"], blobs["bn1_beta"],
                      blobs["bn1_mean"], blobs["bn1_var"], eps)
    out_bn = relu_inplace(out_bn)
    x = out_bn.reshape(-1)

    # pool
    x = ocl.maxpool2x2_nhwc(x, H, Wd, 32)
    H, Wd, Cin = 14, 14, 32

    # Block 2: conv2->bn2->relu, conv3->bn3->relu, pool
    # conv2
    W = blobs["conv2_W"]; b = blobs["conv2_b"]
    out = ocl.conv3x3_nhwc(x, W.reshape(-1), b, H, Wd, Cin, 64)
    out_rs = out.reshape(H*Wd, 64)
    eps = float(blobs["bn2_eps"][0])
    out_bn = bn_infer(out_rs, blobs["bn2_gamma"], blobs["bn2_beta"],
                      blobs["bn2_mean"], blobs["bn2_var"], eps)
    out_bn = relu_inplace(out_bn)
    x = out_bn.reshape(-1)

    # conv3
    W = blobs["conv3_W"]; b = blobs["conv3_b"]
    out = ocl.conv3x3_nhwc(x, W.reshape(-1), b, H, Wd, 64, 64)
    out_rs = out.reshape(H*Wd, 64)
    eps = float(blobs["bn3_eps"][0])
    out_bn = bn_infer(out_rs, blobs["bn3_gamma"], blobs["bn3_beta"],
                      blobs["bn3_mean"], blobs["bn3_var"], eps)
    out_bn = relu_inplace(out_bn)
    x = out_bn.reshape(-1)

    # pool
    x = ocl.maxpool2x2_nhwc(x, H, Wd, 64)
    H, Wd, Cin = 7, 7, 64

    # Head: Flatten -> Dense(128) -> BN(bn4) -> ReLU -> Dense(10) -> Softmax
    flat = x  # already flat
    W0 = blobs["dense0_W"]; b0 = blobs["dense0_b"]
    h  = dense_gemv(flat, W0, b0)   # (128,)

    # dense0 后的 BN（bn4）
    eps = float(blobs["bn4_eps"][0])
    h = bn_infer(h, blobs["bn4_gamma"], blobs["bn4_beta"],
                 blobs["bn4_mean"], blobs["bn4_var"], eps)
    h = relu_inplace(h)

    W1 = blobs["dense1_W"]; b1 = blobs["dense1_b"]
    logits = dense_gemv(h, W1, b1)
    prob = softmax(logits)
    pred = int(np.argmax(prob))
    conf = float(np.max(prob))
    return pred, conf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True, help="图片目录（与 predict.py 一样）")
    ap.add_argument("--weights", default="outputs/models/weights_fp32.npz")
    ap.add_argument("--kernel", default="ocl/kernels.cl")
    ap.add_argument("--binary", default="", help="FPGA 位流（二进制），如 .aocx / .xclbin。留空则从源码编译用于 NVIDIA/CPU 调试")
    ap.add_argument("--platform", default="", help="OpenCL 平台关键字（如 NVIDIA / Intel / Xilinx）")
    ap.add_argument("--device", default="", help="OpenCL 设备关键字（如 GeForce / FPGA）")
    args = ap.parse_args()

    blobs = load_weights(args.weights)
    ocl = OCLRuntime(args.kernel, binary_path=(args.binary or None),
                     platform_hint=(args.platform or None),
                     device_hint=(args.device or None))

    exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff")
    paths = sorted([p for ext in exts for p in glob(os.path.join(args.img_dir, ext))])
    if not paths:
        print(f"[WARN] No images found in {args.img_dir}")
        return

    for p in paths:
        arr = load_and_prepare_image(p)  # (28,28,1) float32 [0,1]
        pred, conf = predict_image(ocl, blobs, arr)
        print(f"{os.path.basename(p)} -> {pred} (conf={conf:.4f})")

if __name__ == "__main__":
    main()

