# src/utils.py
import os
import math
import numpy as np
import cv2
from typing import Tuple
import random
import tensorflow as tf
from PIL import Image

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def normalize_img(x: np.ndarray) -> np.ndarray:
    # 0-255 -> 0-1
    return (x.astype("float32") / 255.0)

def auto_invert_if_needed(img_gray: np.ndarray) -> np.ndarray:
    """
    自动判断是否需要反相（一些本地图片是白底黑字，和 MNIST 相反），
    以整体平均亮度为依据：亮度高说明白底为主 -> 反相。
    """
    mean_val = img_gray.mean()
    return 255 - img_gray if mean_val > 127 else img_gray

def center_and_resize(img_gray: np.ndarray, size: Tuple[int,int]=(28,28)) -> np.ndarray:
    """
    对灰度图进行简单的前景定位、居中、再缩放到 28x28。
    方法：Otsu 阈值 -> 找最小外接矩形 -> 裁剪 -> padding 到正方形 -> resize
    """
    # 防止全黑/全白导致 Otsu 失败
    img = img_gray.copy()
    if img.max() == img.min():
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    # 二值化（反转后前景应该更亮，这里用原图再判）
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 找轮廓
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    crop = img[y:y+h, x:x+w]

    # padding 成正方形
    s = max(w, h)
    canvas = np.zeros((s, s), dtype=np.uint8)
    y_off = (s - h) // 2
    x_off = (s - w) // 2
    canvas[y_off:y_off+h, x_off:x_off+w] = crop

    # resize
    out = cv2.resize(canvas, size, interpolation=cv2.INTER_AREA)
    return out

def load_and_prepare_image(path: str) -> np.ndarray:
    """
    读取任意单张图片，输出形状 (28,28,1)，像素 [0,1] 的 numpy。
    支持 png/jpg 等，自动灰度化、自动反相、居中、缩放。
    """
    img = Image.open(path).convert("L")  # 灰度
    img = np.array(img)
    img = auto_invert_if_needed(img)
    img = center_and_resize(img, (28, 28))
    img = normalize_img(img)
    img = img.reshape(28, 28, 1)
    return img

def build_tfrecord_augmenter():
    """
    轻量数据增强（仅训练时使用），帮助泛化：
    - 轻微平移、旋转、缩放
    """
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomRotation(0.06),
            tf.keras.layers.RandomZoom(0.05)
        ],
        name="data_augmentation"
    )
    return data_augmentation

