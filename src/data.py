# src/data.py
import tensorflow as tf
import numpy as np
from .utils import normalize_img, build_tfrecord_augmenter

def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # 形状调整为 (N, 28, 28, 1)
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    # 归一化
    x_train = normalize_img(x_train)
    x_test = normalize_img(x_test)
    return (x_train, y_train), (x_test, y_test)

def make_datasets(batch_size=128, buffer_size=10000, augment=True):
    (x_train, y_train), (x_test, y_test) = load_mnist()
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    if augment:
        aug = build_tfrecord_augmenter()
        train_ds = train_ds.shuffle(buffer_size).map(
            lambda x, y: (aug(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, test_ds

