import tensorflow as tf

print(f"TensorFlow 版本: {tf.__version__}")
print(f"已编译 CUDA 支持: {tf.test.is_built_with_cuda()}")

gpus = tf.config.list_physical_devices("GPU")
print(f"GPU 可用: {bool(gpus)}")
print(f"GPU 数量: {len(gpus)}")

if gpus:
    for idx, gpu in enumerate(gpus):
        print(f"GPU {idx}: {gpu.name}")