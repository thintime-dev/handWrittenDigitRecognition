# models/cnn.py
from tensorflow.keras import layers, models, regularizers

def build_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    一个在 MNIST 上表现很强的轻量 CNN。
    设计要点：
    - 较小卷积核 + 较深层次
    - BatchNorm + Dropout 防过拟合
    - L2 正则
    """
    weight_decay = 1e-4

    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Block 1
    x = layers.Conv2D(32, (3,3), padding="same",
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, (3,3), padding="same",
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3,3), padding="same",
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (3,3), padding="same",
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    # Head
    x = layers.Flatten()(x)
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="mnist_cnn")
    return model

