"""Kiến trúc: CNN baseline, EfficientNetB0 transfer learning, Grad-CAM."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

L = tf.keras.layers
INPUT_SHAPE = (224, 224, 3)
BACKBONE_NAME = "efficientnetb0"
GRADCAM_LAYER_NAMES = (BACKBONE_NAME, "last_conv_act")  # baseline dùng "last_conv_act"


def compile_model(model: tf.keras.Model, lr: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_baseline_cnn(input_shape=INPUT_SHAPE) -> tf.keras.Model:
    """CNN nhỏ tự xây (3 khối Conv) huấn luyện từ đầu, dùng làm mốc so sánh."""
    inputs = L.Input(input_shape)
    x = L.Rescaling(1.0 / 255)(inputs)
    for i, filters in enumerate((32, 64, 128), start=1):
        x = L.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = L.BatchNormalization()(x)
        x = L.Activation("relu", name="last_conv_act" if i == 3 else f"act{i}")(x)
        if i < 3:
            x = L.MaxPooling2D()(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dense(64, activation="relu")(x)
    x = L.Dropout(0.3)(x)
    outputs = L.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="baseline_cnn")


def build_transfer_model(input_shape=INPUT_SHAPE, weights: str | None = "imagenet") -> tf.keras.Model:
    """EfficientNetB0 (ImageNet) + đầu phân loại mới. Giai đoạn 1: đóng băng backbone.

    Backbone gọi với training=False để các tầng BatchNorm luôn ở chế độ suy luận
    (kể cả khi fine-tune) — tránh phá thống kê ImageNet với batch nhỏ.
    """
    base = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape, include_top=False, weights=weights
    )
    base.trainable = False
    inputs = L.Input(input_shape)
    x = base(inputs, training=False)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.3)(x)
    outputs = L.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="efficientnetb0_pneumonia")


def unfreeze_top_layers(model: tf.keras.Model, n_layers: int = 30) -> tf.keras.Model:
    """Giai đoạn 2: mở khoá n tầng cuối của backbone (giữ nguyên BatchNorm đóng băng).

    Sau khi gọi phải compile lại với learning rate nhỏ (1e-5).
    """
    base = model.get_layer(BACKBONE_NAME)
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers[-n_layers:]:
        layer.trainable = not isinstance(layer, L.BatchNormalization)
    return model


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
class GradCAM:
    """Grad-CAM cho model nhị phân sigmoid: vùng ảnh làm tăng điểm 'PNEUMONIA'.

    Gradient lấy theo logit (trước sigmoid) để không bị triệt tiêu khi xác suất
    bão hoà sát 0 hoặc 1.
    """

    def __init__(self, model: tf.keras.Model):
        # Dựng lại đồ thị bằng cách áp dụng tuần tự các tầng lên Input mới: cách này
        # chạy đúng cả với model vừa nạp từ .keras (backbone lồng nhau).
        inputs = tf.keras.Input(model.input_shape[1:])
        x, conv = inputs, None
        for layer in model.layers[1:-1]:  # bỏ InputLayer và đầu Dense(1, sigmoid)
            x = layer(x)
            if layer.name in GRADCAM_LAYER_NAMES:
                conv = x
        if conv is None:
            raise ValueError("Không tìm thấy tầng conv cuối cho Grad-CAM.")
        kernel, bias = model.layers[-1].get_weights()
        self._kernel = tf.constant(kernel)
        self._bias = tf.constant(bias)
        self._grad_model = tf.keras.Model(inputs, [conv, x])  # x = đặc trưng đưa vào đầu phân loại

    def heatmap(self, image: np.ndarray) -> np.ndarray:
        """image: (H, W, 3) float32 thang 0-255 -> heatmap (h, w) trong [0, 1]."""
        x = tf.convert_to_tensor(image[None], tf.float32)
        with tf.GradientTape() as tape:
            conv, feats = self._grad_model(x, training=False)
            logit = tf.matmul(feats, self._kernel)[:, 0] + self._bias[0]
        grads = tape.gradient(logit, conv)
        weights = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.reduce_sum(conv * weights[:, None, None, :], axis=-1)[0]
        cam = tf.nn.relu(cam)
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        return cam.numpy()
