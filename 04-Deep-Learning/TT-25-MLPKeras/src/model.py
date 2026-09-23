"""
model.py
--------
Các kiến trúc MLP (Keras/TensorFlow) cho bài toán chấm điểm khách hàng
tiềm năng mua bảo hiểm ô tô.

Cung cấp 3 kiểu xây dựng model để phục vụ các bước so sánh trong README:
  - build_mlp:            MLP cơ bản, tuỳ chọn bật/tắt Dropout + BatchNorm
  - build_mlp_embedding:  MLP dùng Embedding cho Region_Code, Policy_Sales_Channel
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def build_mlp(n_features: int, hidden_units=(128, 64), use_dropout_bn: bool = True,
              dropout_rates=None, learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    MLP cơ bản cho input dạng dense (one-hot + numeric).

    hidden_units: tuple số neuron mỗi lớp ẩn, ví dụ (64,), (128,64), (256,128,64)
    use_dropout_bn: bật/tắt BatchNorm + Dropout (mục 5 vs 6 trong README)
    dropout_rates: list tỉ lệ dropout theo từng lớp ẩn; mặc định giảm dần 0.3 -> 0.2 -> 0.1
    """
    if dropout_rates is None:
        base_rates = [0.3, 0.2, 0.1]
        dropout_rates = (base_rates + [0.1] * len(hidden_units))[: len(hidden_units)]

    inputs = layers.Input(shape=(n_features,), name="dense_input")
    x = inputs
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation="relu")(x)
        if use_dropout_bn:
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rates[i])(x)
    outputs = layers.Dense(1, activation="sigmoid", name="response")(x)

    model = models.Model(inputs, outputs, name=f"mlp_{'bn_dropout' if use_dropout_bn else 'plain'}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )
    return model


def build_mlp_embedding(dense_dim: int, vocab_sizes: dict, hidden_units=(128, 64),
                         embedding_dim_rule="auto", dropout_rates=None,
                         learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    MLP với Embedding cho các biến phân loại nhiều mức (Region_Code, Policy_Sales_Channel),
    thay vì one-hot 200+ cột (mục 10 trong README).

    vocab_sizes: dict {"Region_Code": n, "Policy_Sales_Channel": m} (đã +1 cho bucket unknown)
    embedding_dim_rule: "auto" -> dim = min(50, round(vocab_size ** 0.25 * 4)) (kinh nghiệm phổ biến)
    """
    if dropout_rates is None:
        base_rates = [0.3, 0.2, 0.1]
        dropout_rates = (base_rates + [0.1] * len(hidden_units))[: len(hidden_units)]

    dense_input = layers.Input(shape=(dense_dim,), name="dense")
    embed_branches = []
    cat_inputs = {}

    for col, vocab_size in vocab_sizes.items():
        if embedding_dim_rule == "auto":
            emb_dim = min(50, max(4, round((vocab_size ** 0.25) * 4)))
        else:
            emb_dim = embedding_dim_rule
        cat_in = layers.Input(shape=(1,), name=col, dtype="int32")
        emb = layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, name=f"emb_{col}")(cat_in)
        emb = layers.Flatten()(emb)
        cat_inputs[col] = cat_in
        embed_branches.append(emb)

    x = layers.Concatenate()([dense_input] + embed_branches)
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rates[i])(x)
    outputs = layers.Dense(1, activation="sigmoid", name="response")(x)

    inputs = {"dense": dense_input, **cat_inputs}
    model = models.Model(inputs, outputs, name="mlp_embedding")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )
    return model


def default_callbacks(checkpoint_path: str = "models/best.keras", patience_es: int = 10,
                       patience_lr: int = 5):
    """3 callback bắt buộc theo README: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max", patience=patience_es, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=patience_lr
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_pr_auc", mode="max", save_best_only=True
        ),
    ]
