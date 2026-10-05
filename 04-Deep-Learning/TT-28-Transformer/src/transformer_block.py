"""Transformer Encoder tự xây bằng Keras: Positional Encoding + Multi-Head Attention + FFN/Residual/LayerNorm."""
import math

import keras
import numpy as np
from keras import layers, ops


class PositionalEncoding(layers.Layer):
    """① Positional Encoding dạng sin/cos: CỘNG thông tin vị trí vào embedding."""

    def __init__(self, max_len, d_model, **kw):
        super().__init__(**kw)
        pos = np.arange(max_len)[:, None]
        i = np.arange(d_model)[None, :]
        angle = pos / np.power(10000.0, (2 * (i // 2)) / d_model)
        pe = np.zeros((max_len, d_model), dtype="float32")
        pe[:, 0::2] = np.sin(angle[:, 0::2])
        pe[:, 1::2] = np.cos(angle[:, 1::2])
        self.pe = ops.convert_to_tensor(pe[None])

    def call(self, x):
        return x + self.pe[:, : x.shape[1], :]


class KhoiTransformer(layers.Layer):
    """Một khối encoder: ② Multi-Head Attention -> ③ Residual+LayerNorm -> FFN -> Residual+LayerNorm.

    key_dim = d_model // n_heads (chuẩn "Attention Is All You Need") để tổng số tham số
    KHÔNG đổi khi thay số head -> thí nghiệm khảo sát số head công bằng.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kw):
        super().__init__(**kw)
        self.att = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)
        self.ffn = keras.Sequential([layers.Dense(d_ff, activation="relu"), layers.Dense(d_model)])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.do1 = layers.Dropout(dropout)
        self.do2 = layers.Dropout(dropout)

    def call(self, x, pad_mask=None, training=False, return_scores=False):
        # pad_mask: (batch, seq) True = token thật; chặn attention vào vị trí padding
        att_mask = None if pad_mask is None else pad_mask[:, None, :]
        if return_scores:
            a, scores = self.att(x, x, attention_mask=att_mask, return_attention_scores=True)
        else:
            a, scores = self.att(x, x, attention_mask=att_mask), None
        a = self.do1(a, training=training)
        x = self.ln1(x + a)                       # ⭐ residual + norm
        f = self.do2(self.ffn(x), training=training)
        x = self.ln2(x + f)
        return (x, scores) if return_scores else x


class TransformerClassifier(keras.Model):
    """Embedding + (Positional Encoding) + N khối Transformer + masked GlobalAvgPool + Dense."""

    def __init__(self, vocab_size, max_len, d_model, n_heads, d_ff, n_blocks, dropout=0.1,
                 n_classes=4, use_pos=True, **kw):
        super().__init__(**kw)
        self.cfg = dict(vocab_size=vocab_size, max_len=max_len, d_model=d_model, n_heads=n_heads,
                        d_ff=d_ff, n_blocks=n_blocks, dropout=dropout, n_classes=n_classes,
                        use_pos=use_pos)
        self.scale = math.sqrt(d_model)
        self.emb = layers.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(max_len, d_model) if use_pos else None
        self.drop_in = layers.Dropout(dropout)
        self.blocks = [KhoiTransformer(d_model, n_heads, d_ff, dropout) for _ in range(n_blocks)]
        self.drop_out = layers.Dropout(dropout)
        self.out = layers.Dense(n_classes)        # trả về logits

    def call(self, ids, training=False, return_attention=False):
        pad_mask = ops.not_equal(ids, 0)           # id 0 = padding
        x = self.emb(ids) * self.scale
        if self.pos is not None:
            x = self.pos(x)
        x = self.drop_in(x, training=training)
        all_scores = []
        for blk in self.blocks:
            if return_attention:
                x, s = blk(x, pad_mask=pad_mask, training=training, return_scores=True)
                all_scores.append(s)
            else:
                x = blk(x, pad_mask=pad_mask, training=training)
        m = ops.cast(pad_mask, x.dtype)[..., None]  # trung bình chỉ trên token thật
        pooled = ops.sum(x * m, axis=1) / ops.maximum(ops.sum(m, axis=1), 1.0)
        logits = self.out(self.drop_out(pooled, training=training))
        return (logits, all_scores) if return_attention else logits
