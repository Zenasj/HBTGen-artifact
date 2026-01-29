# tf.random.uniform((B, 10, 10, 3), dtype=tf.float32) for each element in tuple of length 5, B=4 (batch size)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow.keras.metrics import AUC

class ConvBN(tf.keras.layers.Layer):
    def __init__(self):
        super(ConvBN, self).__init__()
        self.conv = Conv2D(filters=32, kernel_size=1)
        # SyncBatchNormalization applied 3 times as in original model
        self.bn1 = SyncBatchNormalization()
        self.bn2 = SyncBatchNormalization()
        self.bn3 = SyncBatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn1(x, training=training)
        x = self.bn2(x, training=training)
        x = self.bn3(x, training=training)
        return x

class MultiFrameModel(keras.Model):
    def __init__(self, **kwargs):
        super(MultiFrameModel, self).__init__(name="multi_frame_model", **kwargs)
        # Backbone: 17 ConvBN blocks applied sequentially
        self.backbone = keras.Sequential([ConvBN() for _ in range(17)])
        self.dense = Dense(5)

    def call(self, samples, training=False):
        # samples is a tuple/list of 5 tensors of shape (B, 10, 10, 3)
        backbone_outputs = []
        assert len(samples) == 5
        for img in samples:
            x = self.backbone(img, training=training)
            backbone_outputs.append(x)

        # Concatenate along channels axis (axis=3)
        concat_sample = tf.concat(backbone_outputs, axis=3)
        # Reduce sum over spatial dims (axis 2,3)
        x = tf.reduce_sum(concat_sample, axis=[2,3])
        x = self.dense(x)
        return x

class AucFromLogits(AUC):
    def update_state(self, y_true, logits, sample_weight=None):
        y_pred = tf.math.sigmoid(logits)
        super(AucFromLogits, self).update_state(y_true, y_pred, sample_weight)

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.model = MultiFrameModel()

    def call(self, inputs, training=False):
        # inputs: tuple of 5 tensors (B,10,10,3) each
        return self.model(inputs, training=training)

def my_model_function():
    # Return an instance of MyModel with the same structure
    return MyModel()

def GetInput():
    # Number of replica batch size, choose 4 consistent with multi-GPU example.
    batch_size = 4
    # Generate tuple of 5 tensors each shape (B, 10, 10, 3), default float32 values in [0,1)
    inputs = tuple(tf.random.uniform((batch_size, 10, 10, 3), dtype=tf.float32) for _ in range(5))
    return inputs

