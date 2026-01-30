import math
import random
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers.experimental import SyncBatchNormalization

def train_dataset_input():
    batch_size = 4
    x_value = tuple(tf.random.uniform((1, batch_size, 10, 10, 3)) for _ in range(5))
    y_value = tf.random.uniform((1, batch_size, 5))
    dataset = tf.data.Dataset.from_tensor_slices((x_value, y_value))
    dataset = dataset.repeat()
    return dataset

class ConvBN(tf.keras.layers.Layer):
    def __init__(self,):
        super(ConvBN, self).__init__()
        self.conv = Conv2D(filters=32, kernel_size=1)
        self.bn1 = SyncBatchNormalization()
        self.bn2 = SyncBatchNormalization()
        self.bn3 = SyncBatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn1(x)
        x = self.bn2(x)
        x = self.bn3(x)
        return x

class MultiFrameModel(keras.Model):
    def __init__(self, **kwargs):
        super(MultiFrameModel, self).__init__(name="multi_frame_model", **kwargs)
        self.backbone = keras.Sequential([ConvBN() for _ in range(17)])
        self.dense = Dense(5)

    def call(self, samples, training=False):
        backbone_outputs = []
        assert len(samples) == 5
        for img in samples:
            backbone_outputs.append(self.backbone(img))

        concat_sample = tf.concat(values=backbone_outputs, axis=3)
        x = tf.reduce_sum(concat_sample, [2, 3])
        x = self.dense(x)
        return x

class AucFromLogits(AUC):
    def update_state(self, y_true, logits, sample_weight=None):
        y_pred = tf.math.sigmoid(logits)
        super(AucFromLogits, self).update_state(y_true, y_pred, sample_weight)

def main():
    strategy = tf.distribute.MirroredStrategy()
    assert strategy.num_replicas_in_sync == 4

    train_dataset = train_dataset_input()
    with strategy.scope():
        metrics = [
            AucFromLogits(
                name="auc_from_logits",
                num_thresholds=100,
                curve="PR",
                multi_label=True,
                label_weights=[1.0, 0.0, 0.0, 0.0, 0.0],
            )
        ]

        model = MultiFrameModel()
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.MeanSquaredError(),
            metrics=metrics,
        )

    model.fit(
        x=train_dataset,
        validation_data=train_dataset,
        steps_per_epoch=100,  # number of training steps between eval epochs
        epochs=120,  # epochs = total number of training steps / steps_per_epoch
        validation_steps=100,
        validation_freq=1,
        verbose=2,
    )

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()