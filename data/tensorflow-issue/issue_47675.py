import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

class DummyExample(tf.keras.Model):
    def __init__(
        self,
        output_dim: int,
        decay_rate: float
    ):
        super(DummyExample, self).__init__()
        self.output_dim = output_dim
        self.decay = decay_rate
        self.encoder_1 = self.create_encoder()
        self.encoder_2 = self.create_encoder()

    def create_encoder(self):
        encoder = tf.keras.Sequential()
        encoder.add(tf.keras.Input(shape=(4,), dtype=tf.float32))
        encoder.add(tf.keras.layers.Dense(
            units=self.output_dim,
            activation='relu'))
        return encoder

    def call(self, x: tf.Tensor, training: bool = False):
        return self.encoder_1(x, training=training)
    
    def train_step(self, data):
        x = data
        out_2 = self.encoder_2(x, training=True)
        with tf.GradientTape() as tape:
            out_1 = self.encoder_1(x, training=True)
            loss = self.compiled_loss(out_1, out_2)
        enc_1_grads = tape.gradient(loss, self.encoder_1.trainable_weights)
        self.optimizer.apply_gradients(zip(enc_1_grads, self.encoder_1.trainable_weights))
        
        # update encoder_2
        enc_1_weights = self.encoder_1.weights
        enc_2_weights = self.encoder_2.weights
        for i in range(len(enc_1_weights)):
            enc_2_weights[i] = self.decay * enc_2_weights[i] + (1-self.decay) * enc_1_weights[i]
        self.encoder_2.set_weights(enc_2_weights)
        return {'loss': loss}

dummy_model = DummyExample(2, 0.99)
dummy_model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='mse')

x = np.random.normal(size=(100, 4))
ds = tf.data.Dataset.from_tensor_slices(x)
ds = ds.batch(10)

dummy_model.fit(ds, epochs=2)