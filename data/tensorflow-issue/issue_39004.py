from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

tf.keras.backend.set_floatx('float32')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.datasets import load_linnerud
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Concatenate

X, y = load_linnerud(return_X_y=True)

data = tf.data.Dataset.from_tensor_slices((X, y)).\
    map(lambda a, b: (tf.divide(a, tf.reduce_max(X, axis=0, keepdims=True)), b))

train_data = data.take(16).shuffle(16).batch(4)
test_data = data.skip(16).shuffle(4).batch(4)


class FullyConnectedNetwork(Model):
    def __init__(self):
        super(FullyConnectedNetwork, self).__init__()
        self.layer1 = Dense(9, input_shape=(3,))
        self.layer2 = LSTM(8, return_sequences=True)
        self.layer3 = Dense(27)
        self.layer4 = Dropout(5e-1)
        self.layer5 = Dense(27)
        self.layer6 = Concatenate()
        self.layer7 = Dense(3)

    def __call__(self, x, *args, **kwargs):
        x = tf.nn.tanh(self.layer1(x))
        y = self.layer2(x)
        x = tf.nn.selu(self.layer3(x))
        x = self.layer4(x)
        x = tf.nn.relu(self.layer5(x))
        x = self.layer6([x, y])
        x = self.layer7(x)
        return x


model = FullyConnectedNetwork()

loss_object = tf.keras.losses.Huber()

train_loss = tf.keras.metrics.Mean()
test_loss = tf.keras.metrics.Mean()

optimizer = tf.keras.optimizers.Adamax()


@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        loss = loss_object(outputs, targets)
        train_loss(loss)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


@tf.function
def test_step(inputs, targets):
    outputs = model(inputs)
    print(outputs.dtype, targets.dtype)
    loss = loss_object(outputs, targets)
    test_loss(loss)


def main():
    train_loss.reset_states()
    test_loss.reset_states()

    for epoch in range(1, 10_000 + 1):
        for x, y in train_data:
            train_step(x, y)

        for x, y in test_data:
            test_step(x, y)

        if epoch % 25 == 0:
            print(f'Epoch: {epoch:>4} Train Loss: {train_loss.result().numpy():.2f} '
                  f'Test Loss: {test_loss.result().numpy():.2f}')


if __name__ == '__main__':
    main()