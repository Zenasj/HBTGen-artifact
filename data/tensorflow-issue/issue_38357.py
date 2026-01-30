from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)
inputs = keras.Input(shape=(784,), name='digits')
num_units = 4096
dense1 = layers.Dense(num_units, activation='relu', name='dense_1')
x = dense1(inputs)
dense2 = layers.Dense(1, activation='relu', name='dense_2')
outputs = dense2(x)

model = keras.Model(inputs=inputs, outputs=outputs)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255

optimizer = keras.optimizers.RMSprop()
optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
loss_object = tf.keras.losses.MeanSquaredError()
train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                 .shuffle(10000).batch(1024))

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_object(y, predictions) * 10000.
        scaled_loss = optimizer.get_scaled_loss(loss)
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

for epoch in range(2):
    for i, (x, y) in enumerate(train_dataset):
        mp_loss_scale = optimizer.loss_scale().numpy()
        loss = train_step(x, y)
        print('epoch {}: step {}: loss={}, loss_scale={}'.format(epoch, i, loss, mp_loss_scale))

class LossScaleBelowOneOptimizer(tf.keras.mixed_precision.LossScaleOptimizer):

  MULTIPLIER = 2 ** 10

  @property
  def actual_loss_scale(self):
    return self.loss_scale / self.MULTIPLIER

  def get_scaled_loss(self, loss):
    if callable(loss):
      def new_loss():
        loss_val = loss()
        return loss_val * tf.cast(self.actual_loss_scale, loss_val.dtype)
      return new_loss
    else:
      return loss * tf.cast(self.actual_loss_scale, loss.dtype)

  def get_unscaled_gradients(self, grads):
    reciprocal = 1. / self.actual_loss_scale
    return [g * reciprocal if g is not None else None for g in grads]