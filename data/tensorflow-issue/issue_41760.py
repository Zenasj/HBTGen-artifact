import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dense1 = tf.keras.layers.Dense(3)
dense2 = tf.keras.layers.Dense(3)

@tf.function
def fun(dense, input):
  return dense(input)
  
fun(dense1, input)
fun(dense2, input)

@tf.function
def grad(model, inputs, targets, optimizer):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  grads = tape.gradient(loss_value, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

class MyModel(keras.Model):
    ...
    @tf.function
    def train_step(self, inputs, targets):
      with tf.GradientTape() as tape:
        loss_value = self.loss(inputs, targets, training=True)
      grads = tape.gradient(loss_value, model.trainable_variables)
      self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    
    
model = MyModel(...)
model.compile(loss = loss, optimizer=optimizer)
model.train_step(...)

model2 = MyModel(...)
model2.compile(loss = loss, optimizer=optimizer)
model2.train_step(...)