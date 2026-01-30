import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ModelKeras(tf.keras.Model):

    def __init__(self):
      super().__init__()
      kwargs = {'kernel_initializer': 'normal', 'bias_initializer': 'normal'}
      self.layer_1 = tf.keras.layers.Dense(512, 'relu', **kwargs)
      self.layer_2 = tf.keras.layers.Dense(512, 'relu', **kwargs)
      self.out_layer = tf.keras.layers.Dense(10, **kwargs)

    @property
    def trainable_vars(self):  # merely to leave the rest of the code unchanged
        return self.trainable_variables

    def call(self, inputs):
      output = self.layer_1(inputs)
      output = self.layer_2(output)
      return self.out_layer(output)

# Define a function to compute gradients of a network's weights w.r.t. a given batch.
def compute_gradients(model, x_batch, y_batch):
    with tf.GradientTape() as tape:
        pred = tf.nn.softmax(model(x_batch))
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, pred)
    return tape.gradient(loss, model.trainable_vars)
# Make a tf.function-decorated copy of the previous.
decorated_gradients = tf.function(compute_gradients)

# Gather a training batch.
x_batch, y_batch = next(iter(mnist_dataset()))
# Instantiate two models and build them.
model_custom = Model()  # weights are built at instantiation
model_keras = ModelKeras()
_ = model_keras(x_batch)  # build weights through sample processing
# Set the second model's weights equal to those of the first one.
weights = [
    w.numpy() for pair in zip(model_custom.trainable_vars[:3], model_custom.trainable_vars[3:])
    for w in pair
]
model_keras.set_weights(weights)

# Compute gradients for both models without tf.function.
# Save for ordering, the results are the same for both, as should be.
compute_gradients(model_custom, x_batch, y_batch)
compute_gradients(model_keras, x_batch, y_batch)

# Compute gradients for both models with tf.function.
# Save for ordering, the results are the same for both, as should be.
# However, they differ from the outputs of the the non-decorated function, which is weird.
decorated_gradients(model_custom, x_batch, y_batch)
decorated_gradients(model_keras, x_batch, y_batch)