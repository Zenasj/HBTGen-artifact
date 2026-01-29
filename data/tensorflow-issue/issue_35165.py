# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32) ← inferred input shape from dataset creation (batch, height, width, channels)
import tensorflow as tf

class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3))
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs, training=False):
        # inputs is a tuple (images, targets)
        images, targets = inputs
        x = self.conv1(images)
        x = self.conv2(x)
        x = self.flatten(x)
        # Compute loss as difference of means along feature axis
        loss = tf.reduce_mean(x, axis=1) - tf.reduce_mean(targets, axis=1)
        return loss

def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    """
    Return a tuple of two tensors matching the expected input signature:
    - images: tensor of shape (batch_size, 224, 224, 3), float32
    - targets: tensor of shape (batch_size, 1000), float32
    The batch size is chosen as 32 matching dataset batch size in the issue.
    Values are zeros to mimic original example.
    """
    batch_size = 32
    images = tf.zeros([batch_size, 224, 224, 3], dtype=tf.float32)
    targets = tf.zeros([batch_size, 1000], dtype=tf.float32)
    return (images, targets)


@tf.function
def train_one_step(model, optimizer, inputs):
    with tf.GradientTape() as tape:
        loss = model(inputs, training=True)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


@tf.function
def in_graph_training_loop(model, optimizer, dataset):
    # Using iter(dataset) as recommended in the issue discussion to ensure dataset iterator runs efficiently in graph
    step = 0
    for inputs in iter(dataset):
        loss = train_one_step(model, optimizer, inputs)
        step += 1


def out_graph_training_loop(model, optimizer, dataset):
    # Out-of-graph training loop runs outside tf.function for speed comparison
    step = 0
    for inputs in dataset:
        loss = train_one_step(model, optimizer, inputs)
        step += 1


def create_dataset():
    # Recreate dataset from numpy zeros as in issue
    import numpy as np
    X = np.zeros([10, 224, 224, 3], dtype=np.float32)
    Y = np.zeros([10, 1000], dtype=np.float32)
    x_ds = tf.data.Dataset.from_tensor_slices(X)
    y_ds = tf.data.Dataset.from_tensor_slices(Y)
    ds = tf.data.Dataset.zip((x_ds, y_ds))
    ds = ds.batch(32)
    return ds

# Note: The above code reproduces the structure from the issue,
# modeling the performance difference and typical usage pattern.
# The MyModel call uses inputs tuple, computing a "loss" tensor,
# which training steps reduce and backpropagate on.
# The issues around in-graph vs out-of-graph performance relate
# to usage of tf.function, iter(dataset), and loop placement,
# which are all documented in accompanying comments.


# The code here is compatible with TF 2.20.0 XLA compilation:
# Usage example:
# model = my_model_function()
# optimizer = tf.keras.optimizers.Adam(1e-4)
# dataset = create_dataset()
#
# @tf.function(jit_compile=True)
# def compiled(x):
#     return model(x)

