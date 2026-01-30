from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf

save_model = True

learning_rate = 1e-4
BATCH_SIZE = 100
TEST_BATCH_SIZE = 10
color_channels = 1
imsize = 28

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images[:5000, ::]
test_images = train_images[:1000, ::]
train_images = train_images.reshape(-1, imsize, imsize, 1).astype('float32')
test_images = test_images.reshape(-1, imsize, imsize, 1).astype('float32')
train_images /= 255.
test_images /= 255.
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(TEST_BATCH_SIZE)

class AE(tf.keras.Model):
    def __init__(self):
        super(AE, self).__init__()
        self.network = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(imsize, imsize, color_channels)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(50),
            tf.keras.layers.Dense(imsize**2 * color_channels),
            tf.keras.layers.Reshape(target_shape=(imsize, imsize, color_channels)),
        ])
    def decode(self, input):
        logits = self.network(input)
        return logits

optimizer = tf.keras.optimizers.Adam(learning_rate)
model = AE()

def compute_loss(data):
    logits = model.decode(data)
    loss = tf.reduce_mean(tf.losses.mean_squared_error(logits, data))
    return loss

def train_step(data):
    with tf.GradientTape() as tape:
        loss = compute_loss(data)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, 0

def test_step(data):
    loss = compute_loss(data)
    return loss

input_shape_set = False
epoch = 0
epochs = 20
for epoch in range(epochs):
    for train_x in train_dataset:
        train_step(train_x)
    if epoch % 1 == 0:
        loss = 0.0
        num_batches = 0
        for test_x in test_dataset:
            loss += test_step(test_x)
            num_batches += 1
        loss /= num_batches
        print("Epoch: {}, Loss: {}".format(epoch, loss))

        if save_model:
            print("Saving model...")
            if not input_shape_set:
                # Note: Why set input shape manually and why here:
                # 1. If I do not set input shape manually: ValueError: Model <main.CVAE object at 0x7f1cac2e7c50> cannot be saved because the input shapes have not been set. Usually, input shapes are automatically determined from calling .fit() or .predict(). To manually set the shapes, call model._set_inputs(inputs).
                # 2. If I set input shape manually BEFORE the first actual train step, I get: RuntimeError: Attempting to capture an EagerTensor without building a function.
                model._set_inputs(train_dataset.__iter__().next())
                input_shape_set = True
            # Note: Why choose tf format: model.save('MNIST/Models/model.h5') will return NotImplementedError: Saving the model to HDF5 format requires the model to be a Functional model or a Sequential model. It does not work for subclassed models, because such models are defined via the body of a Python method, which isn't safely serializable. Consider saving to the Tensorflow SavedModel format (by setting save_format="tf") or using save_weights.
            model.save('MNIST/Models/model', save_format='tf')