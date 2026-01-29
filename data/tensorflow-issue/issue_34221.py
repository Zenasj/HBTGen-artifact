# tf.random.uniform((BATCH_SIZE, 32, 32, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original issue highlights the problem with including softmax activation in output layer
        # and unnormalized inputs. The final working approach is:
        # 1) No softmax in last Dense layer (use linear logits output)
        # 2) Normalize input images to [0,1] range
        #
        # So we implement the model as:
        # Flatten input (32x32x3) -> Dense(10 logits)
        self.flatten = tf.keras.layers.Flatten(input_shape=(32,32,3))
        self.dense = tf.keras.layers.Dense(10)  # No softmax here to avoid numerical instability
    
    def call(self, inputs, training=False):
        # Normalize input images from [0,255] uint8 or float to float32 in [0,1]
        # We assume inputs might be uint8 or float; convert and normalize accordingly
        x = tf.cast(inputs, tf.float32) / 255.0
        x = self.flatten(x)
        logits = self.dense(x)
        return logits

def my_model_function():
    # Instantiate model and compile with from_logits=True loss as recommended
    model = MyModel()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Compile with adam optimizer, sparse categorical crossentropy (from_logits=True), and accuracy metric
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return model

def GetInput():
    # Generate a random batch of inputs with shape (BATCH_SIZE, 32, 32, 3),
    # dtype uint8 in range [0, 255], consistent with CIFAR-10 images.
    # BATCH_SIZE from example is 50.
    BATCH_SIZE = 50
    shape = (BATCH_SIZE, 32, 32, 3)
    # Generate int images, as typical in CIFAR-10 dataset
    inputs = tf.random.uniform(shape, minval=0, maxval=256, dtype=tf.int32)
    inputs = tf.cast(inputs, tf.uint8)
    return inputs

