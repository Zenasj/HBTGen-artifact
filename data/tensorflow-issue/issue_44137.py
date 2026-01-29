# tf.random.uniform((32, 32, 32, 3), dtype=tf.float32) ← CIFAR10 input shape (batch_size, height, width, channels)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=1, padding='same',
            activation='relu', input_shape=(32, 32, 3))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D()

        self.conv2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=1, padding='same',
            activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPool2D()

        self.flatten = tf.keras.layers.Flatten()
        # CIFAR-10 has 10 classes
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)
        # Normalize input to [0,1]
        # Note: In the original code, this was done by ImageDataGenerator rescale=1./255,
        # so here we assume input is float in [0,1].
        # If input is uint8, user should normalize before feeding.
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel, compile with optimizer, loss, metrics
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',  # Using sparse since y labels for CIFAR10 are integers (0-9)
        metrics=['accuracy'])
    return model

def color_augmentation(image):
    # Implements random hue augmentation compatible with XLA and multiprocessing
    # We use stateless random hue with a fixed seed for reproducibility per image.
    # This is adapted to avoid stateful random ops within multiprocessing.
    # Since inputs shape is [H,W,C], we need a seed. Here, a fixed seed is used for demonstration.
    seed = (123, 37)  # Some arbitrary fixed seed for stateless random ops
    # tf.image.stateless_random_hue expects inputs in [0,1], so expect float inputs normalized
    return tf.image.stateless_random_hue(image, max_delta=0.1, seed=seed)

@tf.function(jit_compile=True)
def preprocessing_fn(images):
    # Apply random hue augmentation batch-wise
    # images: batch of images, shape [B,H,W,C], float32 in [0,1]
    # We map color_augmentation along batch dimension
    return tf.map_fn(color_augmentation, images, fn_output_signature=tf.float32)

def GetInput():
    # Return a batch of random CIFAR10-like images for input
    # Shape (batch_size, 32, 32, 3), dtype float32, values in [0,1]
    batch_size = 32
    images = tf.random.uniform(
        shape=(batch_size, 32, 32, 3),
        minval=0.0, maxval=1.0,
        dtype=tf.float32)
    # Optionally apply preprocessing (simulate random_hue augmentation)
    images_aug = preprocessing_fn(images)
    return images_aug

