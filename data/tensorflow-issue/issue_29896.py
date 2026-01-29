# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf

# This model is inspired by the example in the TensorFlow TPU Fashion MNIST notebook,
# illustrating transfer learning with a pretrained Keras model and subsequent fine tuning
# on TPU. Since no full model code was provided, we reconstruct a simple CNN for Fashion MNIST.
# We also incorporate the key lesson from the issue:
#  - When loading weights for TPU models, always call model.compile() before model.load_weights()
#  - TPU variable creation and strategy.scope() usage are critical for proper TPU execution

# Assumptions:
# Input shape: (batch_size, 28, 28, 1) from Fashion MNIST grayscale images
# Simple CNN architecture: Conv2D -> BatchNorm -> Activation -> Pooling -> Flatten -> Dense classifier
# The model demonstrates loading weights after compiling inside TPU strategy.scope()
# We'll encapsulate two phases as submodels:
#   pretrain_model: initial training model with normal compile -> fit
#   finetune_model: fine tuning with lower lr following proper pattern
# forward pass of MyModel returns a tuple of predictions after each phase for comparison

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Base feature extractor (simple CNN)
        # Normally would load pretrained weights here after compile
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation=None, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.ReLU()
        self.pool1 = tf.keras.layers.MaxPooling2D()

        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation=None, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.ReLU()
        self.pool2 = tf.keras.layers.MaxPooling2D()

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(128, activation='relu')

        # Classifier layers (simulating different training phases)
        self.classifier_pretrain = tf.keras.layers.Dense(10, activation='softmax')
        self.classifier_finetune = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        # Feature extractor
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        features = self.dense(x)

        # Two "submodels" representing pretrain and finetune outputs:
        out_pretrain = self.classifier_pretrain(features)
        out_finetune = self.classifier_finetune(features)

        # The model returns both outputs for potential comparison or analysis
        return out_pretrain, out_finetune

def my_model_function():
    """
    Create a MyModel instance and illustrate a typical TPU workflow:
    - use strategy.scope()
    - compile before loading weights (simulated)
    For demonstration here, weights are not loaded explicitly,
    but caller can load weights after compiling as per TPU best practice.
    """
    # TPU strategy simulation:
    # In actual TPU usage on Colab, you'd do something like:
    # resolver = tf.contrib.cluster_resolver.TPUClusterResolver(...)
    # tf.contrib.distribute.initialize_tpu_system(resolver)
    # strategy = tf.contrib.distribute.TPUStrategy(resolver)
    #
    # with strategy.scope():
    #     model = MyModel()
    #     model.compile(...)
    #     model.load_weights(...)  <== important to load weights after compile on TPU
    # return model

    model = MyModel()
    # Compile with standard loss and optimizer:
    # We use sparse_categorical_crossentropy and accuracy metrics as typical for Fashion MNIST
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    # Here weights loading should happen after compile if weights exist
    return model

def GetInput():
    """
    Return a random tensor input matching the expected input of MyModel
    Shape: (batch_size, 28, 28, 1) where batch_size is arbitrarily 32
    Data type: tf.float32 matching typical image float inputs normalized between 0 and 1
    """
    batch_size = 32
    height = 28
    width = 28
    channels = 1
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

