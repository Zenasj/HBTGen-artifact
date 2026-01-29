# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← Input shape inferred from CIFAR-10 dataset (batch size B, height 32, width 32, 3 channels)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10, use_sparse_label=False):
        super().__init__()
        # Store config
        self.num_classes = num_classes
        self.use_sparse_label = use_sparse_label

        # Build convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation='relu',
            input_shape=(32, 32, 3)
        )
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation='relu'
        )
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = tf.keras.layers.Dropout(0.20)

        self.conv3 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation='relu'
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout2 = tf.keras.layers.Dropout(0.20)

        # Fully connected layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(300, activation="relu")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(0.20)
        self.dense2 = tf.keras.layers.Dense(200, activation="relu")
        # Output layer - softmax activation for classification probabilities
        self.outputs_layer = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.dropout1(x, training=training)

        x = self.conv3(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        x = self.dropout2(x, training=training)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn3(x, training=training)
        x = self.dropout3(x, training=training)
        x = self.dense2(x)
        outputs = self.outputs_layer(x)
        return outputs

def my_model_function():
    """
    Creates and compiles an instance of MyModel,
    with configuration matching the original setup:
      - num_classes=10 (CIFAR-10)
      - use_sparse_label controls the loss type and metrics
    Compilation uses Adam optimizer with lr=1e-3,
    and the appropriate loss and accuracy metric.
    """
    use_sparse_label = True  # Change here to False to use one-hot labels and CategoricalCrossentropy

    model = MyModel(num_classes=10, use_sparse_label=use_sparse_label)

    learning_rate = 1e-3
    # Choose loss function and accuracy metric based on label format
    if use_sparse_label:
        # Sparse labels: use SparseCategoricalCrossentropy and SparseCategoricalAccuracy explicitly
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    else:
        # One-hot labels: use CategoricalCrossentropy and default accuracy
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        accuracy_metric = "accuracy"  # string identifier works fine here

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
        loss=loss_fn,
        metrics=[accuracy_metric]
    )

    # Model.build to set input shape for summary, optional but helpful
    model.build((None, 32, 32, 3))
    model.summary()
    return model

def GetInput():
    """
    Returns a random batch of input images to the model, shaped (batch_size, 32, 32, 3),
    with float32 type normalized to [0,1].
    Batch size is 16 as used in the original training code.
    """
    import numpy as np
    batch_size = 16
    input_shape = (batch_size, 32, 32, 3)
    # Generate random float32 tensor to simulate CIFAR10-like image inputs scaled [0,1]
    random_input = tf.random.uniform(input_shape, minval=0., maxval=1., dtype=tf.float32)
    return random_input

