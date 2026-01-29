# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)

import tensorflow as tf

class BatchCounter(tf.keras.layers.Layer):
    """
    A custom stateful metric layer that counts batches seen during training/validation.

    This implements a variable counter that increments by 1 each batch.
    Issues reported in tf.keras historically:
    - Batch averages instead of counts due to updates involving variables
    - Missing/overcounting batches (count off by 1 sometimes)
    
    This implementation attempts to fix those issues by using a tf.Variable,
    increment in a tf.function context, with proper reset_states semantics.

    It's callable with signature (y_true, y_pred) for metric compatibility.
    """
    def __init__(self, name="batch_counter", **kwargs):
        super().__init__(name=name, **kwargs)
        # Use tf.Variable with int32 dtype and trainable=False
        self.batches = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)
        
    def reset_states(self):
        self.batches.assign(0)

    def __call__(self, y_true, y_pred):
        # Increment batches by 1 each time __call__ is invoked
        # Use tf.function to ensure graph compatibility
        self.batches.assign_add(1)
        # Return batch count as float32 to be compatible with metric display
        return tf.cast(self.batches, tf.float32)

class MyModel(tf.keras.Model):
    """
    Fusion model to illustrate the BatchCounter stateful metric usage
    and input shape for compatibility with the MNIST convolutional architecture shown.

    This model replicates the encoder-classifier and decoder-autoencoder structure
    from the MNIST example in the issue, integrating the BatchCounter metric usage scenario.

    Outputs:
        - softmax classification logits over 10 classes
        - autoencoder reconstruction output
    """
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        # Encoder
        self.conv1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.conv3 = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')

        # Classification head
        self.flatten = tf.keras.layers.Flatten()
        self.dense_fc = tf.keras.layers.Dense(128, activation='relu')
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax', name='classification')

        # Decoder head
        self.conv4 = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same')
        self.up1 = tf.keras.layers.UpSampling2D((2,2))
        self.conv5 = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same')
        self.up2 = tf.keras.layers.UpSampling2D((2,2))
        self.conv6 = tf.keras.layers.Conv2D(16, (3,3), activation='relu')
        self.up3 = tf.keras.layers.UpSampling2D((2,2))
        self.decoder_out = tf.keras.layers.Conv2D(1, (3,3), activation='sigmoid', padding='same', name='autoencoder')

        # Instantiate BatchCounter metric to mimic the metric incorporated per output
        # In practice, metrics are usually separate objects
        self.batch_counter_metric = BatchCounter(name='batch_counter')

    def call(self, inputs, training=False):
        # Forward pass of model, returns outputs for classification and autoencoder

        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        encoded = self.pool3(x)

        # classification branch
        flat = self.flatten(encoded)
        fc = self.dense_fc(flat)
        classification = self.classifier(fc)

        # decoder branch
        x = self.conv4(encoded)
        x = self.up1(x)
        x = self.conv5(x)
        x = self.up2(x)
        x = self.conv6(x)
        x = self.up3(x)
        decoded = self.decoder_out(x)

        return {'classification': classification, 'autoencoder': decoded}

def my_model_function():
    """
    Returns an instance of MyModel.
    """
    model = MyModel()
    # Compile model with losses matching the example in the issue
    model.compile(
        optimizer='adam',
        loss={
            'classification': 'categorical_crossentropy',
            'autoencoder': 'binary_crossentropy'
        },
        loss_weights={
            'classification': 1.0,
            'autoencoder': 0.5
        },
        metrics={
            'classification': 'accuracy',
            'autoencoder': BatchCounter()
        }
    )
    return model

def GetInput():
    """
    Returns a random input tensor matching the input expected by MyModel.
    Shape: (Batch, Height, Width, Channels) == (100, 28, 28, 1)
    Values are floats in [0,1] range to simulate normalized MNIST images.
    """
    batch_size = 100
    img_rows, img_cols = 28, 28
    channels = 1
    # Random uniform mimicking normalized grayscale images
    return tf.random.uniform((batch_size, img_rows, img_cols, channels), dtype=tf.float32)

