# tf.random.uniform((5, 200, 200, 3), dtype=tf.float32) ← Inferred input shape from example and model definition

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using EfficientNetB2 backbone without top, pretrained on imagenet
        self.backbone = tf.keras.applications.EfficientNetB2(
            weights='imagenet', include_top=False)

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(30, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.backbone(inputs, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batchnorm(x, training=training)
        output = self.dense2(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile the model with SparseCategoricalCrossentropy loss and SparseCategoricalAccuracy metric,
    # matching the example in the issue
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    return model

def GetInput():
    # Return a batch of 5 random input images with shape (200, 200, 3), matching the provided code
    # Using float32 dtype consistent with EfficientNet preprocessing expectations
    # Note: Normalization is omitted for simplicity — users can preprocess if needed.
    return tf.random.uniform((5, 200, 200, 3), dtype=tf.float32)

