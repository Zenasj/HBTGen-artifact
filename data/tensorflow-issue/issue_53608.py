# tf.random.uniform((BATCH_SIZE, 240, 320, 3), dtype=tf.float32)
import tensorflow as tf

BATCH_SIZE = 48
N_CLASSES = 27
INPUT_SHAPE = (240, 320, 3)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using ResNet50V2 backbone without the top classification layer, average pooling at the end
        self.base_model = tf.keras.applications.ResNet50V2(
            include_top=False,
            pooling='avg',
            weights=None,
            input_shape=INPUT_SHAPE
        )
        # Final dense layer with L2 regularization
        self.classifier = tf.keras.layers.Dense(
            N_CLASSES,
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )

    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=training)
        logits = self.classifier(x)
        return logits

def my_model_function():
    # Instantiate and return the MyModel instance
    model = MyModel()
    # Build the model by providing an input shape batch dimension unspecified (None)
    model.build((None,) + INPUT_SHAPE)
    return model

def GetInput():
    # Return a random tensor of shape [BATCH_SIZE, 240, 320, 3], dtype float32
    # matching the input required by ResNet50V2 in the MyModel
    return tf.random.uniform(shape=(BATCH_SIZE,) + INPUT_SHAPE, dtype=tf.float32)

