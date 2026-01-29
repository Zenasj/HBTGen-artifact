# tf.random.uniform((1, 224, 224, 3), dtype=tf.float32) ← Input shape inferred from the MobileNetV3Large example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Load MobileNetV3Large with explicit input shape and batch size 1
        # This matches the issue context and representative dataset shape
        self.mobilenet = tf.keras.applications.MobileNetV3Large(
            input_tensor=tf.keras.Input(shape=(224, 224, 3), batch_size=1, dtype=tf.float32)
        )
        # Wrap in a Keras Model to keep consistency with original snippet
        self.model = tf.keras.Model(inputs=self.mobilenet.inputs,
                                    outputs=self.mobilenet(self.mobilenet.inputs))
        # Compile is optional but done in original example
        self.model.compile()
    
    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        # Forward pass through MobileNetV3Large
        return self.model(inputs, training=training)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random float32 tensor with shape (1, 224, 224, 3)
    # This matches the input shape specified to MobileNetV3Large and representative dataset
    # The dtype is float32, compatible with the model's expectation and TFLite conversion
    return tf.random.uniform(shape=(1, 224, 224, 3), dtype=tf.float32)

