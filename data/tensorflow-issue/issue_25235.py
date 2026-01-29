# tf.random.uniform((B, 28*28), dtype=tf.float32) ← Input is a batch of flattened 28x28 images as float32

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Dense layer for 10-class classification with softmax activation
        self.probas = tf.keras.layers.Dense(10, activation="softmax")
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name="serialized_inputs")
    ])
    def serve(self, serialized):
        # Parse serialized tf.Example proto containing an image feature of shape [28*28]
        expected_features = {
            "image": tf.io.FixedLenFeature([28*28], dtype=tf.float32)
        }
        examples = tf.io.parse_example(serialized, expected_features)
        # Pass parsed images through the model's Dense softmax layer and return probabilities
        return self.probas(examples["image"])
    
    def call(self, inputs):
        # Define forward pass for the model (required for subclassed Keras models)
        return self.probas(inputs)

def my_model_function():
    # Returns an instance of MyModel
    model = MyModel()
    # Compile the model with loss and optimizer to enable training and fitting
    model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd")
    return model

def GetInput():
    # Generate a batch of random images with shape (batch_size, 28*28)
    # Using float32 values between 0 and 1 to mimic normalized MNIST data
    batch_size = 4  # Reasonable example batch size
    # Return a float32 tensor representing flattened images
    return tf.random.uniform((batch_size, 28*28), dtype=tf.float32)

