# tf.random.uniform((B, 299, 299, 3), dtype=tf.float32) ← Assumed input shape based on IMAGE_DIM=299 and image loading preprocessing

import tensorflow as tf
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # The original model is loaded from a SavedModel directory which internally uses custom_objects={'KerasLayer': hub.KerasLayer}
        # Since we can't load here, we simulate the model architecture as a placeholder.
        # Assume the model expects 299x299 RGB images with float values normalized [0,1].
        #
        # Placeholder for the actual hub.KerasLayer based model:
        # For demonstration, use MobileNetV2 from TF Hub similar to nsfw_model,
        # or a stub sequential with a global average layer to simulate output shape of 5 classes.
        
        # Note: In the original repo, nsfw model likely outputs logits/probabilities for 5 categories:
        # ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
        
        # Here we create a minimal replacement that outputs 5 probabilities.
        
        self.model = tf.keras.Sequential([
            # Normalize input range as original loaded images divide by 255
            tf.keras.layers.Rescaling(scale=1./255, input_shape=(299,299,3)),
            tf.keras.layers.Conv2D(16, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')  # 5 category probs
        ])
        
    def call(self, inputs, training=False):
        # Forward pass through the model
        return self.model(inputs)

def my_model_function():
    # Instantiate model. In practice, this would load the pretrained model from disk via load_model.
    return MyModel()

def GetInput():
    # Return a random tensor matching (batch, 299, 299, 3) of floats in range [0,1]
    # The model expects inputs normalized by dividing by 255 on load,
    # so random floats uniform between 0-1 is appropriate.
    batch_size = 4  # a small batch to exercise predict
    return tf.random.uniform((batch_size, 299, 299, 3), dtype=tf.float32)

