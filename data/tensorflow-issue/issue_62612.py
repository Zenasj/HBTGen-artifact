# tf.random.uniform((32, 224, 224, 3), dtype=tf.float32) ← inferred input shape from the data generator's target_size and batch_size

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Load the pre-trained InceptionV3 base model without the top layers
        self.base_model = tf.keras.applications.InceptionV3(
            include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        
        # Freeze the base model layers
        for layer in self.base_model.layers:
            layer.trainable = False
        
        # Additional fine-tuning layers
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.predictions = tf.keras.layers.Dense(3, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=training)
        x = self.global_avg_pool(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.predictions(x)

def my_model_function():
    # Create and compile the model to be consistent with the original code
    model = MyModel()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Generate a batch of random inputs with shape and dtype matching the expected input
    # Batch size 32 as per the original generator's batch_size
    batch_size = 32
    height = 224
    width = 224
    channels = 3
    # Input is likely float32 scaled [0,1], so uniform [0,1) is a reasonable assumption
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

# ---
# ### Explanation and assumptions made:
# - The original issue code is a fine-tuning of InceptionV3 with frozen base layers plus some dense layers.
# - The input shape for the model is inferred from the data generator target_size of (224,224) and batch_size=32, channels=3 (RGB image).
# - The activation functions, layer sizes, and architecture follow exactly as in the original snippet.
# - The model is wrapped as a subclassed `tf.keras.Model` named `MyModel` with the same layers.
# - The model is compiled inside `my_model_function()` to match original behavior.
# - `GetInput()` generates a random tensor simulating a batch from the data generator, scaled between 0 and 1.
# - No extraneous testing or saving code was included.
# - This model is compatible with TensorFlow 2.20.0 and can be jit-compiled with XLA using `tf.function(jit_compile=True)`.
# If you want me to help with the error in the original issue logs related to "SyntaxError: invalid syntax" inside tensorflow's `pywrap_tensorflow_internal.py`, that is likely caused by incompatibility between Python 3.12 and TensorFlow 2.15 (TF 2.15 does not officially support Python 3.12). Upgrading to a TF version supporting Python 3.12 or downgrading Python to 3.9 or 3.10 might be necessary. Please let me know if you want guidance on that as well.