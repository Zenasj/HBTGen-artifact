# tf.random.uniform((16, 224, 224, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A fused model encapsulating both the 'Grouped' and 'Regular' Conv2D stack models as submodules,
    with an output that compares their softmax predictions.
    
    - Input shape: [batch_size=16, height=224, width=224, channels=3]
    - Both models are similar convolutional stacks differing in the use of grouped convolutions (groups=8 vs groups=1).
    - Weights are independent.
    - The call method returns a dictionary with predictions from both branches and their L1 difference.

    This corresponds to the example comparing training speed and behavior of grouped vs regular Conv2D.
    """

    def __init__(self):
        super().__init__()
        # Common parameters
        batch_size = 16  # fixed batch size as per original code comments
        input_shape = (224, 224, 3)

        # Build Grouped model as a submodule using Functional API inside
        # Use InputLayer to define fixed batch size input shape for clarity
        self.grouped = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape, batch_size=batch_size),
            tf.keras.layers.Conv2D(128, kernel_size=1, padding='same', use_bias=False),
            tf.keras.layers.Conv2D(128, 3, padding='same', groups=8, use_bias=False),
            tf.keras.layers.Conv2D(128, 3, padding='same', groups=8, use_bias=False),
            tf.keras.layers.Conv2D(128, 3, padding='same', groups=8, use_bias=False),
            tf.keras.layers.Conv2D(128, 3, padding='same', groups=8, use_bias=False),
            tf.keras.layers.Conv2D(128, 3, padding='same', groups=8, use_bias=False),
            tf.keras.layers.Conv2D(128, 3, padding='same', groups=8, use_bias=False),
            tf.keras.layers.Conv2D(128, 3, padding='same', groups=8, use_bias=False),
            tf.keras.layers.Conv2D(128, 3, padding='same', groups=8, use_bias=False),
            tf.keras.layers.Conv2D(128, 3, padding='same', groups=8, use_bias=False),
            tf.keras.layers.Conv2D(128, 3, padding='same', groups=8, use_bias=False),
            tf.keras.layers.GlobalMaxPooling2D(),
            tf.keras.layers.Dense(2),
            tf.keras.layers.Activation('softmax')
        ], name="Grouped")

        # Build Regular model as a submodule similarly
        self.regular = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape, batch_size=batch_size),
            tf.keras.layers.Conv2D(128, kernel_size=1, padding='same', use_bias=False),
            tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False),
            tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False),
            tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False),
            tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False),
            tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False),
            tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False),
            tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False),
            tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False),
            tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False),
            tf.keras.layers.GlobalMaxPooling2D(),
            tf.keras.layers.Dense(2),
            tf.keras.layers.Activation('softmax')
        ], name="Regular")

    @tf.function(jit_compile=True)
    def call(self, x, training=False):
        # Forward through grouped conv model
        grouped_out = self.grouped(x, training=training)
        # Forward through regular conv model
        regular_out = self.regular(x, training=training)
        
        # Comparison: element-wise absolute difference of softmax outputs
        diff = tf.abs(grouped_out - regular_out)
        
        return {
            "grouped_pred": grouped_out,
            "regular_pred": regular_out,
            "abs_diff": diff,
        }

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random float32 tensor with shape (16, 224, 224, 3) matching the batch_size and input_shape
    # Values in [0,1], similar to the original np.full with 0.5 but random uniform to simulate varied inputs
    return tf.random.uniform(shape=(16, 224, 224, 3), dtype=tf.float32)

