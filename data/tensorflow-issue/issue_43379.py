# tf.random.uniform((2, 1), dtype=tf.uint8)
import tensorflow as tf
import itertools

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model with single input of shape (1,)
        # Two Dense layers outputting (32,) each as in the original example
        self.dense1 = tf.keras.layers.Dense(32)
        self.dense2 = tf.keras.layers.Dense(32)
        
    def call(self, inputs, training=None):
        # Inputs expected shape: (batch_size, 1)
        x = tf.cast(inputs, tf.float32)
        out1 = self.dense1(x)  # shape (batch_size, 32)
        out2 = self.dense2(x)  # shape (batch_size, 32)
        # Return a tuple of outputs for multi-output model
        return (out1, out2)

def my_model_function():
    # Return compiled instance of MyModel with SparseCategoricalCrossentropy loss for each output
    # NOTE: The original issue was about using a single loss for multiple outputs without specifying loss per output,
    # which leads to shape mismatch. So we specify the loss as a list of SparseCategoricalCrossentropy for each output.
    model = MyModel()
    
    # Compile with list of losses for multiple outputs
    # from_logits=True needed because SparseCategoricalCrossentropy expects logits by default (Dense layers have linear activation)
    model.compile(
        optimizer='adam',
        loss=[
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        ]
    )
    return model

def GetInput():
    # Generator that yields inputs and outputs in the expected shapes consistent with model outputs
    
    def gen():
        # Batch size = 2, input shape (1,)
        # Inputs: shape (2,1), produce constant values (just for example)
        # Outputs: two outputs each of shape (2, 32) for SparseCategoricalCrossentropy, labels should be ints in [0,num_classes)
        # To prevent shape mismatch, outputs must be (2,) integers (class labels) or appropriate shape matching logits except last dim
        
        # The original code yields inputs [1,2] shaped (2,) without shape (2,1), and outputs [[11,12],[21,22]] which was shape (2,2)
        # That shape can't work with output Dense(32) layers expecting (2,32).
        # So here we yield inputs shaped (2,1), outputs shaped (2,) with class indices in [0,32)
        
        # For demo, we just cycle sample data here. Use numbers mod 32 as labels.
        batch_size = 2
        num_classes = 32
        i = 0
        while True:
            inputs = tf.random.uniform((batch_size,1), minval=0, maxval=100, dtype=tf.float32)
            # Outputs: batch of class indices for each output, shape (batch_size,)
            # E.g., labels for first output: [i % num_classes, (i+1) % num_classes]
            out1_labels = tf.constant([(i % num_classes), ((i+1) % num_classes)], dtype=tf.int32)
            out2_labels = tf.constant([((i+2) % num_classes), ((i+3) % num_classes)], dtype=tf.int32)
            i += 4
            yield inputs, (out1_labels, out2_labels)

    # Using from_generator with correct output signature matching model inputs and outputs
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(2,1), dtype=tf.float32),
            (
                tf.TensorSpec(shape=(2,), dtype=tf.int32),
                tf.TensorSpec(shape=(2,), dtype=tf.int32)
            )
        )
    )
    return dataset

