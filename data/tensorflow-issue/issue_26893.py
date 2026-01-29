# tf.random.uniform((128, 96, 96, 3), dtype=tf.float32)
import tensorflow as tf

# This code is inferred from the provided issue.
# The original model is a SEResNext variant with ~27M parameters,
# input images are likely 96x96x3 as seen in logs (batch=128).
# The goal was TPU compilation, but ran out of "VMEM" which is an XLA memory space.
# Here, we reconstruct a minimal MyModel class capturing the described architecture.

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Since SEResNextImageNet is undefined, we use a placeholder backbone.
        # Assumption: input shape (96, 96, 3), output tensor shape inferred to be (24,24,256)
        # from fusion labels in logs (128,24,24,256). Thus spatial downsampling of 4 folds.
        # For demonstration, use a few Conv2D layers to produce tensor of that shape.
        
        self.backbone = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, 3, strides=1, padding='same', activation='relu'),
        ])
        # Output shape after backbone ~ (batch, 24, 24, 256)
        
        # Heads from the backbone output
        self.global_max_pool = tf.keras.layers.GlobalMaxPooling2D()
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        
        # Combine the three pooled outputs
        self.concat = tf.keras.layers.Concatenate()
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.final_dense = tf.keras.layers.Dense(1, activation='sigmoid',
                                                 kernel_initializer='glorot_uniform',
                                                 bias_initializer='zeros')
        
    def call(self, inputs, training=False):
        x = self.backbone(inputs)

        # Three different pooling/flatten operations on the same tensor
        out1 = self.global_max_pool(x)
        out2 = self.global_avg_pool(x)
        out3 = self.flatten(x)
        
        out = self.concat([out1, out2, out3])
        out = self.dropout1(out, training=training)
        out = self.dense1(out)
        out = self.dropout2(out, training=training)
        out = self.final_dense(out)
        
        return out

def my_model_function():
    # Return an instance of the model.
    return MyModel()

def GetInput():
    # Return a random input tensor of batch size 128 and shape (96, 96, 3), dtype float32.
    # This matches the expected input shape for MyModel.
    return tf.random.uniform((128, 96, 96, 3), dtype=tf.float32)

