# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Input shape is inferred as a 4D tensor (batch, height, width, channels)

import tensorflow as tf


class MyModel(tf.keras.Model):
    """
    A Keras model replicating a typical Conv2D-based architecture
    whose TFLite conversion triggers errors due to control flow cycles.
    
    NOTE:
    The original issue describes a model with conv layers and a
    final map_fn call causing a cycle during TFLite conversion,
    related to unsupported control flow ops like Merge and NextIteration.
    
    Since the original model code isn't provided, this class simulates
    a Conv2D-based backbone with a dummy control flow (tf.while_loop)
    to illustrate the source of cycle errors.

    The forward pass returns outputs normally, but internally contains
    a tf.while_loop producing potential cycles.
    """
    def __init__(self):
        super().__init__()
        # Basic conv layers as per description
        self.conv1 = tf.keras.layers.Conv2D(16, 3, padding="same", activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(32, 3, padding="same", activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.MaxPool2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool(x)
        
        # Inserted dummy control flow loop resembling map_fn or batch loop.
        # This emulates the problematic control flow that toco rejects.
        # It produces ops like Merge and NextIteration which cause cycles.
        def cond(i, outputs):
            # Run loop for B steps (batch size)
            return i < tf.shape(inputs)[0]

        def body(i, outputs):
            # Simulate some computation per batch element
            batch_element = x[i]
            # Just compute mean as dummy op
            processed = tf.reduce_mean(batch_element)
            outputs = outputs.write(i, processed)
            return i + 1, outputs

        batch_size = tf.shape(inputs)[0]
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=batch_size)
        i = tf.constant(0)
        _, outputs_ta = tf.while_loop(cond, body, loop_vars=[i, outputs_ta])
        outputs = outputs_ta.stack()  # shape (batch_size,)
        
        # Further use outputs to produce prediction (dummy example)
        # Replicate output shape to match expected num classes
        preds = tf.tile(tf.expand_dims(outputs, -1), [1, 10])
        
        # Final dense layer from code to simulate end logic
        pred_final = self.dense(x)  # shape (batch_size, spatial_dims, channels)
        pred_final_flat = self.flatten(pred_final)
        
        # Combine results: simulate final output
        combined = preds + pred_final_flat[:, :10]
        
        return combined


def my_model_function():
    # Instantiate and return the MyModel object.
    return MyModel()


def GetInput():
    # Provide a random 4D input tensor matching the input expected by MyModel
    # Batch size 4, spatial dims 64x64, channels 3 (e.g. RGB image)
    return tf.random.uniform(shape=(4, 64, 64, 3), dtype=tf.float32)

