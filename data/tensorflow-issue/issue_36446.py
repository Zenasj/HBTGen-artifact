# tf.random.uniform((B, 180, 320, 3), dtype=tf.float32) ← inferred input shape from original model example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Recreate backbone (e.g. ResNet50V2 without top) as per issue description:
        # weights=None, include_top=False, input_shape=(180, 320, 3)
        self.backbone = tf.keras.applications.ResNet50V2(
            input_shape=(180, 320, 3),
            weights=None,
            include_top=False,
        )
        
        # Pooling layer
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        
        # Final output dense layer for 8 outputs (regression or classification logits)
        self.classifier = tf.keras.layers.Dense(8)

        # Note: BatchNormalization layers inside backbone cause the "train vs val loss discrepancy".
        # This model does nothing special to fix it here, just reproduces the original model setup.

    def call(self, inputs, training=False):
        x = self.backbone(inputs, training=training)
        x = self.global_pool(x)
        output = self.classifier(x)
        return output

def my_model_function():
    # Instantiate the model - no pre-trained weights, fresh initialization
    model = MyModel()
    # Normally you would compile here outside as well, but leaving it to user as per usual TF pattern.
    return model

def GetInput():
    # Generate a random input tensor matching the input shape used in the issue (2 samples, 180x320 RGB)
    # This can be used as input for the model
    # dtype float32 typically expected by keras applications
    return tf.random.uniform((2, 180, 320, 3), dtype=tf.float32)

