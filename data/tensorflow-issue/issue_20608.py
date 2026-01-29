# tf.random.uniform((B, 784), dtype=tf.float32) ← inferred input shape from MNIST flattened input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, shared_conv_kernel_size=(5,5), conv4_kernel_size=(3,3)):
        """
        Constructs a model similar to the two discussed in the issue:
        - A shared base containing 2 conv layers + maxpool
        - Then a conv layer that differs in kernel_size (3x3 or 5x5)
        - Followed by flatten and a dense layer with 10 outputs (no activation)
        
        This allows us to instantiate two variants and compare their outputs.
        
        Args:
            shared_conv_kernel_size: kernel size for conv_1 (default 5x5)
            conv4_kernel_size: kernel size for the conv_4 layer (3x3 or 5x5)
        """
        super().__init__()
        L = tf.keras.layers
        
        # Shared subgraph/layers (like "input", "conv_1_5x5", "conv_2_3x3", "pool_3_2x2")
        # Note: We combine input reshape with the model call input preprocessing.
        self.shared_conv1 = L.Conv2D(filters=32, kernel_size=shared_conv_kernel_size,
                                    padding="same", activation='relu', name="conv_1")
        self.shared_conv2 = L.Conv2D(filters=32, kernel_size=(3, 3),
                                    padding="same", activation='relu', name="conv_2")
        self.shared_pool = L.MaxPooling2D(pool_size=(2, 2), strides=2, name="pool_3")
        
        # The conv_4 layer differs between models (we'll parametrize kernel size)
        self.conv4 = L.Conv2D(filters=64, kernel_size=conv4_kernel_size,
                             padding="same", activation='relu', name="conv_4")
        
        self.flatten = L.Flatten(name="flatten")
        # Dense 10 units with no activation (logits), to match TF1 behavior
        self.dense = L.Dense(10, activation=None, name="dense")
        
    def call(self, inputs, training=False):
        """
        Expects input of shape (batch, 784), representing flattened MNIST images.
        Reshapes to (batch, 28, 28, 1) internally.
        Returns logits output (batch, 10).
        """
        x = tf.reshape(inputs, [-1, 28, 28, 1])  # reshape flat input to image
        x = self.shared_conv1(x)
        x = self.shared_conv2(x)
        x = self.shared_pool(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.dense(x)
        return logits


class MyModelWrapper(tf.keras.Model):
    """
    This wraps the fusion of two sub-models representing model_1 and model_2
    from the provided issue:
    
    - model1: conv4 kernel size 3x3
    - model2: conv4 kernel size 5x5
    
    We provide a single interface that takes input and returns a dictionary with:
    {
       'model1_logits': logits from model1,
       'model2_logits': logits from model2,
       'logits_difference': elementwise difference,
       'outputs_close': boolean tensor if all close within tolerance
    }
    """
    def __init__(self, rtol=1e-5, atol=1e-8):
        super().__init__()
        self.model1 = MyModel(conv4_kernel_size=(3,3))
        self.model2 = MyModel(conv4_kernel_size=(5,5))
        self.rtol = rtol
        self.atol = atol
    
    def call(self, inputs, training=False):
        logits1 = self.model1(inputs, training=training)
        logits2 = self.model2(inputs, training=training)
        
        diff = logits1 - logits2
        # Check if outputs are close within the given tolerance elementwise
        outputs_close = tf.reduce_all(tf.math.abs(diff) <= (self.atol + self.rtol * tf.math.abs(logits2)))
        
        return {
            'model1_logits': logits1,
            'model2_logits': logits2,
            'logits_difference': diff,
            'outputs_close': outputs_close
        }


def my_model_function():
    # Return a fresh instance of the fused MyModelWrapper class which 
    # incorporates both models for shared weights + comparison as described.
    return MyModelWrapper()


def GetInput():
    # The input is expected as (batch, 784) flattened MNIST-like tensor,
    # with dtype float32. We'll create a random uniform tensor simulating
    # a batch of 32 samples. Values normalized between 0 and 1 as typical for MNIST.
    batch_size = 32
    return tf.random.uniform(shape=(batch_size, 784), minval=0, maxval=1, dtype=tf.float32)

