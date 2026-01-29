# tf.random.uniform((B, 64), dtype=tf.int32) ← Inputs are three tensors: source (64 int32 tokens), target (64 int32 tokens), relationship (1 int32 token)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Embedding layers for source and target sequences
        self.embedding_source = tf.keras.layers.Embedding(input_dim=512, output_dim=512, input_length=64)
        self.embedding_target = tf.keras.layers.Embedding(input_dim=512, output_dim=512, input_length=64)

        # Custom layer that defines a trainable kernel weight but returns first input as output
        # This mimics the original CustomModel layer behavior
        self.custom_layer = CustomModel()

    def call(self, inputs):
        source, target, relationship = inputs
        emb_source = self.embedding_source(source)        # Shape: (batch_size, 64, 512)
        emb_target = self.embedding_target(target)        # Shape: (batch_size, 64, 512)

        # Pass embeddings along with relationship input to the custom layer
        # Note: based on original bug report, order of inputs matters; using source embedding, target embedding, then relationship
        output = self.custom_layer([emb_source, emb_target, relationship])
        return output

class CustomModel(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomModel, self).__init__()

    def build(self, input_shape):
        # The kernel shape: (32, 512, 512) from original snippet
        # Note: this is a dummy trainable weight not used in call (as per original)
        self.kernel = self.add_weight(
            shape=(32, 512, 512),
            initializer=tf.keras.initializers.glorot_uniform(seed=1),
            trainable=True,
            name="kernel"
        )

    def call(self, inputs):
        # According to the original bug report model, just return the first input (embedding source)
        return inputs[0]

def my_model_function():
    # Instantiate the MyModel instance
    return MyModel()

def GetInput():
    # Generate a tuple of three inputs matching the model inputs:
    # source: int32 tensor shape (batch_size, 64)
    # target: int32 tensor shape (batch_size, 64)
    # relationship: int32 tensor shape (batch_size, 1)
    batch_size = 8  # reasonable default batch size for input example
    
    source = tf.random.uniform(shape=(batch_size, 64), minval=0, maxval=512, dtype=tf.int32)
    target = tf.random.uniform(shape=(batch_size, 64), minval=0, maxval=512, dtype=tf.int32)
    relationship = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=512, dtype=tf.int32)

    return (source, target, relationship)

