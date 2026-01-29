# tf.random.uniform((B, 10), dtype=tf.int32) ← Input shape inferred from X_train shape in the issue: (batch_size, 10)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_targets=3811, input_length=10, vocab_size=170000, embedding_dim=100):
        super().__init__()
        # Embedding layer to map vocab ids to embeddings
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, 
                                                   output_dim=embedding_dim,
                                                   input_length=input_length)
        # Global average pooling to aggregate sequence embeddings
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        # Dense layer with sigmoid activation for multilabel binary classification
        self.classifier = tf.keras.layers.Dense(num_targets, activation='sigmoid')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.global_pool(x)
        outputs = self.classifier(x)
        return outputs

def my_model_function():
    # Return an instance of MyModel with default parameters matching the issue
    return MyModel()

def GetInput():
    # The model input is a batch of sequences with shape (batch_size, 10), dtype int32
    # We'll use a reasonably small batch size to avoid memory issues during testing
    batch_size = 4  # Chosen small for demonstration; can be adjusted
    input_length = 10
    vocab_size = 170000
    # Random integers in [0, vocab_size) as token indices
    return tf.random.uniform((batch_size, input_length), minval=0, maxval=vocab_size, dtype=tf.int32)

# ---
# ### Explanation / Assumptions:
# - The original issue revolves around a model that processes input sequences of length 10, vocabulary size approx 170000, output dimension 3811.
# - The original Sequential model has embedding, pooling, dense with either softmax or sigmoid activation.
# - Since the last question in the issue is about using *binary_crossentropy* with *sigmoid* output (no sparse version), the code here models that case.
# - The input shape is (batch_size, 10) integer indices; output shape is (batch_size, 3811) with sigmoid.
# - The user had enormous datasets; here we use a small batch size in GetInput for demo purposes.
# - All layers and parameters are inferred from the posted code snippets to replicate the core model.
# - No explicit loss or compilation is included here, just the model and input tensor generator, as requested.
# - This model is compatible with TF 2.20.0 and can be used with XLA jit compiling if desired.
# If you want me to include a compiled training step or evaluation mode, just ask!