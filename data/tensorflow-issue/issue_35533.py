# tf.random.uniform((B, sequence_length), dtype=tf.int32) ← Input is integer token indices batch for embedding lookup

import tensorflow as tf

# Assumptions / Inferences:
# - Input shape: (batch_size, sequence_length) with integer token indices
# - From code, embeddings shape: (vocab_size+2, embedding_dim)
# - FLAGS replaced by constants for demonstration
# - Activation used is ReLU or similar; issue discussed relates to BatchNorm, Dropout, and from_logits setting
# - We'll fuse the two variants (one with BatchNorm and Dropout, one without) as two submodels,
#   and present a combined model that can run both with comparison logic for demonstration.
# - We assume sequence_length = 100 (typical NLP value), vocab_size=5000, embedding_dim=300, num_classes=14 (from LABEL_IDX_DICT)
# - This is stripped of external dependencies (gensim, preprocessing) to focus on model code and comparison.
# - The fused MyModel outputs a boolean indicating if outputs from the two variants match within tolerance.

# Configuration mimicking FLAGS (simplified)
VOCAB_SIZE = 5000
EMBEDDING_DIM = 300
SEQUENCE_LENGTH = 100
FILTERS = 128
KERNEL_SIZE = 3
KERNEL_DISTANCE = 2
CONV_POOL_NUM = 3
CONV_STRIDES = 1
CONV_PADDING = "valid"
POOL_SIZE = 2
POOL_STRIDES = 2
POOL_PADDING = "valid"
DENSE_UNITS = 128
ACTIVATION = "relu"
DROPOUT_RATE = 0.5
NUM_CLASSES = 14

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        
        # Create a dummy constant embeddings matrix - normally pretrained embeddings would be used
        self.embeddings = tf.Variable(
            tf.random.normal([VOCAB_SIZE + 2, EMBEDDING_DIM]), trainable=False, name="embeddings"
        )
        
        # Embedding layer (non-trainable, constant initialized)
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=VOCAB_SIZE + 2, output_dim=EMBEDDING_DIM, 
            embeddings_initializer=tf.keras.initializers.Constant(self.embeddings),
            trainable=False
        )
        
        # Conv and pool layers, created per CONV_POOL_NUM, progressively larger kernels
        self.conv_layers = [
            tf.keras.layers.Conv1D(
                filters=FILTERS, 
                kernel_size=KERNEL_SIZE + idx * KERNEL_DISTANCE,
                strides=CONV_STRIDES,
                padding=CONV_PADDING,
                activation=None,  # We'll apply activation explicitly to emphasize difference
                name=f"conv_{idx}"
            ) for idx in range(CONV_POOL_NUM)
        ]
        self.pool_layers = [
            tf.keras.layers.MaxPool1D(
                pool_size=POOL_SIZE,
                strides=POOL_STRIDES,
                padding=POOL_PADDING,
                name=f"pool_{idx}"
            ) for idx in range(CONV_POOL_NUM)
        ]
        
        # Layers for variant 1: With BatchNorm, Dropout, Dense + activation + Dense-softmax
        self.batchnorm_layer = tf.keras.layers.BatchNormalization(name="batchnorm")
        self.dropout_layer = tf.keras.layers.Dropout(DROPOUT_RATE, name="dropout")
        self.dense1_layer = tf.keras.layers.Dense(DENSE_UNITS, activation=ACTIVATION, name="dense1")
        self.dense2_layer_softmax = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="dense2_softmax")
        
        # Layers for variant 2: No BatchNorm, No Dropout, Dense + activation + Dense (logits)
        self.dense1_layer_v2 = tf.keras.layers.Dense(DENSE_UNITS, activation=ACTIVATION, name="dense1_v2")
        self.dense2_layer_logits = tf.keras.layers.Dense(NUM_CLASSES, activation=None, name="dense2_logits")
        
    def call(self, inputs, training=False):
        """
        inputs: tf.Tensor of shape (batch_size, sequence_length), dtype int32
        
        Output: boolean tensor of shape () indicating whether outputs of variant1 and variant2
        are sufficiently close, comparing softmax logits vs logits+softmax.
        
        This emulates the fusion of the two variants described in the issue:
        - Variant1: original with BatchNorm and Dropout, outputs softmax
        - Variant2: simplified without BatchNorm and Dropout, final Dense output logits (no softmax)
        The output will be a tf.reduce_all boolean indicating if predictions match closely.
        """
        x = inputs
        # Embed tokens
        x_embed = self.embedding_layer(x)  # (B, seq_len, embed_dim)
        
        # Apply conv + pool for each branch identically
        conv_outputs = []
        for idx in range(CONV_POOL_NUM):
            c = self.conv_layers[idx](x_embed)  # Conv1D output, no activation yet
            c = tf.nn.relu(c)  # Apply ReLU activation explicitly
            p = self.pool_layers[idx](c)
            conv_outputs.append(p)
        
        concatenated = tf.concat(conv_outputs, axis=-1)  # concat channels/features axis
        flat = tf.keras.layers.Flatten()(concatenated)
        
        # Variant 1 forward pass: with BatchNorm + Dropout + Dense + softmax
        x1 = self.batchnorm_layer(flat, training=training)
        x1 = self.dropout_layer(x1, training=training)
        x1 = self.dense1_layer(x1)
        out1 = self.dense2_layer_softmax(x1)  # probabilities shape (B, NUM_CLASSES)
        
        # Variant 2 forward pass: without BatchNorm + Dropout + Dense + logits output
        x2 = self.dense1_layer_v2(flat)
        out2_logits = self.dense2_layer_logits(x2)  # logits shape (B, NUM_CLASSES)
        out2 = tf.nn.softmax(out2_logits)  # Probabilities for comparison
        
        # Now compare outputs: check if both predictions are close (within tolerance)
        # Using absolute difference tolerance 1e-5 for example
        close = tf.reduce_all(tf.abs(out1 - out2) < 1e-5)
        return close

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random integer tensor as input matching embedding lookup input
    # Shape: (batch_size, sequence_length)
    # Values: random int in [0, VOCAB_SIZE+1]
    batch_size = 8  # chosen for example
    return tf.random.uniform(
        shape=(batch_size, SEQUENCE_LENGTH), minval=0, maxval=VOCAB_SIZE + 2, dtype=tf.int32
    )

