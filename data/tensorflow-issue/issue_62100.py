# tf.random.uniform((B, 100), dtype=tf.int32) and tf.random.uniform((B, 50), dtype=tf.int32) ← batch size B, input lengths 100 and 50

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Constants from the issue's model
        self.vocab_size = 200_000
        self.vocab_dim = 100
        self.length1 = 50
        self.length2 = 100
        self.conv_dim = 128
        self.window_sizes = [2, 3, 4, 5]
        self.num_classes = 2000
        
        # Embeddings for each input feature
        self.embedding1 = tf.keras.layers.Embedding(self.vocab_size, self.vocab_dim)
        self.embedding2 = tf.keras.layers.Embedding(self.vocab_size, self.vocab_dim)
        
        # Convolution/BN/ReLU/Pooling blocks for each window size and each input
        # Create layers for input1 set
        self.conv_blocks1 = []
        for w in self.window_sizes:
            self.conv_blocks1.append([
                tf.keras.layers.Conv1D(self.conv_dim, w),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.GlobalMaxPooling1D()
            ])
        
        # Create layers for input2 set
        self.conv_blocks2 = []
        for w in self.window_sizes:
            self.conv_blocks2.append([
                tf.keras.layers.Conv1D(self.conv_dim, w),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.GlobalMaxPooling1D()
            ])
        
        # Dropout and final dense layer for classification
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense = tf.keras.layers.Dense(self.num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        # inputs is expected to be a list/tuple of two tensors: [input1, input2]
        input1, input2 = inputs
        
        # Embed inputs
        embed1 = self.embedding1(input1)  # shape: (B, length1, vocab_dim)
        embed2 = self.embedding2(input2)  # shape: (B, length2, vocab_dim)
        
        hidden_tensors = []
        # Process input1 through conv blocks
        for conv, bn, act, pool in self.conv_blocks1:
            x = conv(embed1)
            x = bn(x, training=training)
            x = act(x)
            x = pool(x)
            hidden_tensors.append(x)
        
        # Process input2 through conv blocks
        for conv, bn, act, pool in self.conv_blocks2:
            x = conv(embed2)
            x = bn(x, training=training)
            x = act(x)
            x = pool(x)
            hidden_tensors.append(x)
        
        # Concatenate all pooled outputs
        hidden = tf.keras.layers.concatenate(hidden_tensors)
        
        # Apply dropout
        hidden = self.dropout(hidden, training=training)
        
        # Final classification layer with softmax
        output = self.dense(hidden)
        
        return output

def my_model_function():
    # Create an instance of the model, no pretrained weights specified
    return MyModel()

def GetInput():
    # Infer batch size B = 256 as in original training batch size for compatibility
    B = 256
    # Input1 shape: (B, 50), dtype int32 vocab indices
    input1 = tf.random.uniform(shape=(B, 50), maxval=200_000, dtype=tf.int32)
    # Input2 shape: (B, 100), dtype int32 vocab indices
    input2 = tf.random.uniform(shape=(B, 100), maxval=200_000, dtype=tf.int32)
    return [input1, input2]

