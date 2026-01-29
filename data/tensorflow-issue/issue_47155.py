# tf.random.uniform((B, H), dtype=tf.int32) ← Assuming input shape (batch_size, conv_input_height) with integer word indices

import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Assumptions based on code in the issue:
        # - Vocabulary size = vocab_size (W.shape[0])
        # - Embedding dim = embed_dim (W.shape[1])
        # - conv_input_height and conv_input_width inferred from W and input shape
        
        # These constants would normally be passed/loaded:
        # For demonstration, we assume some example values.
        # In practice, set vocab_size, embed_dim, conv_input_height, conv_input_width accordingly.
        
        # Since no weights loaded, we'll initialize embeddings randomly.
        # The original model used pre-trained weights 'W'.
        
        vocab_size = 10000  # placeholder, inferred from W.shape[0]
        embed_dim = 50      # placeholder, inferred from W.shape[1]
        conv_input_height = 100  # input sequence length (height)
        conv_input_width = embed_dim  # set to embed_dim, width after embedding reshape
        
        N_fm = 200  # number of convolutional feature maps
        kernel_size = 5
        
        # Embedding layer without embeddings_constraint (to avoid sparse constraint error)
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            input_length=conv_input_height,
            # weights would be set here if available, but omitted due to constraint issues
            # embeddings_constraint=tf.keras.constraints.UnitNorm(axis=1),  # Removed per issue discussion
        )
        # Reshape layer from (batch, seq_len, embed_dim) to (batch, 1, seq_len, embed_dim)
        self.reshape = tf.keras.layers.Reshape((1, conv_input_height, conv_input_width))
        
        # Conv2D layer: filters=N_fm, kernel_size=(kernel_size, conv_input_width)
        # padding='valid' as in original code
        # kernel_initializer='random_uniform'
        self.conv2d = tf.keras.layers.Conv2D(
            filters=N_fm,
            kernel_size=(kernel_size, conv_input_width),
            padding='valid',
            kernel_initializer='random_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            data_format='channels_first'
        )
        
        # Activation ReLU
        self.relu = tf.keras.layers.Activation('relu')
        
        # MaxPooling2D with pool size: (conv_input_height + kernel_size + 1, 1) and padding same
        # data_format channels_first means (batch, channels, height, width)
        self.maxpool = tf.keras.layers.MaxPooling2D(
            pool_size=(conv_input_height + kernel_size + 1, 1),
            padding='same',
            data_format='channels_first'
        )
        
        self.flatten = tf.keras.layers.Flatten()
        self.dropout1 = tf.keras.layers.Dropout(0.4)
        self.dense1 = tf.keras.layers.Dense(128, kernel_initializer='random_uniform')
        self.relu2 = tf.keras.layers.Activation('relu')
        self.dropout2 = tf.keras.layers.Dropout(0.4)
        self.dense2 = tf.keras.layers.Dense(2)
        self.softmax = tf.keras.layers.Activation('softmax')
        
    def call(self, inputs, training=None):
        """
        inputs: integer tensor shape (batch_size, conv_input_height)
        """
        x = self.embedding(inputs)  # (batch, conv_input_height, embed_dim)
        x = self.reshape(x)         # (batch, 1, conv_input_height, conv_input_width)
        x = self.conv2d(x)          # (batch, N_fm, new_height, 1)
        x = self.relu(x)
        x = self.maxpool(x)         # (batch, N_fm, 1, 1)
        x = self.flatten(x)         # (batch, N_fm)
        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        x = self.relu2(x)
        x = self.dropout2(x, training=training)
        x = self.dense2(x)
        output = self.softmax(x)
        return output


def my_model_function():
    # Return an instance of MyModel; no pretrained weights loaded due to scope and constraint issue
    model = MyModel()
    # Compile with RMSprop to match original setup
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['accuracy']
    )
    return model


def GetInput():
    # Return a random integer tensor input that matches the model's input:
    # Shape (batch_size, conv_input_height)
    batch_size = 32
    conv_input_height = 100  # assumed from above
    vocab_size = 10000       # must match embedding input_dim
    
    # Random integers in [0, vocab_size) for word indices
    return tf.random.uniform(
        (batch_size, conv_input_height), minval=0, maxval=vocab_size, dtype=tf.int32
    )

