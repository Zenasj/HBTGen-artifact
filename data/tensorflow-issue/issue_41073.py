# tf.random.uniform((B, 6), dtype=tf.string) ← Input shape inferred from 'text' input with max_length=6 and dtype string

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Note: feature_column.shared_embeddings and SequenceFeatures are from tf.compat.v1 APIs.
        # We replicate the model architecture from the issue using the TensorFlow 2.x compatible approach,
        # but to keep close to the original example, we use the feature_column API in compatibility mode.

        # Define the categorical column for sequence categorical data with vocabulary size 2
        self.text_column = tf.compat.v1.feature_column.sequence_categorical_column_with_vocabulary_list(
            key='text', vocabulary_list=['asd', 'asdf'])

        # Create a shared embedding for the categorical column with dimension 64
        shared_embedding = tf.compat.v1.feature_column.shared_embeddings(
            [self.text_column], dimension=64)

        # SequenceFeatures layer processes sequence categorical columns and returns a tuple of tensors
        self.sequence_feature_layer = tf.keras.experimental.SequenceFeatures(shared_embedding)

        # Following the original model architecture: Conv1D, MaxPooling1D, Dense, Dropout, GlobalAveragePooling1D, final Dense
        self.conv1d = tf.keras.layers.Conv1D(8, 4)
        self.maxpool = tf.keras.layers.MaxPooling1D(2)
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.final_dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        # inputs is expected to be a dictionary, e.g. {'text': tensor_of_shape_(batch, 6)}

        # The SequenceFeatures layer returns tuple of (tensor_output, feature_state)
        sequence_outputs, _ = self.sequence_feature_layer(inputs)

        x = self.conv1d(sequence_outputs)
        x = self.maxpool(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.global_avg_pool(x)
        output = self.final_dense(x)

        return output

def my_model_function():
    # Create and return an instance of MyModel
    model = MyModel()

    # Build model by calling on a dummy input with batch size 1 to initialize weights
    dummy_input = GetInput()
    _ = model(dummy_input)

    # Compile the model consistent with original example
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a dictionary with key 'text' containing a batch of random string tensors of shape (batch, 6)
    # Since model uses vocabulary_list=['asd', 'asdf'], generate random choice from those values

    batch_size = 4  # arbitrary small batch size

    vocab = tf.constant(['asd', 'asdf'])

    # Randomly select strings from vocab for each element in (batch_size, 6)
    indices = tf.random.uniform(shape=(batch_size, 6), minval=0, maxval=2, dtype=tf.int32)
    random_strings = tf.gather(vocab, indices)

    return {'text': random_strings}

