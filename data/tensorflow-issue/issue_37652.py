# tf.random.uniform((batch_size, variable_length), dtype=tf.string) ← Input is a sparse variable-length string tensor per batch element

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        
        # Define categorical feature column with fixed vocabulary (R, G, B)
        color_cat = tf.feature_column.categorical_column_with_vocabulary_list(
            key='color', vocabulary_list=["R", "G", "B"]
        )
        
        # Embedding column with dimension 4 and mean combiner for variable-length sparse inputs
        self.color_emb = tf.feature_column.embedding_column(color_cat, dimension=4, combiner='mean')

        # Use DenseFeatures layer to transform feature columns (expects dict input)
        self.dense_features = tf.keras.layers.DenseFeatures([self.color_emb])

        # Final classification layer (binary classification)
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output')

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        """
        inputs: dict with key 'color' containing a sparse or RaggedTensor of variable length strings of shape (batch_size, None)
        """
        # Pass inputs through DenseFeatures which handles sparse inputs and embedding lookups
        x = self.dense_features(inputs)
        x = self.output_layer(x)
        return x

def my_model_function():
    # Instantiate the model
    model = MyModel()
    
    # Compile the model with loss and optimizer to match original example
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Note: weights are random since we are not loading saved weights here
    return model

def GetInput():
    """
    Create a dummy input tensor dictionary structured as the model expects.
    The input is a sparse tensor of strings with a variable number of entries per batch example.
    
    We'll create a batch size 2 input with variable lengths to match the original use case:
    e.g. batch element 0: ["R"]
         batch element 1: ["G", "B"]
    """
    batch_size = 2
    
    # Simulate variable length input as a tf.RaggedTensor and convert to SparseTensor
    ragged_tensor = tf.ragged.constant([["R"], ["G", "B"]], dtype=tf.string)
    sparse_input = ragged_tensor.to_sparse()
    
    # Return a dictionary with the expected input key 'color' for the model
    return {'color': sparse_input}

