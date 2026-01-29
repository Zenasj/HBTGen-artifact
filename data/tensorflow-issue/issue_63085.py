# tf.random.uniform(shape=(), dtype=tf.string) ← Input is scalar string tensor (a batch of strings)
import tensorflow as tf
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Load the Universal Sentence Encoder from TF Hub
        # Using trainable=True as in the original snippet
        self.embedding_layer = hub.KerasLayer(
            'https://tfhub.dev/google/universal-sentence-encoder-large/5',
            trainable=True,
            name='USE_embedding'
        )
        # Wrap the hub layer in a Lambda layer to avoid Sequential model add errors
        self.embed_lambda = tf.keras.layers.Lambda(lambda x: self.embedding_layer(x), name='USE_lambda')
        
        # The dense output layer with sigmoid activation for regression/classification
        self.output_dense = tf.keras.layers.Dense(1, activation='sigmoid', name='output_dense')

    def call(self, inputs, training=False):
        # Input is string tensor(s)
        x = self.embed_lambda(inputs)
        x = self.output_dense(x)
        return x

def my_model_function():
    # Return a new instance of MyModel
    return MyModel()

def GetInput():
    # Generate a batch of 4 random example sentences as string tensors
    # (batch_size=4)
    example_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "TensorFlow is an open source machine learning framework.",
        "Testing Universal Sentence Encoder integration.",
        "This is a sample input to the model."
    ]
    return tf.constant(example_texts, dtype=tf.string)

