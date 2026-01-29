# tf.random.uniform((B,)) with dtype=tf.string ← Model input expects a batch of strings of variable shape

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Because the original model is a BERT text classification model using TF Hub layers
        # inference: 
        #   Input: batch of strings, shape (batch_size,)
        #   Layers:
        #     - TF Hub preprocessing layer
        #     - TF Hub BERT encoder layer (trainable)
        #     - Several Dense layers with ReLU activations and final softmax output

        # Placeholder URLs - user must replace with actual TFHub handles
        self.tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
        self.tfhub_handle_encoder = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"

        import tensorflow_hub as hub
        # TF Hub layers
        self.preprocessing_layer = hub.KerasLayer(self.tfhub_handle_preprocess, name='preprocessing')
        self.encoder = hub.KerasLayer(self.tfhub_handle_encoder, trainable=True, name='BERT_encoder')

        # Dropout for regularization
        self.dropout = tf.keras.layers.Dropout(0.1)

        # Dense layers as per original, note: careful with naming duplicates
        self.dense_1024 = tf.keras.layers.Dense(1024, activation='relu', name='hidden')
        self.dense_256_a = tf.keras.layers.Dense(256, activation='relu', name='hidden_1')
        self.dense_256_b = tf.keras.layers.Dense(256, activation='relu', name='hidden_2')
        self.dense_128 = tf.keras.layers.Dense(128, activation='relu', name='hidden_3')
        self.dense_64_a = tf.keras.layers.Dense(64, activation='relu', name='hidden_4')
        self.dense_64_b = tf.keras.layers.Dense(64, activation='relu', name='hidden_5')
        self.dense_16 = tf.keras.layers.Dense(16, activation='relu', name='hidden_6')
        self.output_layer = tf.keras.layers.Dense(3, activation='softmax', name='output')

    def call(self, inputs, training=False):
        """
        Forward pass.
        inputs: tf.Tensor of dtype tf.string and shape (batch_size,)
        """

        # 1) Preprocessing (tokenization, etc) via TFHub preprocessing layer
        encoder_inputs = self.preprocessing_layer(inputs)

        # 2) BERT encoding, returns dict with 'pooled_output' and 'sequence_output'
        bert_outputs = self.encoder(encoder_inputs)
        pooled_output = bert_outputs['pooled_output']  # shape (batch_size, 768)

        # 3) Dropout applied on pooled_output (only during training)
        x = self.dropout(pooled_output, training=training)

        # 4) Fully connected layers as per original
        x = self.dense_1024(x)
        x = self.dense_256_a(x)
        x = self.dense_256_b(x)
        x = self.dense_128(x)
        x = self.dense_64_a(x)
        x = self.dense_64_b(x)
        x = self.dense_16(x)
        outputs = self.output_layer(x)  # softmax probabilities over 3 classes

        return outputs


def my_model_function():
    """
    Returns an instance of MyModel with initialized TF Hub layers.
    Note: requires tensorflow_hub installed and internet to download modules or 
    cached modules on local disk. 
    """
    return MyModel()


def GetInput():
    """
    Returns a random tensor input that matches the expected input of MyModel.
    Since input shape is (batch_size,) with dtype string, we generate
    a batch of random sentences (dummy strings).

    Assumption: batch size 32, and each string length approx 10 tokens (words).
    We generate dummy strings of repeated tokens for simplicity.
    """
    batch_size = 32
    # Create dummy "sentences": e.g. "word word word ..." repeated 10 times
    dummy_sentences = ["hello tensorflow keras model" for _ in range(batch_size)]
    return tf.constant(dummy_sentences, dtype=tf.string)

