# tf.random.uniform((None,), dtype=tf.string) ← Model input is a batch of strings (text sequences)

import tensorflow as tf
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self, 
                 tfhub_handle_preprocess="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
                 tfhub_handle_encoder="https://tfhub.dev/tensorflow/bert_en_uncased_L-6_H-512_A-8/3",
                 hidden_layer_dims=256,
                 dropout_rate=0.1):
        """
        Initialize the sequence classification model using a BERT TF-Hub encoder and preprocessor.
        
        Args:
            tfhub_handle_preprocess: URL of the TF-Hub preprocessing model (updated to v3 to avoid known issues).
            tfhub_handle_encoder: URL of the TF-Hub BERT encoder model.
            hidden_layer_dims: Number of units in the dense pre-classification layer.
            dropout_rate: Dropout probability.
        """
        super().__init__()
        # Load preprocessing and encoder layers
        self.preprocessor = hub.KerasLayer(tfhub_handle_preprocess, name="preprocessing")
        self.encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name="BERT_encoder")

        # Classification head layers
        self.pre_classifier = tf.keras.layers.Dense(
            hidden_layer_dims,
            activation="relu",
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.002),
            name="pre_classifier"
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        # Binary classification with 2 outputs and sigmoid activation
        self.classifier = tf.keras.layers.Dense(2, activation="sigmoid", name="classifier")

    def call(self, inputs, training=False):
        """
        Forward pass for the model: preprocess input text, encode with BERT, classify.
        
        Args:
            inputs: batch of strings, shape (batch_size,)
            training: Boolean, whether in training mode (for dropout)
        
        Returns:
            Tensor with shape (batch_size, 2), sigmoid probabilities for each class
        """
        # Preprocessing step transforms raw text to BERT inputs
        encoder_inputs = self.preprocessor(inputs)
        # Encoder outputs dictionary with pooled_output for classification
        encoder_outputs = self.encoder(encoder_inputs)
        pooled_output = encoder_outputs["pooled_output"]  # Shape: (batch_size, hidden_size)

        x = self.pre_classifier(pooled_output)
        x = self.dropout(x, training=training)
        output = self.classifier(x)
        return output


def my_model_function():
    """
    Returns an instance of the MyModel with default parameters.
    This is the same model used in the provided issue code snippet.
    """
    return MyModel()


def GetInput():
    """
    Generates a batch of random string inputs suitable for the MyModel.
    Since the model expects a batch of raw text strings (shape=(batch_size, )),
    this generates dummy string tensors.
    
    Note:
      For BERT preprocessing, the input is expected to be a vector of strings,
      each string representing one input example.
    """
    batch_size = 4  # Small batch for example
    dummy_text = tf.constant([
        "This is a sample input text.",
        "Another example text for classification.",
        "TensorFlow is great for deep learning.",
        "Let's test the BERT model preprocessing."
    ])
    # Repeat or truncate to match batch_size if needed
    return dummy_text[:batch_size]

