# tf.random.uniform((B, 10), dtype=tf.int32) ← inferred input shape for input_ids and attention_mask: (batch_size, 10)

import tensorflow as tf

# Based on the issue description and examples, this code reconstructs a typical model 
# that uses two integer tensor inputs of shape (None, 10), named 'input_ids' and 'attention_mask',
# then passes them to a language model (huggingface transformers TFAutoModel),
# then applies a TimeDistributed Dense classifier producing outputs shaped (None, 10, num_classes).

# Key issue faced in the original problem was saved model failing on inference because of
# the saved signature expecting additional arguments (training=True/False) but
# the loaded model call did not match signatures.

# This causes ValueError around function signatures for saved and loaded model calls.

# To fix this we write a subclassed tf.keras.Model that accepts the inputs as a tuple of two tensors
# plus a 'training' boolean keyword argument with default False,
# which aligns with the signature expected by the original Transformer models.

# Also, implement call with signatures accepting 'training' argument so saving/loading works robustly.

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=2, max_seq_len=10):
        super().__init__()
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len

        # Define inputs as placeholders here inside __init__ purely for shape reference
        # Actual inputs given at call, so these Input layers won't be used functionally.
        self.input_ids = tf.keras.layers.Input(shape=(self.max_seq_len,), dtype=tf.int32, name='input_ids')
        self.attention_mask = tf.keras.layers.Input(shape=(self.max_seq_len,), dtype=tf.int32, name='attention_mask')

        # Load a huggingface transformer language model inside the keras model
        # Note: Huggingface models must be passed inputs as a dict or tuple of tensors accordingly.
        from transformers import TFAutoModel

        self.language_model = TFAutoModel.from_pretrained("bert-base-uncased")

        # TimeDistributed Dense layer as classifier on top of language model outputs
        self.classifier = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.num_classes, name='classifier')
        )

    # Call method accepts tuple of inputs (input_ids, attention_mask) and training bool
    def call(self, inputs, training=False):
        # inputs comes as tuple of (input_ids, attention_mask)
        # Huggingface TFAutoModel expects dict or list inputs with named parameters
        input_ids, attention_mask = inputs

        # Forward pass through transformer model with inputs and training flag
        # The model returns a tuple where first element is sequence output (batch_size, seq_len, hidden_size)
        lang_outputs = self.language_model(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           training=training)
        seq_output = lang_outputs[0]  # sequence output

        # Pass sequence output to classifier
        logits = self.classifier(seq_output)

        return logits

def my_model_function():
    # Return an instance of MyModel with default params matching input shapes from issue
    return MyModel(num_classes=2, max_seq_len=10)

def GetInput():
    # Return input tuple matching expected input to MyModel.call()
    # Note: as per issue, inputs are int32 tensors of shape (batch_size, 10)
    batch_size = 1  # example batch size for test input
    max_seq_len = 10

    input_ids = tf.random.uniform(shape=(batch_size, max_seq_len), maxval=10000, dtype=tf.int32)
    attention_mask = tf.ones(shape=(batch_size, max_seq_len), dtype=tf.int32)

    return (input_ids, attention_mask)

