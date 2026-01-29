# tf.random.uniform((1, 128), dtype=tf.int32) ← Input is token ids for sequence length 128, batch size 1

import tensorflow as tf
import numpy as np
from transformers import TFDistilBertModel, DistilBertTokenizer, DistilBertConfig
from typing import Dict

class MyModel(tf.keras.Model):
    """
    A fused model encapsulating:
    - The original HuggingFace TFDistilBertModel (with return_dict=True)
    - Simulated converted output logic

    The call() method returns a dictionary with keys:
    'original_output': Tensor output from TFDistilBertModel's last_hidden_state
    'converted_output': Simulated converted model output (for demonstration)
    'difference': Absolute difference between original and converted outputs

    This reflects the reported accuracy mismatch issue on conversion.
    """

    def __init__(self):
        super().__init__()
        # Load pre-trained DistilBert model (multilingual)
        self.distilbert = TFDistilBertModel.from_pretrained('distilbert-base-multilingual-cased', return_dict=True)
        # Initialize tokenizer to generate input optionally if needed
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Provide dummy inputs required to build the model.
        Here, input_ids tensor shape: (1, 128) of ints.
        """
        return {"input_ids": tf.zeros([1, 128], dtype=tf.int32)}

    def call(self, inputs, training=False):
        """
        Forward pass.
        Args:
            inputs: dict with 'input_ids' Tensor of shape (batch_size, 128)
        Returns:
            dict with
            - 'original_output': last_hidden_state from TFDistilBertModel (shape (1,128,768))
            - 'converted_output': simulated converted model output to reflect the reported output mismatch
            - 'difference': abs(original_output - converted_output)
        """
        # Get original model output
        outputs = self.distilbert(inputs, training=training)
        original_last_hidden = outputs.last_hidden_state  # shape: (1,128,768), float32

        # Simulate converted output described in the issue:
        # The converted output was numerically different, with higher values, so for demonstration,
        # create a tensor that differs by adding a constant or scaled noise
        # Since we do not have the actual converted TFLite model inference here,
        # we'll simulate "converted output" by adding a positive bias + some noise

        # Create a random but consistent tensor similar shape to original with an offset
        # Using a fixed seed to keep reproducibility
        tf.random.set_seed(42)
        noise = tf.random.uniform(tf.shape(original_last_hidden), minval=0.1, maxval=0.4, dtype=tf.float32)
        offset = 0.1
        converted_simulated = original_last_hidden + noise + offset

        # Compute absolute difference
        difference = tf.abs(original_last_hidden - converted_simulated)

        return {
            'original_output': original_last_hidden,
            'converted_output': converted_simulated,
            'difference': difference
        }


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    """
    Returns a dummy input tensor dictionary that matches the input expected by MyModel.
    Specifically, a dict with 'input_ids': tf.Tensor of shape (1, 128) integers.

    To keep consistent with the tokenizer used in the original code,
    we generate a random token id array tensor with typical vocab indices.
    """
    # Assuming the DistilBert vocab size of multilingual-cased is ~ 119547 (Huggingface DistilBert vocab):
    # For safe side, use max token id as 119546
    batch_size = 1
    seq_length = 128
    max_token_id = 119546

    # Generate random input token ids between 0 and max_token_id
    input_ids = tf.random.uniform(
        shape=(batch_size, seq_length),
        minval=0,
        maxval=max_token_id,
        dtype=tf.int32
    )

    return {"input_ids": input_ids}

