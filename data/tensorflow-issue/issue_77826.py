# tf.random.uniform((B, 100), dtype=tf.int32) for input_ids and attention_mask

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


class MyModel(tf.keras.Model):
    def __init__(self, num_labels=2, units=64, learning_rate=2e-5, **kwargs):
        super().__init__(**kwargs)
        # Initialize pretrained BERT model from HuggingFace transformers
        # We assume from_pretrained will load bert-base-uncased weights.
        self.bert = TFBertModel.from_pretrained("bert-base-uncased", trainable=True)
        # Pooling layer to get a fixed size vector from token embeddings
        self.pooler = GlobalAveragePooling1D()
        # Dense layers for classification
        self.dense1 = Dense(units, activation='relu')
        self.classifier = Dense(num_labels, activation='softmax')  # softmax for multi-class classification
        # Loss & optimizer are standard for classification task
        self.loss_fn = SparseCategoricalCrossentropy(from_logits=False)
        self.optimizer = Adam(learning_rate=learning_rate)

    def call(self, inputs, training=False):
        # Expect inputs = dict with keys 'input_ids' and 'attention_mask'
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            # Defensive fallback: create attention mask of ones if None to avoid None passed to BERT
            attention_mask = tf.ones_like(input_ids)
        # Pass inputs through BERT model
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, training=training)
        sequence_output = bert_outputs.last_hidden_state  # shape (batch, seq_len, hidden_dim)
        pooled_output = self.pooler(sequence_output)  # shape (batch, hidden_dim)
        x = self.dense1(pooled_output)
        logits = self.classifier(x)
        return logits

    def compile(self, **kwargs):
        # Override compile to use the predefined optimizer and loss
        super().compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=["accuracy"], **kwargs)

def my_model_function():
    # Return an instance of MyModel with default hyperparameters suitable for this example
    # Units and learning rate chosen to match the tuning ranges hinted in the issue
    return MyModel(num_labels=2, units=64, learning_rate=2e-5)

def GetInput():
    # Inputs must be a dict with keys `input_ids` and `attention_mask`, 
    # both shaped (batch_size, seq_length), dtype int32.
    batch_size = 3  # batch size used in error logs
    seq_length = 100  # fixed max sequence length used in tokenization
    # Generate uniform random token IDs in vocab range [0, 30522) typical of bert-base-uncased vocab size
    vocab_size = 30522
    input_ids = tf.random.uniform((batch_size, seq_length), minval=0, maxval=vocab_size, dtype=tf.int32)
    attention_mask = tf.ones((batch_size, seq_length), dtype=tf.int32)  # valid attention mask (all tokens attended)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

