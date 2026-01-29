# tf.random.uniform((BATCH_SIZE, MAX_LEN), dtype=tf.int32), and dict of tf.string lists for meta inputs

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from transformers import TFDistilBertModel, DistilBertTokenizer

MAX_LEN = 20

STRING_CATEGORICAL_COLUMNS = [
    "Organization",
    "Sector",
    "Content_type",
    "Geography",
    "Themes",
]

VOCAB = {
    "Organization": ["BNS", "FED", "ECB"],
    "Sector": ["BANK", "ASS", "MARKET"],
    "Content_type": ["LAW", "NOTES", "PAPER"],
    "Geography": ["UK", "FR", "DE", "CH", "US", "ES", "NA"],
    "Themes": ["A", "B", "C", "D", "E", "F", "G"],
}

DIM = {
    "Organization": 7,
    "Sector": 2,
    "Content_type": 3,
    "Geography": 4,
    "Themes": 5,
}


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Load DistilBert pretrained model
        self.bert_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased", name="tfbert")

        # Meta branch layers: for each column, build a lookup + embedding + pooling
        self.string_lookups = {}
        self.embeddings = {}
        self.global_pools = {}
        for key in VOCAB:
            vocab_list = VOCAB[key]
            vocab_size = len(vocab_list)
            embed_dim = DIM[key]

            # String lookup layer with mask token "PAD" and 1 OOV index for unknown
            self.string_lookups[key] = layers.StringLookup(
                vocabulary=vocab_list,
                num_oov_indices=1,
                mask_token="PAD",
                output_mode='int',
                name="lookup_" + key,
            )

            # Embedding layer including PAD + NA tokens
            self.embeddings[key] = layers.Embedding(
                input_dim=vocab_size + 2,  # +2 for PAD and NA tokens
                output_dim=embed_dim,
                mask_zero=True,
                name="embedding_" + key,
            )

            self.global_pools[key] = layers.GlobalAveragePooling1D(name="poolembedding_" + key)

        # Dense layers after concatenation
        self.concat_meta = layers.Concatenate(name="concatenate_meta")
        self.concat_all = layers.Concatenate(name="concatenate_all")
        self.dense1 = layers.Dense(128, activation="relu", name="dense")
        self.class_output = layers.Dense(4, name="class_output")

    def call(self, inputs, training=False):
        # inputs: dict with keys for bert inputs and meta inputs
        # Expected keys:
        # For BERT branch: "input_ids", "attention_mask" (int32 tensors shape (batch, MAX_LEN))
        # For meta branch: STRING features, each shape (batch, variable length sequence)

        # BERT branch forward
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # DistilBERT model returns sequence_output; we take CLS token embedding as [0] index along seq dim
        bert_outputs = self.bert_model(input_ids, attention_mask=attention_mask, training=training)
        sequence_output = bert_outputs.last_hidden_state  # shape (batch, seq_len, hidden_size)
        cls_embedding = sequence_output[:, 0, :]  # shape (batch, hidden_size)

        # Meta branch forward
        meta_embeddings = []
        for key in VOCAB:
            x = inputs[key]  # string tensor (batch, seq_len_meta)
            # String lookup converts string tokens to int indices
            x_i = self.string_lookups[key](x)
            # Embedding for tokens
            x_e = self.embeddings[key](x_i)
            # Pool along sequence dimension (variable length)
            x_p = self.global_pools[key](x_e)
            meta_embeddings.append(x_p)

        meta_concat = self.concat_meta(meta_embeddings)

        # Combine branches
        combined = self.concat_all([cls_embedding, meta_concat])
        x = self.dense1(combined)
        logits = self.class_output(x)

        return logits


def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile the model with appropriate loss and optimizer,
    # matching the original example for classification
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    return model


def GetInput():
    """
    Returns a dict input compatible with MyModel.call().
    Shapes:
    - input_ids: (batch, MAX_LEN) int32 tensor of token ids (random within vocab size)
    - attention_mask: (batch, MAX_LEN) int32 tensor (ones or random 0/1)
    - For each meta feature column: (batch, variable_length) tf.string tensor
    We use batch size 5 as in original example, and variable meta sequence lengths.
    """

    batch_size = 5

    # Create random int32 token ids in DistilBERT vocab size (30522)
    # In practice tokenizer encodes real tokens, but here random is OK for testing
    vocab_size_bert = 30522
    input_ids = tf.random.uniform(
        (batch_size, MAX_LEN), minval=0, maxval=vocab_size_bert, dtype=tf.int32
    )
    # attention_mask: 1s
    attention_mask = tf.ones((batch_size, MAX_LEN), dtype=tf.int32)

    # For meta features: simulate batches of sequences of strings with vocab from VOCAB keys,
    # each with some padding "PAD" and "NA" possible.
    dict_meta = {}
    for key in VOCAB:
        vocab_list = VOCAB[key] + ["PAD", "NA"]
        # simulate sequence length for that feature (random 1 to 3)
        seq_len = 3
        # generate batch_size sequences of strings from vocab_list
        import numpy as np

        # Random selection of strings from vocab including PAD and NA
        np.random.seed(0)  # for reproducibility
        samples = np.random.choice(vocab_list, size=(batch_size, seq_len))
        # Convert to tf.string tensor
        dict_meta[key] = tf.constant(samples)

    # Package inputs in a single dict as expected by MyModel
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    inputs.update(dict_meta)

    return inputs

