from tensorflow import keras
from tensorflow.keras import optimizers

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, Model, layers
from transformers import DistilBertTokenizer, TFDistilBertModel


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


# BERT branch
tf_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased", name="tfbert")

input_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")
attention_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="attention_mask")


embedding = tf_model(input_ids, attention_mask=attention_mask)[0][:, 0]

bert_input = {"input_ids": input_ids, "attention_mask": attention_mask}
model_bert = Model(inputs=[bert_input], outputs=[embedding])


# meta branch
meta_inputs = {}
meta_prepocs = []

for key in VOCAB:
    inputs = Input(shape=(None,), dtype=tf.string, name=key)
    meta_inputs[key] = inputs

    vocab_list = VOCAB[key]
    vocab_size = len(vocab_list)
    embed_dim = DIM[key]

    x = layers.experimental.preprocessing.StringLookup(
        vocabulary=vocab_list, num_oov_indices=1, mask_token="PAD", name="lookup_" + key
    )(inputs)

    x = layers.Embedding(
        input_dim=vocab_size + 2,  # 2 = PAD + NA
        output_dim=embed_dim,
        mask_zero=True,
        name="embedding_" + key,
    )(x)

    x = layers.GlobalAveragePooling1D(
        data_format="channels_last", name="poolembedding_" + key
    )(x)

    meta_prepocs.append(x)

meta_output = layers.concatenate(meta_prepocs, name="concatenate_meta")
model_meta = Model(meta_inputs, meta_output)


# combining branches
combined = layers.concatenate(
    [model_bert.output, model_meta.output], name="concatenate_all"
)
ouput = layers.Dense(128, activation="relu", name="dense")(combined)
ouput = layers.Dense(4, name="class_output")(ouput)
model = Model(inputs=[model_bert.input, model_meta.input], outputs=ouput)

model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

# input meta
dict_meta = {
    "Organization": [
        ["BNS", "NA"],
        ["ECB", "PAD"],
        ["NA", "PAD"],
        ["NA", "PAD"],
        ["NA", "PAD"],
    ],
    "Sector": [
        ["BANK", "PAD", "PAD"],
        ["ASS", "PAD", "NA"],
        ["MARKET", "NA", "NA"],
        ["NA", "PAD", "NA"],
        ["NA", "PAD", "NA"],
    ],
    "Content_type": [
        ["NOTES", "PAD"],
        ["PAPER", "UNK"],
        ["LAW", "PAD"],
        ["LAW", "PAD"],
        ["LAW", "NOTES"],
    ],
    "Geography": [
        ["UK", "FR"],
        ["DE", "CH"],
        ["US", "ES"],
        ["ES", "PAD"],
        ["NA", "PAD"],
    ],
    "Themes": [["A", "B"], ["B", "C"], ["C", "PAD"], ["C", "PAD"], ["G", "PAD"]],
}

# input text
list_text = [
    "Trump in denial over election defeat as Biden gears up to fight Covid",
    "Feds seize $1 billion in bitcoins they say were stolen from Silk Road",
    "Kevin de Bruyne misses penalty as Manchester City and Liverpool draw",
    "United States nears 10 million coronavirus cases",
    "Fiji resort offers the ultimate in social distancing",
]

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
params = {
    "max_length": MAX_LEN,
    "padding": "max_length",
    "truncation": True,
}
tokenized = tokenizer(list_text, **params)
dict_text = tokenized.data

#input label
label = [[1], [0], [1], [0], [1]]

ds_meta = tf.data.Dataset.from_tensor_slices((dict_meta))
ds_meta = ds_meta.batch(5)
example_meta = next(iter(ds_meta))

ds_text = tf.data.Dataset.from_tensor_slices((dict_text))
ds_text = ds_text.batch(5)
example_text = next(iter(ds_text))

ds_label = tf.data.Dataset.from_tensor_slices((label))
ds_label = ds_label.batch(5)
example_label = next(iter(ds_label))

model.fit([example_text, example_meta], example_label)

ds = tf.data.Dataset.from_tensor_slices(
    (
        {
            "attention_mask": dict_text["attention_mask"],
            "input_ids": dict_text["input_ids"],
            "Content_type": dict_meta["Organization"],
            "Geography": dict_meta["Geography"],
            "Organization": dict_meta["Organization"],
            "Sector": dict_meta["Sector"],
            "Themes": dict_meta["Themes"],
        },
        {"class_output": label},
    )
)


ds = ds.batch(5)
model.fit(ds, epochs=1)