from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from transformers import BertTokenizer, TFBertForSequenceClassification, AdamWeightDecay, TFBertModel
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D

from tensorflow.keras.models import Model
from keras_tuner import HyperParameters, RandomSearch

import numpy as np
import tensorflow as tf

import joblib, os
import pandas as pd
import tensorflow as tf
import numpy as np
import pickle

import numpy as np
import tensorflow as tf
from transformers import TFBertModel
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from keras_tuner import RandomSearch
from tensorflow.keras.models import Model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

INPUT_DIR = "../Output/proto_models_rev2"
# Load data
df_train = pd.read_csv(os.path.join(INPUT_DIR, 'train_cleaned.csv'))
df_test = pd.read_csv(os.path.join(INPUT_DIR, 'test_cleaned.csv'))
# df_test = processor.load_data()
X_train = df_train['review']
X_test = df_test['review']
y_train = df_train['polarity']
y_test = df_test['polarity']

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenizing the datasets
X_train_tokens = tokenizer(
    text=list(X_train),
    add_special_tokens=True,
    max_length=100,
    padding='max_length',
    truncation=True,
    return_tensors='tf',
    return_token_type_ids=False,
    return_attention_mask=True
)

X_test_tokens = tokenizer(
    text=list(X_test),
    add_special_tokens=True,
    max_length=100,
    padding='max_length',
    truncation=True,
    return_tensors='tf',
    return_token_type_ids=False,
    return_attention_mask=True
)


# # Define BERT Model
# bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# # Use Hugging Face's AdamWeightDecay optimizer
# optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
# # Compile the model using a standard loss function
# loss_fn = SparseCategoricalCrossentropy(from_logits=True)
# bert_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
# # Model Summary
# bert_model.summary()

# BERT Model Definition
def build_bert_model(hp):
    input_ids = Input(shape=(100,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(100,), dtype=tf.int32, name="attention_mask")

    bert_model = TFBertModel.from_pretrained("bert-base-uncased")
    bert_output = bert_model(input_ids=input_ids, attention_mask=attention_mask)[0]
    
    pooled_output = GlobalAveragePooling1D()(bert_output)
    dense = Dense(units=hp.Int("units", min_value=32, max_value=128, step=32), activation='relu')(pooled_output)
    output = Dense(2, activation='softmax')(dense)
    
    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    
    learning_rate = hp.Choice("learning_rate", values=[1e-5, 2e-5, 3e-5])
    optimizer = Adam(learning_rate=learning_rate, weight_decay_rate=0.01)
    model.compile(optimizer=optimizer, loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    
    return model

# Convert to TensorFlow Tensor
def create_tf_dataset(X, y, batch_size=32):
    input_ids = tf.convert_to_tensor(X["input_ids"])
    attention_mask = tf.convert_to_tensor(X["attention_mask"])
    y = tf.convert_to_tensor(y)
    
    dataset = tf.data.Dataset.from_tensor_slices(({"input_ids": input_ids, "attention_mask": attention_mask}, y))
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# Convert datasets
train_dataset = create_tf_dataset(X_train_tokens, y_train)
val_dataset = create_tf_dataset(X_test_tokens, y_test)

# Hyperparameter tuner
tuner = RandomSearch(
    build_bert_model,
    objective="val_accuracy",
    max_trials=5,
    executions_per_trial=1,
    directory="bert_tuning",
    project_name="bert_sentiment_analysis"
)


# Perform the hyperparameter tuning search
tuner.search(
    train_dataset,
    validation_data=val_dataset,
    epochs=3
)