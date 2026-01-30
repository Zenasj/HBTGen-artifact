from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import re
import string
import logging

import numpy as np
import pandas as pd
from datetime import datetime

from spellchecker import SpellChecker
import tokenization

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


# Logger setup
logging.basicConfig(
    format="%(asctime)s {} %(levelname)s: %(message)s".format(__name__),
    level=logging.DEBUG,
)
logger = logging.getLogger(name=__name__)
logging.getLogger("py4j").setLevel(logging.ERROR)


class ModelBERT(object):
    def __init__(
        self, 
        pre_train_handle, 
        column_text="text", 
        column_target="target", 
        data_prep=False,
        vector_len=200,
        tokenizer=None
    ):
        logger.info("Initialize ModelBERT along with regex pattarens")
        self.pre_train_handle = pre_train_handle
        self.column_text = column_text
        self.column_target = column_target
        self.data_prep = data_prep
        self.vector_len = vector_len
        self.tokenizer = tokenizer
        self.model_weight_path = "model.h5"
        self.func_list = [self.translate_smiley, self.remove_things]

        self.pattern_hashtag = re.compile(pattern=r"#\w+", flags=re.IGNORECASE)
        self.pattern_uppercase_words = re.compile(pattern=r"[A-Z]+")
        self.pattern_lowercase_words = re.compile(pattern=r"[a-z]+")
        self.pattern_url = re.compile(
            pattern=r"\b[.*]?http[s]?[a-zA-Z0-9_:/\.]+\b", 
            flags=re.IGNORECASE
        )
        self.pattern_handle = re.compile(pattern=r"@\w+", flags=re.IGNORECASE)
        self.pattern_smiley_happy = re.compile(pattern=r"[:|;]\)+")
        self.pattern_smiley_sad = re.compile(pattern=r"[:|;]\(+")
        self.pattern_punctuation = re.compile(r"[{}]".format(string.punctuation))


    def translate_smiley(self, df):
        logger.info("Translating smileys")
        df[self.column_text] = df[self.column_text] \
            .apply(lambda t: self.pattern_smiley_happy.sub(repl="happy", string=t)) \
            .apply(lambda t: self.pattern_smiley_sad.sub(repl="sad", string=t))

        return df


    def remove_things(self, df):
        """Remove URL | handles | punctuations"""
        logger.info("Remove URL, handles and punctuations")
        df[self.column_text] = df[self.column_text] \
            .apply(lambda t: self.pattern_url.sub(repl="", string=t)) \
            .apply(lambda t: self.pattern_handle.sub(repl="", string=t)) \
            .apply(lambda t: self.pattern_punctuation.sub(repl="", string=t))
        
        return df


    def correct_spelling(self, df):
        logger.info("Correcting spellings")
        corrector = SpellChecker().correction
        def correct_me(text):
            return " ".join(list(map(corrector, text.split())))
        df[self.column_text] = df[self.column_text].apply(correct_me)
        
        return df


    def setup_pre_train_layer(self):
        logger.info("Setting up pretrain BERT layer")
        self.pre_train_layer = hub.KerasLayer(handle=self.pre_train_handle, trainable=True)
        logger.debug(f"Pre trained layer: {self.pre_train_layer}")


    def setup_tokenizer(self):
        logger.info("Setting up tokenizer")
        self.setup_pre_train_layer()
        
        vocab_file = self.pre_train_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.pre_train_layer.resolved_object.do_lower_case.numpy()

        logger.debug(f"vocab_file: {vocab_file}")
        logger.debug(f"do_lower_case: {do_lower_case}")

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, 
            do_lower_case=do_lower_case
        )


    def encode_text(self, texts):
        if not self.tokenizer:
            self.setup_tokenizer()

        logger.debug(f"Encoding text with vector length: {self.vector_len}")
        all_tokens = []
        all_masks = []
        all_segments = []

        for text in texts:
            text = self.tokenizer.tokenize(text)

            text = text[:self.vector_len - 2]
            input_sequence = ["[CLS]"] + text + ["[SEP]"]
            pad_len = self.vector_len - len(input_sequence)

            tokens = self.tokenizer.convert_tokens_to_ids(input_sequence)
            tokens += [0] * pad_len
            pad_masks = [1] * len(input_sequence) + [0] * pad_len
            segmen_ids = [0] * self.vector_len

            all_tokens.append(tokens)
            all_masks.append(pad_masks)
            all_segments.append(segmen_ids)

        return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


    def create_model(self):
        logger.info("Creating model")
        input_word_ids = Input(shape=(self.vector_len,), name="input_word_ids", dtype=tf.int32)
        input_masks = Input(shape=(self.vector_len,), name="input_masks", dtype=tf.int32)
        segment_ids = Input(shape=(self.vector_len,), name="segment_ids", dtype=tf.int32)

        _, sequence_output = self.pre_train_layer([input_word_ids, input_masks, segment_ids])
        clf_output = sequence_output[:, 0, :]
        outputs = Dense(1, activation="sigmoid")(clf_output)

        self.model = Model(inputs=[input_word_ids, input_masks, segment_ids], outputs=outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=3e-6), 
            loss="binary_crossentropy", 
            metrics=["accuracy"]
        )
        logger.info(self.model.summary())


    def prepare_data(self, df):
        if self.func_list:
            for func in self.func_list:
                logger.debug(f"Applying function: {str(func)}")
                df = func(df)

        return df


    def train_model(self, X, y, batch_size=10, epochs=1, verbose=1, validation_split=0.2):
        if self.data_prep:
            df = self.prepare_data(df)

        X = self.encode_text(X[self.column_text].values)
        y = np.asarray(y)

        logger.debug(f"Model checkpoint: {self.model_weight_path}")
        self.checkpoint = ModelCheckpoint(
            filepath=self.model_weight_path, 
            monitor="val_loss", 
            save_best_only=True
        )

        # log_dir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.create_model()
        logger.info("Training model")
        self.model.fit(
            x=X, 
            y=y, 
            batch_size=batch_size, 
            epochs=epochs, 
            verbose=verbose, 
            callbacks=[self.checkpoint],
            validation_split=validation_split
        )


    def predict(self, X):
        if self.data_prep:
            X = self.prepare_data(X)

        X = self.encode_text(X[self.column_text].values)
        self.model.load_weights(self.model_weight_path)

        logger.info("Making predictions")
        prediction = self.model.predict(x=X)

        return prediction



path_train = "input/nlp-getting-started/train.csv"
path_test = "input/nlp-getting-started/test.csv"

df_train = pd.read_csv(path_train)

X_train, X_test, y_train, y_test = train_test_split(
    df_train.drop("target", axis=1), 
    df_train["target"], 
    test_size=0.1, 
    random_state=999
)

pre_train_handle = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
# pre_train_handle = "/Users/sardarmrinal/Downloads/bert_en_uncased_L-12_H-768_A-12"

model_bert = ModelBERT(pre_train_handle=pre_train_handle)
model_bert.vector_len = 2
# model_bert.data_prep = True
model_bert.func_list = [model_bert.translate_smiley, model_bert.remove_things]

# model_bert.train_model(df_train)
model_bert.train_model(
    X=X_train, 
    y=y_train, 
    batch_size=500, 
    epochs=1, 
    verbose=1, 
    validation_split=0.1
)

y_predict = model_bert.predict(X=X_test)
accuracy = accuracy_score(np.asarray(y_test), y_predict.round().astype(int).reshape(len(X_test)))
print(f"Test accuracy score: {accuracy}")



# Preparing submission
df_eval = pd.read_csv(path_test)

prediction = model_bert.predict(df_eval)

df_sub = pd.DataFrame(
    data={
        "id": df_eval["id"].values,
        "target": prediction.round().astype(int).reshape(df_eval.shape[0])
    }
)

df_sub.to_csv("./submission_bert.csv", header=True, index=False)