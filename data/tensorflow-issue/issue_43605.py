from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

py
import pathlib
import time

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

__SELECT_COLUMN_NAMES = ['age', 'education', 'income_bracket']


def get_train_test_pandas_data():
    # data can be download from: https://www.kaggle.com/uciml/adult-census-income?select=adult.csv
    census = pd.read_csv("adult_data.csv")

    census['income_bracket'] = census['income_bracket'].apply(lambda label: 0 if label == ' <=50K' else 1)
    census = census[__SELECT_COLUMN_NAMES]

    y_labels = census.pop('income_bracket')
    x_data = census

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_labels, test_size=0.3)

    return x_train, x_test, y_train, y_test


def get_feature_columns():
    age = tf.feature_column.numeric_column("age", dtype=tf.int64)
    education = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000),
        dimension=100)

    feat_cols = [age, education]

    return feat_cols


if (tf.__version__ < '2.0'):
    tf.enable_eager_execution()

x_train, _, y_train, _ = get_train_test_pandas_data()

dataset = tf.data.Dataset.from_tensor_slices((dict(x_train), y_train))

dataset = dataset.shuffle(len(x_train)).batch(4)

feat_cols = get_feature_columns()


class mymodel(tf.keras.Model):
    def __init__(self):
        super(mymodel, self).__init__()
        self.layer1 = tf.keras.layers.DenseFeatures(feature_columns=feat_cols)
        self.layer2 = tf.keras.layers.Dense(10, activation='relu')
        self.layer3 = tf.keras.layers.Dense(10, activation='relu')
        self.layer4 = tf.keras.layers.Dense(1, activation='sigmoid')

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


model = mymodel()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(dataset, epochs=1)


__SAVED_MODEL_DIR = './saved_models/census_keras/{}'.format(int(time.time()))
pathlib.Path(__SAVED_MODEL_DIR).mkdir(parents=True, exist_ok=True)

tf.saved_model.save(model, export_dir=__SAVED_MODEL_DIR)

model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(feature_columns=feat_cols),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

py
import tensorflow as tf

# loaded_model = tf.keras.models.load_model("./saved_models/census_keras/1601196783")  # tf.saved_model.load("saved/1")
loaded_model = tf.keras.models.load_model("./saved_models/census_keras/1601196783")

y_pred = loaded_model.call({"age": [35],
                            "education": ["Bachelors"]})
print(y_pred)


y_pred = loaded_model.call({"age": [40],
                            "education": ["Assoc-voc"]})
print(y_pred)

y_pred = loaded_model.call({"age": [[35]], "education": [["Bachelors"]]})
y_pred = loaded_model.call({"age": [[40]], "education": [["Assoc-voc"]]})

y_pred = loaded_model.call({"age": [[35], [40]],  "education": [["Bachelors"], ["Assoc-voc"]]})

a = tf.constant([35])
b = tf.constant([[35]])
c = tf.constant([[35],[40]])
a.shape # TensorShape([1])
b.shape # TensorShape([1, 1])
c.shape  # TensorShape([2, 1])