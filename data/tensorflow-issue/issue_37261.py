from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import matplotlib.pyplot as plt
import matplotlib.image as mpimag

img = mpimage.imread("test_image.jpg")
implot = plt.imshow(img)
prob = loaded_model.predict(img)  # load your model

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_text)

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from absl import logging
import numpy as np
import os
import pandas as pd
import seaborn as sns


dir = os.path.dirname(os.path.realpath(__file__))
print(dir)
file_name="traindata.txt"

train_df = {}
test_df = {}
remain = {}

with open (file_name, 'r', encoding="utf8") as l:
  lines = l.readlines()

#my predict data 
with open ('remain.txt', 'r', encoding="utf8" ) as l:
    remain_data = l.readlines()

remain["html"] = [i.lower() for i in remain_data]

#Split data into train and test 
train_df["html"] = [i.lower() for i in lines[:1100]]
test_df["html"] = [i.lower() for i in lines[1100:]]

#Split labels into train and test
with open ('trainlabel.txt', 'r') as l:
  labels =  l.readlines()
  labels = [int(i) for i in labels]
  train_df["polarity"] = labels[:1100]
  test_df["polarity"] = labels[1100:]


train_df = pd.DataFrame.from_dict(train_df)
test_df = pd.DataFrame.from_dict(test_df)
remain_df = pd.DataFrame.from_dict(remain)

# Reduce logging output.
logging.set_verbosity(logging.ERROR)


# Training input on the whole training set with no limit on training epochs.
train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(train_df, train_df["polarity"], num_epochs=None, shuffle=True)

# Prediction on the whole training set.
predict_train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(train_df, train_df["polarity"], shuffle=False)

# Prediction on the test set.
predict_test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(test_df, test_df["polarity"], shuffle=False)

# Prediction on the remain  set.
predict_remain_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(test_df, test_df["polarity"], shuffle=False)

embedded_text_feature_column = hub.text_embedding_column(
    key="html",
    module_spec="https://tfhub.dev/google/random-nnlm-en-dim128/1")

estimator = tf.estimator.DNNClassifier(
    hidden_units=[64, 32],
    feature_columns=[embedded_text_feature_column],
    n_classes=2,
    optimizer=tf.keras.optimizers.Adagrad(lr=0.003))


estimator.train(input_fn=train_input_fn, steps=5000)

train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)


print("Training set accuracy: {accuracy}".format(**train_eval_result))  ## Ressult %99,
print("Test set accuracy: {accuracy}".format(**test_eval_result))       ## Result ~ %79,

Prediction = estimator.predict(input)

Prediction_Classes = tf.argmax(Prediction, axis=1)

for Input_File_Name, Pred_Class in zip(Input_Files, Prediction_Classes):
    print('Prediction for {} is {}'.format(Input_File_Name, Labels_List[(Pred_Class.numpy()-1)]))