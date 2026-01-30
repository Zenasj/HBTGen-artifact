import random

# collab:  https://colab.research.google.com/drive/1q13ZwWqgfFcnY8f5oU_KnK3wVf_Gr1JA?usp=sharing
# gist:      https://gist.github.com/moprules/def9b2bda642a064b35e51b8914a28dd

# fast code
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

vocabulary_size = 10000
num_tags = 100
num_departments = 4

# define three model inputs
title = keras.Input(shape=(vocabulary_size,), name="title")
text_body = keras.Input(shape=(vocabulary_size,), name="text_body")
tags = keras.Input(shape=(num_tags,), name="tags")

features = layers.Concatenate()([title, text_body, tags])
# one intermediate layer
features = layers.Dense(64, activation="relu")(features)

# Define two model outputs
priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
department = layers.Dense(num_departments, activation="softmax", name="department")(features)

# set the model
model = keras.Model(inputs=[title, text_body, tags],
                    outputs=[priority, department])
# prepare data
num_samples = 1280
# The data is filled in with zeros and ones
title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

# priority: [0., 1.]
priority_data = np.random.random(size=(num_samples, 1))
# class of 4 labels
department_data = np.random.randint(0, 2, size=(num_samples, num_departments))

# compile model
model.compile(optimizer="rmsprop",
              loss={"priority": "mean_squared_error",
                    "department": "categorical_crossentropy"},
              metrics={"priority": ["mean_absolute_error"],
                       "department": ["accuracy"]})

# It doesn't matter how the model is compiled
# model.compile(optimizer="rmsprop",
#               loss=["mean_squared_error", "categorical_crossentropy"],
#               metrics=[["mean_absolute_error"], ["accuracy"]])


# NOT WORKING
# TRAIN MODEL WITH transferring the DICTIONARY to the method
model.fit({"title": title_data, "text_body": text_body_data, "tags": tags_data},
          {"priority": priority_data, "department": department_data},
          epochs=1
)

# WORK
# TRAIN MODEL WITHOUT transferring the DICTIONARY to the method
model.fit([title_data, text_body_data, tags_data],
          [priority_data, department_data],
          epochs=1
)

# ALSO WORK
# TRAIN MODEL WITH transferring the DICTIONARY to the method
# REPLACE priority and department
model.fit({"title": title_data, "text_body": text_body_data, "tags": tags_data},
          {"priority": department_data, "department": priority_data},
          epochs=1
)

# Define two model outputs
priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
department = layers.Dense(4, activation="softmax", name="department")(features)

# set the model
model = keras.Model(inputs=[title, text_body, tags], outputs=[priority, department])