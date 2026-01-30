from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras
import pandas as pd

from keras.layers import Input, TextVectorization, Embedding, GlobalAveragePooling1D, Dense
from keras.models import Model

train_df = pd.DataFrame({'review':['This is shit', 'I hate this', 'Great I love this!!!'],
                         'label': [0, 0, 1]})
X_train = train_df['review'].values
y_train = train_df['label'].values

# Define a Functional model:
vectorizer = TextVectorization(max_tokens=20, output_mode='int', output_sequence_length=5, pad_to_max_tokens=True)
vectorizer.adapt(X_train)

text_input = Input(shape=(1,), dtype=tf.string, name='text_input')
vectorizer_text = vectorizer(text_input)
embedding = Embedding(input_dim=20, input_length=5, output_dim=100)(vectorizer_text)
pooled = GlobalAveragePooling1D()(embedding)
output = Dense(1, activation='sigmoid', name='output')(pooled)

# Create the functional model
model = Model(inputs=text_input, outputs=output)

# Compile
optimizer_config = {'class_name': 'rmsprop', 'config': {'lr': 0.0001}}
model.compile(optimizer=tf.keras.optimizers.get(optimizer_config), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10)

# Save the model:
save_path = '/content/'
model.save(save_path + 'my_model.keras')

# Load the model:
loaded_model = tf.keras.models.load_model(save_path + 'my_model.keras')