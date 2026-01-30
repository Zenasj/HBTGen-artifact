from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Data Pre omitted

# Define the model
embedding_dim = 100
model = Sequential([
    Input(shape=(maxlen,)),
    Embedding(max_features, embedding_dim),
    Bidirectional(GRU(64, return_sequences=True)),
    Bidirectional(GRU(64)),
    Dense(3, activation='softmax')  # Adjusted for 3 categories
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Adjusted for multi-class classification
              metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
model.fit(train_x, train_y,
                    batch_size=100,
                    epochs=10,
                    validation_data=(val_x, val_y))


# Save the model in the native Keras format
model.save('my_model.keras')

# Load the model
loaded_model = tf.keras.models.load_model('my_model.keras')