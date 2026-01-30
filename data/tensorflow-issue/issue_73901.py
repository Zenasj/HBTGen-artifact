from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

# remove the line below or comment it out to reproduce training bug
import tensorflow as tf

def build_and_train_model(X_train, y_train, X_test, y_test, num_features, num_classes):
    model = Sequential([
        Input(shape=(num_features,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        X_train, 
        y_train, 
        epochs=300, 
        batch_size=128, 
        validation_data=(X_test, y_test),
        verbose=1
    )
    return model, history

# Build and train model
model, history = build_and_train_model(X_train, y_train, X_test, y_test, X_scaled.shape[1], num_classes)