import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Input, BatchNormalization, Dropout, Flatten, Identity


args = {"activation": "relu",
        "batch_norm": True}

@keras.saving.register_keras_serializable()
class CustomModel1(Model):
    def __init__(self):
        super().__init__()
        self.dense = Dense(32)

    def call(self, inputs):
        x = self.dense(inputs)
        return x


@keras.saving.register_keras_serializable()
class CustomModel2(Model):
    def __init__(self):
        super().__init__()
        self.dense = Dense(32)

    def call(self, inputs):
        x = self.dense(inputs)
        return x


@keras.saving.register_keras_serializable()
class CustomModel3(Model):
    def __init__(self):
        super().__init__()
        self.net1 = CustomModel1()
        self.net2 = CustomModel2()

    def call(self, inputs):
        z = self.net1(inputs)
        x = self.net2(z)
        return z, x

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            # z, y_pred = self(x)                 # this fixes it instead
            y_pred = self.net2(self.net1(x))      # this line throws the error
            loss = self.compiled_loss(y, y_pred)

        trainable_vars = self.trainable_weights
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


# Instantiate the model
model = CustomModel3()

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Create some dummy data for training
x_train = np.random.random((1000, 32))
y_train = np.random.randint(10, size=(1000,))

# Train the model for one epoch
model.fit(x_train, y_train, epochs=1)

# Save the model
model.save('custom_model.keras', save_format='keras')

# Load the model again
loaded_model = tf.keras.models.load_model('custom_model.keras')

# Generate some sample data for prediction
x_sample = np.random.random((10, 32))  # Assuming 10 samples with 32 features each

# Make predictions using the loaded model
predictions = loaded_model.predict(x_sample)
print(predictions)
# Print the predictions
print(model.summary())

loaded_model = tf.keras.models.load_model('custom_model.h5')  # For HDF5 format