# tf.random.uniform((1, 224, 224, 3), dtype=tf.float32) ← Inferred input shape from original VGG19 model instantiation (BATCH_SIZE=1, IMG_SIZE=224, CHANNELS=3)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# Constants based on issue details
IMG_SIZE = 224
CHANNELS = 3
BATCH_SIZE = 1
LR_2 = 0.003
EPOCHS = 2

# Number of output labels inferred from final Dense layer param count: 44
LABELS = 44

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Emulate the original VGG19 Sequential architecture with same layers and parameters
        self.model = Sequential([
            Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
            Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            
            Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
            Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            
            Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
            Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
            Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
            Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            
            Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'),
            Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'),
            Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'),
            Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            
            Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'),
            Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'),
            Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'),
            Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(LABELS, activation='sigmoid'),
        ])

        # Use Adam optimizer with given learning rate and decay per original code
        # Decay = LR_2 / EPOCHS
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR_2, decay=LR_2 / EPOCHS)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.accuracy_metric = tf.keras.metrics.BinaryAccuracy()

    def call(self, inputs, training=False):
        # Forward pass
        x = self.model(inputs, training=training)
        return x

    def compile(self, **kwargs):
        # Override compile to set optimizer, loss, metrics
        super().compile(optimizer=self.optimizer,
                        loss=self.loss_fn,
                        metrics=[self.accuracy_metric],
                        **kwargs)

def my_model_function():
    # Return an instance of MyModel with compilation done as per original
    model = MyModel()
    model.compile()  # set optimizer, loss, metrics
    return model

def GetInput():
    # Return a random tensor input that matches the model input shape (BATCH_SIZE, 224, 224, 3)
    return tf.random.uniform((BATCH_SIZE, IMG_SIZE, IMG_SIZE, CHANNELS), dtype=tf.float32)

