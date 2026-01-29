# tf.random.uniform((1024, 224, 224, 3), dtype=tf.float32) ← input shape inferred from dataset and model input layer

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model from issue example: sequential with GlobalMaxPool2D and Dense(1000, softmax)
        self.pool = tf.keras.layers.GlobalMaxPool2D(input_shape=(224, 224, 3))
        self.classifier = tf.keras.layers.Dense(1000, activation='softmax')

    def call(self, inputs):
        x = self.pool(inputs)
        return self.classifier(x)

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile the model similarly to example code
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
    )
    return model

def GetInput():
    # Return a random tensor that matches the expected input shape (batch_size=1024, 224x224 RGB images)
    # dtype float32 for the preprocessed images
    # Input tensor simulates a batch for model inference or training
    return tf.random.uniform((1024, 224, 224, 3), dtype=tf.float32)

