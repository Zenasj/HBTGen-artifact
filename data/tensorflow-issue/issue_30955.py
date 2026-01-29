# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)  ← Input shape is (batch, 28, 28, 1) as per example

import tensorflow as tf

NUM_CLASSES = 10
TARGET_LAYER_NAME = 'target_layer_name'

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__(name="MyModel")

        # First Conv2D layer
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), activation='relu')
        
        # Second Conv2D layer named TARGET_LAYER_NAME for extraction
        self.conv_2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), activation='relu', name=TARGET_LAYER_NAME)
        
        # MaxPooling layer
        self.maxpool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        
        # Flatten and output Dense layer with softmax for classification
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')

    def call(self, inputs, **kwargs):
        x = inputs
        # Apply layers sequentially
        for layer in [self.conv_1, self.conv_2, self.maxpool_1, self.flatten, self.dense_1]:
            x = layer(x)
        return x

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([shape[0], NUM_CLASSES])

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    
    # Build the model by running once on dummy input to ensure variables initialized
    dummy_input = tf.random.uniform((1, 28, 28, 1), dtype=tf.float32)
    model(dummy_input)
    
    # Compile with common settings for example purpose
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def GetInput():
    # Returns a batch of random images with shape (batch=4, 28, 28, 1) float32,
    # matching the expected input shape of MyModel.
    return tf.random.uniform((4, 28, 28, 1), dtype=tf.float32)

