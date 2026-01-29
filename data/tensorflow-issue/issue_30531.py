# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32) ← Input shape inferred from use of VGG16 with input_shape=(224,224,3)

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.vgg16 import VGG16

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Build the base VGG16 model, without the top classification layers, weights from imagenet
        self.base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        
        # Additional classification head
        self.flatten = layers.Flatten(name="flatten")
        self.fc1 = layers.Dense(1024, activation='relu')
        self.dropout1 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(512, activation='relu')
        self.dropout2 = layers.Dropout(0.3)
        self.output_layer = layers.Dense(6, activation='softmax', name="output")
        
    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=training)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        output = self.output_layer(x)
        return output

def my_model_function():
    """
    Return an instance of MyModel compiled with categorical_crossentropy loss and RMSProp optimizer.
    Note: In TF 2.x, tf.keras.optimizers.RMSprop is used instead of tf.train.RMSPropOptimizer.
    """
    model = MyModel()
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model

def GetInput():
    """
    Return a random tensor input matching MyModel's expected input shape:
    Batch size 8 here is arbitrary for testing.
    """
    # Use batch size 8, height 224, width 224, channels 3, dtype float32
    return tf.random.uniform((8, 224, 224, 3), dtype=tf.float32)

