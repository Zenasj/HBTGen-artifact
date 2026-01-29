# tf.random.uniform((B, 160, 160, 3), dtype=tf.float32) ← Inferred input shape is (None, 160, 160, 3) from model summary and dataset preprocessing

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ])
        
        # Preprocess input to ResNet50 expected range [-1,1]
        # This preprocess_input from mobilenet_v2 was used in example, 
        # but to be true to original the call from ResNet is possible 
        self.preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        
        # Base model: ResNet50 (imagenet weights, include_top=False, output feature map)
        # Trainable set dynamically outside or inside model
        self.base_model = tf.keras.applications.ResNet50(
            input_shape=(160,160,3),
            include_top=False,
            weights='imagenet')
        
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.prediction_layer = tf.keras.layers.Dense(3)  # for 3 classes
        
    def call(self, inputs, training=False):
        x = self.data_augmentation(inputs, training=training)
        x = self.preprocess_input(x)
        x = self.base_model(x, training=training)  # Pass training arg to base model
        x = self.global_average_layer(x)
        x = self.dropout(x, training=training)
        outputs = self.prediction_layer(x)
        return outputs


def my_model_function():
    # Create an instance of MyModel
    model = MyModel()
    
    # To mirror original use: base_model.trainable = True during training
    # You can toggle trainability if training is desired:
    model.base_model.trainable = True
    
    # Compile model for completeness (optimizer/loss can be adapted to your needs)
    base_learning_rate = 0.0001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def GetInput():
    # Return a random float32 tensor with expected input shape (B, 160, 160, 3)
    # Batch size arbitrary; use 1 here for simplicity
    return tf.random.uniform((1, 160, 160, 3), dtype=tf.float32)

