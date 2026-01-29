# tf.random.uniform((None, None, None, None, 3), dtype=tf.float32) 
# Assumptions:
# - Input shape is a 5D tensor since the model uses TimeDistributed CNN over frames and images.
# - The innermost dimension channels=3 (typical RGB images).
# - The dimensions: (batch_size, nbr_frame, H, W, C) where nbr_frame = number of frames, H and W are spatial dims.
#   These are not explicitly given, so placeholders are used for unknown frame count and spatial dims.
# - For LSTM input, TimeDistributed expects sequences of frames; nbr_frame is the sequence length.

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class MyModel(tf.keras.Model):
    def __init__(self, input_shape=(None, 224, 224, 3), nbr_frame=10, num_classes=2):
        super().__init__()
        # This model replicates the issue's architecture:
        # - A time-distributed MobileNetV2 CNN (with pretrained Imagenet weights, non-trainable)
        # - A LSTM layer over time steps
        # - TimeDistributed Dense layers, Flatten, Dense layers with Dropout
        
        # Base MobileNetV2 CNN backbone without top layers, frozen weights
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling=None  # no global pooling here, will add pooling after CNN
        )
        base_model.trainable = False

        # CNN to apply over every frame independently
        self.cnn = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2)
        ])
        
        # TimeDistributed CNN over frames
        # input shape to model is expected: (batch_size, nbr_frame, H, W, C)
        # The CNN is applied to each (H, W, C) individually per frame.
        # We freeze base_model by design.
        self.time_distributed_cnn = layers.TimeDistributed(self.cnn, input_shape=(nbr_frame, ) + input_shape)
        
        # LSTM layer processing the temporal sequence output by the CNN
        # nbr_frame sequence length, return_sequences=True to maintain temporal dimension
        self.lstm = layers.LSTM(nbr_frame, return_sequences=True)
        
        # TimeDistributed Dense layer over LSTM outputs (using relu activation)
        self.time_distributed_dense = layers.TimeDistributed(
            layers.Dense(nbr_frame, activation='relu')
        )
        
        # Flatten output across all frames/time steps and features
        self.flatten = layers.Flatten()
        
        # Fully connected layers mimicking filter1, filter2, last dense layers from the original
        self.fc_filter1 = layers.Dense(164, activation='relu', name="filter1")
        self.dropout1 = layers.Dropout(0.2)
        self.fc_filter2 = layers.Dense(24, activation='sigmoid', name="filter2")
        self.dropout2 = layers.Dropout(0.1)
        self.fc_last = layers.Dense(num_classes, activation='sigmoid', name="last")
        
        # Compile parameters: RMSprop optimizer, categorical crossentropy loss & categorical accuracy
        self.optimizer = optimizers.RMSprop()
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.metrics_list = [tf.keras.metrics.CategoricalAccuracy(name='accuracy', dtype=tf.float32)]

    def call(self, inputs, training=False):
        # Forward pass
        x = self.time_distributed_cnn(inputs, training=training)
        x = self.lstm(x, training=training)
        x = self.time_distributed_dense(x, training=training)
        x = self.flatten(x)
        x = self.fc_filter1(x)
        x = self.dropout1(x, training=training)
        x = self.fc_filter2(x)
        x = self.dropout2(x, training=training)
        x = self.fc_last(x)
        return x
    
    def compile(self, **kwargs):
        super().compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=self.metrics_list,
            **kwargs
        )

def my_model_function():
    # Assumptions for input shape:
    # - nbr_frame (number of frames per sequence) = 10 (arbitrary choice from issue)
    # - input_shape is (224, 224, 3) typical MobileNetV2 input size
    nbr_frame = 10
    input_shape = (224, 224, 3)
    num_classes = 2
    
    model = MyModel(input_shape=input_shape, nbr_frame=nbr_frame, num_classes=num_classes)
    model.compile()
    return model

def GetInput():
    # Returns a random tensor matching input shape expected by MyModel:
    # Shape: (batch_size, nbr_frame, H, W, C)
    # Batch size is arbitrary - using 4 here
    batch_size = 4
    nbr_frame = 10
    H, W, C = 224, 224, 3
    
    # Generate random floats in [0,1) matching float32 dtype,
    # suitable as dummy input for model
    return tf.random.uniform(
        shape=(batch_size, nbr_frame, H, W, C),
        minval=0,
        maxval=1,
        dtype=tf.float32
    )

