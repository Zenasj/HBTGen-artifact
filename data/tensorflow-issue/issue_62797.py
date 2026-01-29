# tf.random.uniform((B, 217, 306, 1), dtype=tf.float32) ← inferred from input image shape (217, 306, 1)

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define a CNN model architecture matching the original description
        # Sequential style but implemented as layers here for clarity and subclassing
        self.conv1 = layers.Conv2D(128, (3,3), activation='relu', name='conv2d_1')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D((2,2))
        
        self.conv2 = layers.Conv2D(64, (3,3), activation='relu', name='conv2d_2')
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D((2,2))
        
        self.conv3 = layers.Conv2D(64, (3,3), activation='relu', name='conv2d_3')
        self.bn3 = layers.BatchNormalization()
        
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation='tanh')
        self.dropout1 = layers.Dropout(0.2)
        
        # Output size: 50 (flattened as in original)
        self.dense_out = layers.Dense(50)
        self.dropout_out = layers.Dropout(0.2)
        self.reshape_out = layers.Reshape((50,))  # Output shape (50,)
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        
        x = self.dense_out(x)
        x = self.dropout_out(x, training=training)
        x = self.reshape_out(x)
        return x

    def grad_cam(self, img_tensor):
        """
        Compute Grad-CAM heatmap for a single input image tensor.
        
        Args:
          img_tensor: 4D tensor with shape (1, H, W, C)
        
        Returns:
          heatmap: 2D numpy array with spatial Grad-CAM heatmap
        """
        # Use the output of conv2 layer for Grad-CAM as it's the last conv layer before flatten
        grad_model = Model(inputs=self.input, outputs=self.get_layer('conv2d_2').output if hasattr(self, 'get_layer') else self.conv2)
        
        # If self.get_layer doesn't exist (since subclassed model), create functional model dynamically:
        # We'll create a new Model for grad_cam with input and conv2 output:
        # To avoid complexity, we re-define grad_model here from inputs to conv2 output.
        
        # Because subclassed model lacks get_layer, build grad_model via functional API:
        # We'll create inputs and replicate forward passes until conv2d_2 output:
        inputs = tf.keras.Input(shape=(217, 306, 1), dtype=tf.float32)
        x = self.conv1(inputs)
        x = self.bn1(x, training=False)
        x = self.pool1(x)
        x = self.conv2(x)
        # grad_model outputs conv2 output
        grad_model = Model(inputs=inputs, outputs=x)
        
        with tf.GradientTape() as tape:
            # Enable gradient tracing on conv2 output
            conv_outputs = grad_model(img_tensor)
            tape.watch(conv_outputs)
            
            # Forward pass through full model
            preds = self(img_tensor, training=False)  # shape (1,50)
            
            top_class_index = tf.argmax(preds[0])
            
            # Compute gradients with unconnected gradients handled as zeros
            grads = tape.gradient(preds[:, top_class_index], conv_outputs,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
            
        # Handle case where grads could be None or all zeros
        if grads is None:
            grads = tf.zeros_like(conv_outputs)
        
        # Global average pooling of gradients for each channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape (channels,)
        
        conv_outputs = conv_outputs[0]  # remove batch dim, shape (H, W, channels)
        
        # Multiply each channel in conv_outputs by corresponding gradient importance
        # broadcast pooled_grads over spatial dims
        heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
        
        # Convert to numpy and normalize
        heatmap = heatmap.numpy()
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap) if np.max(heatmap) != 0 else 1e-10
        heatmap /= max_val
        return heatmap


def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Return a random input tensor consistent with model's expected input shape
    # Batch size = 1 for single image inference
    input_shape = (217, 306, 1)  # height=217, width=306, channels=1 as per preprocess crop & padding
    x = tf.random.uniform((1,) + input_shape, dtype=tf.float32)
    return x

