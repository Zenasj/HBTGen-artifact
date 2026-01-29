# tf.random.uniform((B, 256, 1600, 3), dtype=tf.float32)

import tensorflow as tf
from tensorflow.keras import layers, backend as K


# Assumptions and reasoning:
# - Input shape inferred from DataGenerator dimensions: (256, 1600, 3)
# - Output mask shape is (256, 1600, 4) for 4 defect classes (multilabel segmentation)
# - Model described is a UNet-like for multilabel segmentation
# - Since TPU support issues with custom Sequence generators, typical approach is to rewrite 
#   dataset with tf.data (not included here as user code)
# - We'll create a basic UNet model suitable for (256,1600,3) input and 4-class sigmoid output mask.
# - Provide soft dice loss and metric inside model scope.
# - Output is segmentation mask of shape (256, 1600, 4), float32 between 0 and 1 (sigmoid)

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Basic UNet-like architecture adapted for TPU compatibility and large input width.
        # Using simple conv blocks and down/upsampling.
        
        # Encoder
        self.conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')
        self.conv1b = layers.Conv2D(32, 3, activation='relu', padding='same')
        self.pool1 = layers.MaxPooling2D((2, 2))
        
        self.conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')
        self.conv2b = layers.Conv2D(64, 3, activation='same', activation='relu')
        self.pool2 = layers.MaxPooling2D((2, 2))
        
        self.conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')
        self.conv3b = layers.Conv2D(128, 3, activation='relu', padding='same')
        self.pool3 = layers.MaxPooling2D((2, 2))
        
        self.conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')
        self.conv4b = layers.Conv2D(256, 3, activation='relu', padding='same')
        
        # Decoder
        self.up3 = layers.UpSampling2D((2, 2))
        self.conv5 = layers.Conv2D(128, 3, activation='relu', padding='same')
        self.conv5b = layers.Conv2D(128, 3, activation='relu', padding='same')
        
        self.up2 = layers.UpSampling2D((2, 2))
        self.conv6 = layers.Conv2D(64, 3, activation='relu', padding='same')
        self.conv6b = layers.Conv2D(64, 3, activation='relu', padding='same')
        
        self.up1 = layers.UpSampling2D((2, 2))
        self.conv7 = layers.Conv2D(32, 3, activation='relu', padding='same')
        self.conv7b = layers.Conv2D(32, 3, activation='relu', padding='same')
        
        # Output layer with sigmoid activation for multilabel segmentation
        self.out_conv = layers.Conv2D(4, 1, activation='sigmoid', padding='same')
        
    def call(self, inputs, training=False):
        # Encoder path
        c1 = self.conv1(inputs)
        c1 = self.conv1b(c1)
        p1 = self.pool1(c1)
        
        c2 = self.conv2(p1)
        c2 = self.conv2b(c2)
        p2 = self.pool2(c2)
        
        c3 = self.conv3(p2)
        c3 = self.conv3b(c3)
        p3 = self.pool3(c3)
        
        c4 = self.conv4(p3)
        c4 = self.conv4b(c4)
        
        # Decoder path + skip connections
        u3 = self.up3(c4)
        u3 = tf.concat([u3, c3], axis=-1)
        c5 = self.conv5(u3)
        c5 = self.conv5b(c5)
        
        u2 = self.up2(c5)
        u2 = tf.concat([u2, c2], axis=-1)
        c6 = self.conv6(u2)
        c6 = self.conv6b(c6)
        
        u1 = self.up1(c6)
        u1 = tf.concat([u1, c1], axis=-1)
        c7 = self.conv7(u1)
        c7 = self.conv7b(c7)
        
        output = self.out_conv(c7)
        return output


def my_model_function():
    # Instantiate model and compile with soft dice loss and metric
    model = MyModel()
    
    def soft_dice_loss(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = 2 * K.sum(y_true_f * y_pred_f) + 1e-9
        denominator = K.sum(y_true_f ** 2) + K.sum(y_pred_f ** 2) + 1e-9
        return 1 - (intersection / denominator)
    
    def soft_dice_coeff(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = 2 * K.sum(y_true_f * y_pred_f) + 1e-9
        denominator = K.sum(K.abs(y_true_f)) + K.sum(K.abs(y_pred_f)) + 1e-9
        return K.mean(intersection / denominator)
    
    model.compile(optimizer='adam', loss=soft_dice_loss, metrics=[soft_dice_coeff])
    return model


def GetInput():
    # Generate a random tensor input matching the model input:
    # batch size = 4 (typical for TPU, but any batch is fine here),
    # height=256, width=1600, channels=3
    return tf.random.uniform((4, 256, 1600, 3), dtype=tf.float32)

