# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        weight_decay = 1e-4
        num_classes = 10
        
        # Define layers as in base_model_v2 Sequential
        self.conv1 = tf.keras.layers.Conv2D(
            32, (3,3), padding='same', 
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay), 
            input_shape=(32, 32, 3))
        self.act1 = tf.keras.layers.Activation('elu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.conv2 = tf.keras.layers.Conv2D(
            32, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self.act2 = tf.keras.layers.Activation('elu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        
        self.conv3 = tf.keras.layers.Conv2D(
            64, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self.act3 = tf.keras.layers.Activation('elu')
        self.bn3 = tf.keras.layers.BatchNormalization()
        
        self.conv4 = tf.keras.layers.Conv2D(
            64, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self.act4 = tf.keras.layers.Activation('elu')
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        
        self.conv5 = tf.keras.layers.Conv2D(
            128, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self.act5 = tf.keras.layers.Activation('elu')
        self.bn5 = tf.keras.layers.BatchNormalization()
        
        self.conv6 = tf.keras.layers.Conv2D(
            256, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self.act6 = tf.keras.layers.Activation('elu')
        self.bn6 = tf.keras.layers.BatchNormalization()
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.dropout3 = tf.keras.layers.Dropout(0.4)
        
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.act1(x)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x, training=training)
        x = self.pool1(x)
        x = self.dropout1(x, training=training)
        
        x = self.conv3(x)
        x = self.act3(x)
        x = self.bn3(x, training=training)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.bn4(x, training=training)
        x = self.pool2(x)
        x = self.dropout2(x, training=training)
        
        x = self.conv5(x)
        x = self.act5(x)
        x = self.bn5(x, training=training)
        x = self.conv6(x)
        x = self.act6(x)
        x = self.bn6(x, training=training)
        x = self.pool3(x)
        x = self.dropout3(x, training=training)
        
        x = self.flatten(x)
        x = self.fc(x)
        return x

def my_model_function():
    # Return a compiled model instance compatible with TF 2.x
    model = MyModel()
    # Use Adam optimizer with learning rate 0.0001 as in original
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a random input tensor with shape (batch_size, 32, 32, 3), normalized [0,1] float32
    # Batch size can be arbitrary; use 1 for simplicity
    batch_size = 1
    input_tensor = tf.random.uniform(
        (batch_size, 32, 32, 3), minval=0, maxval=1, dtype=tf.float32
    )
    return input_tensor

