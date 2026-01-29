# tf.random.uniform((256, 128, 1), dtype=tf.float32) ← input shape (batch_size=256, sequence_length=128, channels=1)

import tensorflow as tf

RNN_SEQ_LEN = 128     # Sequence length for LSTM input
L_AMOUNT = 2          # Number of output labels/classes

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        DROPOUT = 0.2
        
        # Define the 3-layer stacked LSTM architecture replicating the Sequential model from issue
        self.lstm1 = tf.keras.layers.LSTM(
            units=RNN_SEQ_LEN, 
            return_sequences=True, 
            input_shape=(RNN_SEQ_LEN, 1))
        self.drop1 = tf.keras.layers.Dropout(DROPOUT)
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.lstm2 = tf.keras.layers.LSTM(
            units=RNN_SEQ_LEN, 
            return_sequences=True)
        self.drop2 = tf.keras.layers.Dropout(DROPOUT / 2)
        self.bn2 = tf.keras.layers.BatchNormalization()
        
        self.lstm3 = tf.keras.layers.LSTM(units=RNN_SEQ_LEN)
        self.drop3 = tf.keras.layers.Dropout(DROPOUT)
        self.bn3 = tf.keras.layers.BatchNormalization()
        
        self.dense_out = tf.keras.layers.Dense(
            L_AMOUNT, activation='softmax')

    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.drop1(x, training=training)
        x = self.bn1(x, training=training)
        
        x = self.lstm2(x)
        x = self.drop2(x, training=training)
        x = self.bn2(x, training=training)
        
        x = self.lstm3(x)
        x = self.drop3(x, training=training)
        x = self.bn3(x, training=training)
        
        x = self.dense_out(x)
        return x


def my_model_function():
    """
    Returns an instance of MyModel compiled with 
    Adam optimizer with learning rate and decay as per issue,
    categorical crossentropy loss and accuracy metric.
    """
    model = MyModel()
    
    # Compile the model same as in the issue
    LEARNING_RATE = 0.001
    DECAY = 3e-5
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, decay=DECAY)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    
    return model


def GetInput():
    """
    Generates a random tensor input with shape:
    (BATCH_SIZE=256, SEQUENCE_LENGTH=128, CHANNELS=1)
    matching the model input shape.
    Values are floats in [0, 1).
    """
    BATCH_SIZE = 256
    SEQ_LENGTH = 128
    CHANNELS = 1
    return tf.random.uniform((BATCH_SIZE, SEQ_LENGTH, CHANNELS), dtype=tf.float32)

