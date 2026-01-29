# tf.random.uniform((B, 60, 9), dtype=tf.float32) ← Input shape inferred: batch_size x 60 time steps x 9 features
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # LSTM layers as per the issue's original sequential model architecture
        self.lstm1 = tf.keras.layers.LSTM(units=100, return_sequences=True)
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.lstm2 = tf.keras.layers.LSTM(units=100, return_sequences=False)
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        
        # Dense layers: 50 units + output unit
        self.dense1 = tf.keras.layers.Dense(units=50)
        self.dense2 = tf.keras.layers.Dense(units=1)
        
    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        if training:
            x = self.dropout1(x, training=training)
        else:
            x = self.dropout1(x, training=False)
        
        x = self.lstm2(x)
        if training:
            x = self.dropout2(x, training=training)
        else:
            x = self.dropout2(x, training=False)
        
        x = self.dense1(x)
        output = self.dense2(x)
        return output

def my_model_function():
    # Instantiate the model, no pre-trained weights available from the issue context
    # User can train further if needed
    return MyModel()

def GetInput():
    # Returns a random tensor with shape (batch_size, 60, 9)
    # Batch size chosen as 16 arbitrarily for typical training/testing
    # Features count is 9 based on features: ['SMA_20', 'EMA_20', 'SMA_50', 'EMA_50', 'RSI', 
    # 'Volatility', 'ROC', 'MACD', 'Signal_Line'] from the code
    batch_size = 16
    time_steps = 60
    feature_dim = 9
    return tf.random.uniform(shape=(batch_size, time_steps, feature_dim), dtype=tf.float32)

