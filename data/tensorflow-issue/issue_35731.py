# tf.random.uniform((B, 10, 91), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Two stacked LSTMCells as in the original code, with distinct names for each cell
        self.rnn_cells = [
            tf.keras.layers.LSTMCell(512, recurrent_dropout=0, name=f"rnn_cell{i}") 
            for i in range(2)
        ]
        # RNN layer wrapping the stacked LSTM cells
        self.rnn_layer = tf.keras.layers.RNN(self.rnn_cells, return_sequences=True, name="rnn_layer")
        # Dense layer to predict 91 features
        self.pred_feat_layer = tf.keras.layers.Dense(91, name="prediction_features")
        # Softmax layer
        self.softmax = tf.keras.layers.Softmax()
        
    def call(self, inputs, training=False):
        # Forward pass through stacked RNN
        rnn_output = self.rnn_layer(inputs, training=training)
        pred_feat = self.pred_feat_layer(rnn_output)
        pred = self.softmax(pred_feat)
        # Output both softmax and logits as tuple, matching original model outputs
        return pred, pred_feat

def my_model_function():
    # Return an instance of MyModel, no pretrained weights specified
    return MyModel()

def GetInput():
    # Return a random input tensor shaped (batch, time, features)
    # The input shape was (10, 91) (time=10, features=91), batch size assumed 4 as example
    # Use float32 as default dtype for compatibility with the model
    batch_size = 4
    time_steps = 10
    feature_dim = 91
    return tf.random.uniform((batch_size, time_steps, feature_dim), dtype=tf.float32)

