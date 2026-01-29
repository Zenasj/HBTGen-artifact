# tf.random.uniform((32, 10), dtype=tf.int32) → input shape (batch_size=32, seq_length=10), integer token IDs for embedding input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=1000, output_dim=64)
        self.lstm = tf.keras.layers.LSTM(128, return_sequences=True)
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

        # Custom metric implementation similar to the workaround from the issue
        # This metric calculates in_top_k accuracy for sparse targets in sequence outputs,
        # flattening the predictions and labels to rank 2 and 1 respectively to avoid the shape mismatch.
        self.metric = InTopK(k=5)  # default k=5 as SparseTopKCategoricalAccuracy uses top-5

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.lstm(x)
        x = self.dense(x)
        return x

    def compute_metric(self, y_true, y_pred):
        # Using the custom InTopK metric instance to compute metric value
        self.metric.reset_states()
        self.metric.update_state(y_true, y_pred)
        return self.metric.result()

class InTopK(tf.keras.metrics.Mean):
    def __init__(self, k, name='in_top_k', **kwargs):
        super(InTopK, self).__init__(name=name, **kwargs)
        self._k = k

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Flatten y_true to shape [batch_size * seq_length]
        y_true_flat = tf.reshape(tf.cast(y_true, tf.int32), [-1])  
        # Flatten y_pred to [batch_size * seq_length, num_classes]
        y_pred_flat = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
        
        # Compute in_top_k matching bool tensor
        matches = tf.nn.in_top_k(predictions=y_pred_flat, targets=y_true_flat, k=self._k)
        matches_float = tf.cast(matches, self._dtype)

        return super(InTopK, self).update_state(matches_float, sample_weight=sample_weight)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate random integer input matching the model's expected input shape:
    # batch_size = 32, sequence length = 10, integers [0, 1000)
    return tf.random.uniform((32, 10), maxval=1000, dtype=tf.int32)

