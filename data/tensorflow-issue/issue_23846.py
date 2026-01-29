# tf.random.uniform((B, 1), dtype=tf.int64) ← The inputs user_id and item_id have shape [B, 1] (batch size and 1 feature)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embeddings for users and items
        # Replicating original variable initialization details from TF1 snippet
        # Use tf.Variable with appropriate initialization for TF2 compatibility
        self.user_embedding_variable = tf.Variable(
            initial_value=tf.random.uniform([1000000, 32], minval=-0.5, maxval=0.5, dtype=tf.float32),
            trainable=True, name='user_emb_var')
        self.item_embedding_variable = tf.Variable(
            initial_value=tf.random.uniform([500000, 32], minval=-0.5, maxval=0.5, dtype=tf.float32),
            trainable=True, name='item_emb_var')
        self.bias = tf.Variable(tf.zeros([1], dtype=tf.float32), trainable=True, name='bias')
        weight_np = np.zeros((1, 2), dtype=np.float32)
        weight_np[:, 1] = 1  # weight not trainable per original code
        self.weight = tf.constant(weight_np, dtype=tf.float32, name='weight')

    def call(self, inputs):
        """
        inputs: tuple of (user_id, item_id)
          user_id: tf.int64 tensor shape [B, 1]
          item_id: tf.int64 tensor shape [B, 1]
        Returns:
          logits of shape [B, 2], float32
        """
        user_id, item_id = inputs
        
        # Embedding lookup
        user_emb = tf.nn.embedding_lookup(self.user_embedding_variable, tf.reshape(user_id, [-1]))
        item_emb = tf.nn.embedding_lookup(self.item_embedding_variable, tf.reshape(item_id, [-1]))
        
        # Both embeddings shape [B, 32]
        user_emb = tf.reshape(user_emb, [-1, 32])
        item_emb = tf.reshape(item_emb, [-1, 32])
        
        # Compute dot product for each example, shape [B, 1]
        cross = tf.reduce_sum(user_emb * item_emb, axis=1, keepdims=True)  # [B,1]
        
        # Add bias [1], broadcast to [B,1]
        layer = cross + self.bias
        
        # matmul with weight [1,2] to get logits [B,2]
        logits = tf.matmul(layer, self.weight)
        return logits


def my_model_function():
    # Initialize and return an instance of MyModel with random weights as above
    return MyModel()


def GetInput():
    # Produce random user_id and item_id integer tensors matching expected input shape [B,1]
    # Assuming batch size 32 as a typical example batch size
    batch_size = 32
    # user_id range: [0, 999999], item_id range: [0, 499999]
    user_id = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=1000000, dtype=tf.int64)
    item_id = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=500000, dtype=tf.int64)
    return user_id, item_id

