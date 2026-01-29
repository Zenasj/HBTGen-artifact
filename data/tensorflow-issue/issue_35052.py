# tf.random.uniform((batch_size, 1), dtype=tf.int32)
import tensorflow as tf

bucket = int(1e7)

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

    def build(self, input_shape):
        # User and item embeddings of size (bucket+1, 32)
        self.user_emb = self.add_weight(
            shape=(bucket + 1, 32),
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(),
            name="user_emb")
        self.item_emb = self.add_weight(
            shape=(bucket + 1, 32),
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(),
            name="item_emb")
        self.bias = tf.Variable(0.0, dtype=tf.float32)

    def call(self, inputs):
        # inputs is a tuple: (user_id, item_id)
        user_id, item_id = inputs
        # Flatten input ids to shape (batch_size,)
        user_id = tf.reshape(user_id, [-1])
        item_id = tf.reshape(item_id, [-1])
        # Look up embeddings: (batch_size, 32)
        user_vec = tf.gather(self.user_emb, user_id)
        item_vec = tf.gather(self.item_emb, item_id)
        # Element-wise multiply user and item embeddings
        interaction = user_vec * item_vec
        # Sum interaction vectors along embedding dimension -> (batch_size, 1)
        out = tf.reduce_sum(interaction, axis=1, keepdims=True) + self.bias
        # Apply sigmoid to get predicted probability
        out = tf.sigmoid(out)
        return out

def my_model_function():
    # Return an instance of MyModel with weights initialized but not loaded pretrained
    return MyModel()

def GetInput():
    # Generate a tuple of tensors (user_id, item_id) for batch input
    # Both have shape (batch_size, 1), dtype int32, values in [0, bucket)
    batch_size = 32
    user_id = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=bucket, dtype=tf.int32)
    item_id = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=bucket, dtype=tf.int32)
    return (user_id, item_id)

