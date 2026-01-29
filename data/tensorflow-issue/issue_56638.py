# tf.random.uniform((B, 10), dtype=tf.int64) ← Input shape is (batch_size, 10), dtype int64 as per Demo model input

import tensorflow as tf

def _get_current_replica_id_in_group_sync():
    replica_ctx = tf.distribute.get_replica_context()
    if replica_ctx:
        replica_id = replica_ctx.replica_id_in_sync_group
    else:
        from tensorflow.python.distribute import distribute_lib
        from tensorflow.python.ops import array_ops
        replica_id = distribute_lib.get_update_replica_id()
        if replica_id is None:
            replica_id = array_ops.constant(0, dtype=array_ops.dtypes.int32)
    return replica_id

def test(values):
    global_replica_id = _get_current_replica_id_in_group_sync()
    tf.print("global_replica_id: {}".format(global_replica_id))
    vector = tf.zeros_like(values)
    return vector


class TestLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TestLayer, self).__init__(**kwargs)

    def call(self, inputs, training=False):
        emb_vector = test(values=inputs)
        return emb_vector


class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.test_layer = TestLayer()
        # Dense layer with 1 output unit, kernel initialized to ones, bias zeros
        self.dense_layer = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer="ones",
            bias_initializer="zeros"
        )

    def call(self, inputs, training=False):
        # Call the custom TestLayer which prints replica id and returns zeros_like input
        vector = self.test_layer(inputs)
        logit = self.dense_layer(vector)
        # Return logits and vector as per original Demo model
        return logit, vector

    def summary(self):
        # Build and print summary by creating a Model temporarily
        inputs = tf.keras.Input(shape=(10,), dtype=tf.int64)
        tmp_model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        tmp_model.summary()


def my_model_function():
    # Construct and return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate a random input tensor matching expected input: shape (batch_size, 10), dtype int64
    # Use batch size 8 for example
    batch_size = 8
    input_tensor = tf.random.uniform(
        shape=(batch_size, 10),
        minval=0,
        maxval=100,
        dtype=tf.int64
    )
    return input_tensor

