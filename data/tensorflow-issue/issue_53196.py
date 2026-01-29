# tf.random.uniform((B, 10), dtype=tf.int64) ← Inputs are integer indices for embedding lookup

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, strategy):
        super().__init__()
        # Large embedding table shards explicitly placed on CPU due to size constraints
        with tf.device('cpu:0'):
            self.embeddings = Embeddings()
        # Smaller MLP layers placed under strategy scope (likely on GPU)
        with strategy.scope():
            self.mlp = tf.keras.Sequential([tf.keras.layers.Dense(1)])

    def call(self, inputs):
        # Embedding lookup on CPU
        x = self.embeddings(inputs)
        # Forward through the MLP (likely on GPU)
        out = self.mlp(x)
        return out


class Embeddings(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # Create 20 shards of embedding table variables, each shape [2_000_000, 256]
        # Explicitly placed on CPU to avoid GPU OOM
        with tf.device('cpu:0'):
            self.table = [
                tf.Variable(
                    name=f'shard_{idx}',
                    initial_value=tf.keras.initializers.glorot_normal()([int(2e6), 256])
                )
                for idx in range(20)
            ]

    def call(self, inputs):
        # The input is expected to be integer indices into the shards
        with tf.device('cpu:0'):
            # Perform embedding lookup across shards and sum the result per example
            # inputs shape: [batch_size, 10] (per the dataset example)
            looked_up = tf.reduce_sum(tf.nn.embedding_lookup(self.table, inputs), axis=1)
        return looked_up


# Loss function and averaging wrapper
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

def _replica_loss(labels, logits):
    # Use compute_average_loss for distributed loss normalization (global batch size 10)
    return tf.nn.compute_average_loss(loss_fn(labels, logits), global_batch_size=10)


def split_variables(tv):
    """Partition trainable variables into MirroredVariables and non-Mirrored (e.g. CPU variables)."""
    from itertools import filterfalse, tee
    from tensorflow.python.distribute.values import MirroredVariable

    def partition(pred, iterable):
        t1, t2 = tee(iterable)
        return list(filterfalse(pred, t1)), list(filter(pred, t2))

    return partition(lambda v: isinstance(v, MirroredVariable), tv)


def save_or_restore(save_dir: str, step: int, **to_save):
    # This function performs save or restore depending on whether checkpoint exists
    ckpt = tf.train.Checkpoint(**to_save)
    manager = tf.train.CheckpointManager(ckpt, save_dir, max_to_keep=10)
    latest_checkpoint = tf.train.latest_checkpoint(save_dir)
    if latest_checkpoint is not None:
        tf.print(f'Restoring checkpoint: {latest_checkpoint}')
        ckpt.restore(latest_checkpoint).expect_partial()
    else:
        manager.save(checkpoint_number=step)


def _train_step(inputs, labels, model, emb_optimizer, mlp_optimizer):
    # Run one forward and backward pass, applying gradients separately for CPU and GPU vars
    with tf.GradientTape() as tape:
        logits = model(inputs)
        loss = _replica_loss(labels, logits)

    emb_var, mlp_var = split_variables(model.trainable_variables)
    emb_grad, mlp_grad = tape.gradient(loss, [emb_var, mlp_var])
    mlp_optimizer.apply_gradients(zip(mlp_grad, mlp_var))
    # Return CPU vars and grads for application in distribute_step
    return loss, emb_var, emb_grad


def distribute_step(step_fn, strategy, model, emb_optimizer, mlp_optimizer):
    @tf.function
    def _step(*step_args):
        # Run the step function on replicas, merge losses and grads, apply CPU grads separately
        loss, emb_var, emb_grad = strategy.run(step_fn, args=step_args)
        loss = strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
        emb_grad = strategy.reduce(tf.distribute.ReduceOp.SUM, emb_grad, axis=None)
        emb_optimizer.apply_gradients(zip(emb_grad, emb_var))
        return loss

    return _step


def my_model_function():
    # Create strategy instance and build the model under it
    strategy = tf.distribute.MirroredStrategy()
    model = MyModel(strategy)
    return model, strategy


def GetInput():
    # Return a batch of random integer indices for embedding lookup in shape [batch_size, 10]
    # Indices must be in range [0, 20) because there are 20 shards.
    # But the original code uses embedding lookup on a list of variables, and tf.nn.embedding_lookup
    # treats that list as a list of tensors indexed by inputs.
    # Inputs shape is [batch, 10], dtype int64, with values 0..19 (selecting which embedding to lookup),
    # and then inside Embeddings.call it sums over embeddings for each position.
    # So inputs elements are indices (0-19) selecting shards per position.

    # To match original usage semantically:
    # Inputs shape: [batch_size, 10], each entry an integer in [0, 19]
    batch_size = 10
    input_shape = (batch_size, 10)
    input_tensor = tf.random.uniform(shape=input_shape, minval=0, maxval=20, dtype=tf.int64)
    return input_tensor

