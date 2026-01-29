# tf.random.uniform((global_batch_size, 1), dtype=tf.float32)
import tensorflow as tf

# Assumptions and notes based on issue:
# - Input shape inferred from example dataset: (global_batch_size, 1)
# - The model is a simple single Dense layer with 1 unit
# - The issue is about distributed training with MirroredStrategy and reducing losses with ReduceOp.MEAN failing
# - The step function returns per-example cross-entropy losses (shape [batch_size])
# - The main error is about attempting to reduce a non-DistributedValue scalar or tensor incorrectly across replicas
# - The sample code uses softmax_cross_entropy_with_logits and applies gradients on that
# - The fix is to sum losses per replica, then use mirrored_strategy.reduce with ReduceOp.SUM,
#   and then normalize dividing by global batch size outside reduce
# - The code example below integrates this fix into a tf.keras.Model subclass for testing


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple model matching the example: one Dense layer with 1 unit
        # Input shape is (batch_size, 1)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        # Forward pass
        return self.dense(inputs)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random float tensor with shape (global_batch_size, 1), dtype float32
    # Assuming global_batch_size = 20 as in example
    global_batch_size = 20
    return tf.random.uniform((global_batch_size, 1), dtype=tf.float32)


# Additional helper function to demonstrate the "fixed" train_step version
# This is NOT part of the requirements but clarifies how to do a distributed reduction correctly

def train_step_example(model, optimizer, mirrored_strategy, dist_inputs, global_batch_size):
    """
    Executes a training step for distributed dataset inputs on mirrored_strategy.

    Args:
      model: instance of MyModel
      optimizer: tf.keras optimizer instance
      mirrored_strategy: tf.distribute.MirroredStrategy instance
      dist_inputs: per-replica distributed inputs tuple (features, labels)
      global_batch_size: int scalar
   
    Returns:
      Scalar tensor: mean loss across global batch
    """

    def step_fn(inputs):
        features, labels = inputs

        with tf.GradientTape() as tape:
            logits = model(features, training=True)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels
            )  # shape: (local_batch_size,)
            loss = tf.reduce_sum(cross_entropy)  # sum over local batch
            # Scaling loss by 1/global_batch, but done after reduce, see below
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss  # sum loss per replica

    per_replica_loss = mirrored_strategy.run(step_fn, args=(dist_inputs,))

    # Reduce sum losses from all replicas to single scalar
    sum_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    # Normalize by global batch size to get mean loss
    mean_loss = sum_loss / tf.cast(global_batch_size, dtype=sum_loss.dtype)

    return mean_loss

