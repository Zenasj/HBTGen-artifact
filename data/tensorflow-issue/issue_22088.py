from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
layers = tf.keras.layers

class MyModel(tf.keras.Model):
    """
    Simple CNN model.
    """

    def __init__(self, name=''):
        super(MyModel, self).__init__(name=name)

        self.flatten = layers.Flatten()

        kernel_initializer = tf.variance_scaling_initializer(scale=1.0 / 3.0, distribution='uniform')

        self.conv1 = layers.Conv2D(32, (3, 3), name='conv1', activation=tf.nn.relu6,
                                   kernel_initializer=kernel_initializer)
        self.conv2 = layers.Conv2D(64, (3, 3), name='conv2', activation=tf.nn.relu6,
                                   kernel_initializer=kernel_initializer)
        self.conv3 = layers.Conv2D(64, (3, 3), name='conv3', activation=tf.nn.relu6,
                                   kernel_initializer=kernel_initializer)
        self.fc1 = layers.Dense(512, name='fc1', activation=tf.nn.relu6)
        self.steer_predictor = layers.Dense(1, name='steer_predictor')

    def call(self, inputs, training=True):
        y = self.conv1(inputs)

        y = self.conv2(y)

        y = self.conv3(y)
        y = self.flatten(y)

        y = self.fc1(y)
        y = self.steer_predictor(y)

        return y


def get_distribution_strategy(num_gpus, all_reduce_alg=None):
    """Return a DistributionStrategy for running the model.
    Args:
      num_gpus: Number of GPUs to run this model.
      all_reduce_alg: Specify which algorithm to use when performing all-reduce.
        See tf.contrib.distribute.AllReduceCrossTowerOps for available algorithms.
        If None, DistributionStrategy will choose based on device topology.
    Returns:
      tf.contrib.distribute.DistibutionStrategy object.
    """
    if num_gpus == 0:
        return tf.contrib.distribute.OneDeviceStrategy("device:CPU:0")
    elif num_gpus == 1:
        return tf.contrib.distribute.OneDeviceStrategy("device:GPU:0")
    else:
        if all_reduce_alg:
            return tf.contrib.distribute.MirroredStrategy(
                num_gpus=num_gpus,
                cross_tower_ops=tf.contrib.distribute.AllReduceCrossTowerOps(
                    all_reduce_alg, num_packs=num_gpus))
        else:
            return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)


def model_fn(features, labels, mode, params):
    """ Model function to be used by the estimator.

    Returns:
      An EstimatorSpec object
    """

    is_training = mode == tf.estimator.ModeKeys.TRAIN

    model = MyModel()

    predictions = model(features, training=is_training)

    loss = tf.losses.mean_squared_error(labels, predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:

        global_step = tf.train.get_or_create_global_step()

        learning_rate = tf.train.linear_cosine_decay(0.0001,
                                                     global_step,
                                                     10000,
                                                     beta=0.01)

        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)

        train_op = tf.contrib.training.create_train_op(loss,
                                                       optimizer,
                                                       global_step,
                                                       summarize_gradients=False)
        # summaries
        tf.summary.image('inputs', features, max_outputs=6)
        tf.summary.scalar('training/learning_rate', learning_rate)

        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=None,
                                          loss=loss,
                                          train_op=train_op,
                                          training_hooks=None)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'mae': tf.metrics.mean_absolute_error(labels, predictions),
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops
        )


def create_input_fn():

    def input_fn():
        features = tf.random_uniform([100, 88, 200, 3])
        labels = tf.random_uniform([100, 1])
        data = tf.data.Dataset.from_tensor_slices((features, labels)).repeat().batch(128)
        return data

    return input_fn



def main(_):

    num_gpus = 2

    # run configuration
    distribution = get_distribution_strategy(num_gpus)

    # tf session config
    session_config = tf.ConfigProto(inter_op_parallelism_threads=64,
                                    intra_op_parallelism_threads=64,
                                    allow_soft_placement=True)

    run_config = tf.estimator.RunConfig(save_summary_steps=100,
                                        train_distribute=distribution,
                                        session_config=session_config)

    # Create estimator that trains and evaluates the model
    ml_estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir='/tmp/model',
        config=run_config,
        params={}
    )

    ml_estimator.train(input_fn=create_input_fn(), steps=100)


if __name__ == '__main__':
    # Set tensorflow verbosity
    tf.logging.set_verbosity(tf.logging.INFO)

    # Run the experiment
    tf.app.run()