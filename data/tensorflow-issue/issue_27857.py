import tensorflow as tf
import random
import os
import json

class Generator:
    def __init__(self, mode, batch_size=100):
        self._i = 0
        self._mode = mode
        self._batch_size = batch_size

    def _get_random(self):
        return random.uniform(0, 100)

    def next_batch(self):
        self._i += 1
        if self._mode != tf.estimator.ModeKeys.TRAIN and self._i > 200:
            raise StopIteration
        features = {'a': [], 'b': []}
        labels = []
        for _ in xrange(self._batch_size):
            label = 0.0
            for key in features:
                r = self._get_random()
                features[key].append(r)
                label += r
            labels.append(label)
        return features, labels

    def output_types(self):
        return ({'a': tf.float32, 'b': tf.float32}, tf.float32)

    def output_shapes(self):
        return ({'a': [None], 'b': [None]}, [None])

def _dataset(mode):
    generator = Generator(mode)

    def generate_data():
        while True:
            yield generator.next_batch()

    return tf.data.Dataset.from_generator(
            generator=generate_data,
            output_types=generator.output_types(),
            output_shapes=generator.output_shapes(),
            args=[])

def _my_model_fn(features, labels, mode, params):
    learning_rate = params['learning_rate']
    keep_prob = params['keep_prob']
    feature_columns = [tf.feature_column.numeric_column('a'),
            tf.feature_column.numeric_column('b')]
    dense_tensor = tf.feature_column.input_layer(features, feature_columns)
    dense_tensor = tf.nn.dropout(dense_tensor, keep_prob=keep_prob)
    for units in [64, 32]:
        dense_tensor = tf.layers.dense(dense_tensor, units, tf.nn.relu)
    predictions = tf.layers.dense(dense_tensor, 1)
    predictions = tf.squeeze(predictions, [1])
    loss = tf.losses.absolute_difference(labels=labels, predictions=predictions)
    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy_op = tf.metrics.accuracy(
                labels=labels,
                predictions=predictions,
                name='accuracy_op')
        eval_metric_ops = {'accuracy': accuracy_op}
        spec = tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.EVAL,
                loss=loss,
                eval_metric_ops=eval_metric_ops)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        spec = tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.TRAIN,
                loss=loss,
                train_op=train_op)
    return spec

def _cluster():
    return {'worker': ['localhost:2222', 'localhost:2223', 'localhost:2224']}

def _set_tf_config(index):
    tf_config = {
            'cluster': _cluster(),
            'task': {'type': 'worker', 'index': index}}
    os.environ['TF_CONFIG'] = json.dumps(tf_config)

def main(argv):
    distribution = tf.contrib.distribute.CollectiveAllReduceStrategy()
    config = tf.estimator.RunConfig(
            save_checkpoints_steps=2000,
            keep_checkpoint_max=1,
            train_distribute=distribution,
            eval_distribute=distribution)
    model_dir = './model'
    learning_rate = 1e-6
    keep_prob = 0.75
    estimator = tf.estimator.Estimator(
            model_fn=_my_model_fn,
            model_dir=model_dir,
            config=config,
            params={
                'learning_rate': learning_rate,
                'keep_prob': keep_prob
            })
    train_spec = tf.estimator.TrainSpec(
            input_fn=lambda : _dataset(tf.estimator.ModeKeys.TRAIN),
            max_steps=4000)
    eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda : _dataset(tf.estimator.ModeKeys.EVAL))
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer(
        'index', 0, 'input task index')
    _set_tf_config(FLAGS.index)
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()