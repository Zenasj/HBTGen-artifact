from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf

BUFFER_SIZE = 10000
BATCH_SIZE = 64
LEARNING_RATE = 1e-4


def input_fn(mode, input_context=None):
    max_seq_len = 3
    rnn_dataset = tf.data.Dataset\
        .range(10)\
        .repeat(10 * BUFFER_SIZE) \
        .map(lambda x: (
        tf.ones(shape=(max_seq_len,), dtype=tf.int64),
        tf.ones(shape=(max_seq_len,), dtype=tf.int64)))
    if input_context:
        rnn_dataset = rnn_dataset.shard(
            input_context.num_input_pipelines,
            input_context.input_pipeline_id)
    return rnn_dataset.batch(BATCH_SIZE)


def model_fn(features, labels, mode):
    vocab_size = 100
    embed_size = 16
    state_size = 7
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_size),
        tf.keras.layers.LSTM(units=state_size, return_sequences=True),
        tf.keras.layers.Dense(10, activation='softmax')])
    logits = model(features, training=False)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            tf.estimator.ModeKeys.PREDICT,
            predictions={'logits': logits})

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(
        learning_rate=LEARNING_RATE)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE)
    loss = tf.reduce_sum(loss(labels, logits)) * (1. / BATCH_SIZE)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss,
        train_op=optimizer.minimize(
            loss, tf.compat.v1.train.get_or_create_global_step()))


def main():
    strategy = tf.distribute.MirroredStrategy()
    config = tf.estimator.RunConfig(
        train_distribute=strategy,
        log_step_count_steps=1)

    classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir='/tmp/multiworker', config=config)
    tf.estimator.train_and_evaluate(
        classifier,
        train_spec=tf.estimator.TrainSpec(input_fn=input_fn, max_steps=10),
        eval_spec=tf.estimator.EvalSpec(input_fn=input_fn))


if __name__ == '__main__':
    main()

def build_and_compile_rnn_model():
    vocab_size = 100
    embed_size = 16
    state_size = 7
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_size),
        tf.keras.layers.LSTM(units=state_size, return_sequences=True),
        tf.keras.layers.Dense(10, activation='softmax')])
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))
    return model

def get_rnn_dataset():
    max_seq_len = 3
    rnn_dataset = tf.data.Dataset\
        .range(10)\
        .repeat(10 * BUFFER_SIZE) \
        .map(lambda x: (
            tf.ones(shape=(max_seq_len,), dtype=tf.int64),
            tf.ones(shape=(max_seq_len,), dtype=tf.int64)))
    return rnn_dataset.batch(BATCH_SIZE)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_and_compile_rnn_model()
model.fit(x=get_rnn_dataset(), epochs=3)