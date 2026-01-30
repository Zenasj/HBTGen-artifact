from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

runConfig = tf.estimator.RunConfig(session_config=config, 
                                model_dir=log_dir,
                                save_summary_steps=1,
                                save_checkpoints_steps=train_steps)
estimator = tf.keras.estimator.model_to_estimator(model, model_dir=log_dir, config=runConfig) 
train_spec = tf.estimator.TrainSpec(input_fn=lambda: read_dataset(...), max_steps=...) 
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: read_dataset(...), start_delay_secs=1,
                                throttle_secs=1, steps=None) 
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

import json
import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
data_dir = '.\\MNIST_data'
log_dir = '.\\log_dist_y'
batch_size = 512
tf.logging.set_verbosity(tf.logging.INFO)

def keras_model(lr, decay):
    """Return a CNN Keras model"""
    input_tensor = tf.keras.layers.Input(shape=(784,), name='input')

    temp = tf.keras.layers.Reshape([28, 28, 1], name='input_image')(input_tensor)
    for i, n_units in enumerate([32, 64]):
        temp = tf.keras.layers.Conv2D(n_units, kernel_size=3, strides=(2, 2),
                                      activation='relu', name='cnn'+str(i))(temp)
        temp = tf.keras.layers.Dropout(0.5, name='dropout'+str(i))(temp)
    temp = tf.keras.layers.GlobalAvgPool2D(name='average')(temp)
    output = tf.keras.layers.Dense(10, activation='softmax', name='output')(temp)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=output)
    optimizer = tf.keras.optimizers.Adam(lr=lr, decay=decay)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def main():
    """Main function"""
    data = read_data_sets(data_dir,
                          one_hot=False,
                          fake_data=False)
    model = keras_model(lr=0.001, decay=0.001)
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    config = tf.estimator.RunConfig(
                train_distribute=strategy,
                eval_distribute=strategy,
                model_dir=log_dir,
                save_summary_steps=1,
                save_checkpoints_steps=100)
    estimator = tf.keras.estimator.model_to_estimator(model, model_dir=log_dir, config=config)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
                         x={'input': data.train.images},
                         y=data.train.labels,
                         num_epochs=None,   # run forever
                         batch_size=batch_size,
                         shuffle=True)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                         x={'input': data.test.images},
                         y=data.test.labels,
                         num_epochs=1,
                         shuffle=False)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=100)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      #throttle_secs=1,
                                      steps=None    # until the end of evaluation data
                                      )

    evaluate_result = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    print("Evaluation results:")
    for key in evaluate_result[0].keys():
        print("   {}: {}".format(key, evaluate_result[0][key]))

TF_CONFIG = {
        'task': {
            'type': 'chief',
            'index': 0
        },
        'cluster': {
            'chief': ['IP1:PORT1'],
            'worker': ['IP2:PORT2'],
            'ps': ['IP1:PORT3']
        }
    }
os.environ['TF_CONFIG'] = json.dumps(TF_CONFIG)