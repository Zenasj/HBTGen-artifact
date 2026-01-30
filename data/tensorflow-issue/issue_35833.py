from tensorflow import keras
from tensorflow.keras import layers

# python 3.6. Tested with tensorflow-gpu-1.14 and tensorflow-cpu-2.0
import tensorflow as tf
import numpy as np


def get_model(IM_WIDTH=28, num_color_channels=1):
    """Create a very simple convolutional neural network using a tf.keras Functional Model."""
    input = tf.keras.Input(shape=(IM_WIDTH, IM_WIDTH, num_color_channels))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(input)
    x = tf.keras.layers.MaxPooling2D(3)(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(3)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs=[input], outputs=[output])
    model.compile(optimizer='adam', loss="mae",
                  metrics=['mae'])
    model.summary()
    return model


def input_fun(train=True):
    """Load MNIST and return the training or test set as a tf.data.Dataset; Valid input function for tf.estimator"""
    (train_images, train_labels), (eval_images, eval_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape((60_000, 28, 28, 1)).astype(np.float32) / 255.
    eval_images = eval_images.reshape((10_000, 28, 28, 1)).astype(np.float32) / 255.
    # train_labels = train_labels.astype(np.float32)  # these two lines don't affect behaviour.
    # eval_labels = eval_labels.astype(np.float32)
    # For a neural network with one neuron in the final layer, it doesn't seem to matter if target data is float or int.

    if train:
        dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        dataset = dataset.shuffle(buffer_size=100).repeat(None).batch(32).prefetch(1)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((eval_images, eval_labels))
        dataset = dataset.batch(32).prefetch(1)  # note: prefetching does not affect behaviour

    return dataset


model = get_model()
train_input_fn = lambda: input_fun(train=True)
eval_input_fn = lambda: input_fun(train=False)

NUM_EPOCHS, STEPS_PER_EPOCH = 4, 1875  # 1875 = number_of_train_images(=60.000)  /  batch_size(=32)
USE_ESTIMATOR = False  # change this to compare model/estimator. Estimator performs much worse for no apparent reason
if USE_ESTIMATOR:
    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model, model_dir="model_directory",
        config=tf.estimator.RunConfig(save_checkpoints_steps=200, save_summary_steps=200))

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=STEPS_PER_EPOCH * NUM_EPOCHS)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=0)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    print("Training complete. Evaluating Estimator:")
    print(estimator.evaluate(eval_input_fn))
    # final train loss with estimator: ~2.5 (mean abs. error).
else:
    dataset = train_input_fn()
    model.fit(dataset, steps_per_epoch=STEPS_PER_EPOCH, epochs=NUM_EPOCHS)
    print("Training complete. Evaluating Keras model:")
    print(model.evaluate(eval_input_fn()))
    # final train loss with Keras model: ~0.4 (mean abs. error).