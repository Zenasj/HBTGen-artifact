from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.python.client import device_lib
import time

# Uncomment this to observe the device used to 
# run tensorflow operations
#tf.debugging.set_log_device_placement(True)
''' 
Description:
Runs three different tests to measure performance on both a GPU and CPU if
available. Each test uses tensorflow
alpha 2.0, and trains using the MNIST dataset provided/bundled by tensorflow.
Each test runs for 1 epoch comprised of 1000 mini-batches. A mini-batch contains
32 samples. All tests use Adam optimizer, and sparse categorical loss.

Test descriptions:
- Test 1: Uses the Keras 'compile' and 'fit' function
- Test 2: Uses tf.GradientTape and does not use the @tf.function decorator
- Test 3: Uses tf.GradientTape and the @tf.function decorator. 
    The @tf.function decorator is applied to the function training 
    on MNIST mini-batches

Issue Observed:
(example below for GPU for example)
Test 1 provides a benchmark accuracy of ~70% (6.4 sec runtime) on the training data after 1000 
steps. Test 3 achieves roughly the same accuracy in a similar time (even 
a little better ~80% with 3.75 sec runtime). Test 2 achieves ~20-40% accuracy (14-16 sec runtime) 
after training for number of batches. The only difference
between Test 2 and Test 3 is no utilization of the tf.function decorator in Test 2. 

The slow down in Test 2 is expected when not using the tf.function decorator.
However, I would expect roughly the same training accuracy after 1000 steps for each 
test.
'''

# ------------------- Create Several Helper Functions and variables 
# ------------------- used for training
def print_info():
    print(tf.__version__)
    print(tf.executing_eagerly())
    print(device_lib.list_local_devices())
    print()

def get_available_device_names():
    local_devices = device_lib.list_local_devices()
    names = []
    for device in local_devices:
        names.append(device.name)
    return names

def get_and_prepare_data():
    data, info = tfds.load(
        'mnist',with_info=True, as_supervised=True)

    train_data, test_data = data['train'], data['test']

    batch_size = 32
    prep_train_data = train_data.batch(batch_size).prefetch(1).repeat()
    return prep_train_data, info

def create_model():
    img_shape = info.features['image'].shape
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=img_shape),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

def train_batch(model, images, label, loss_fn, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        pred = model(images)
        loss = loss_fn(label, pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    grads_and_vars = zip(gradients, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars)

    train_loss(loss)
    train_accuracy(label, pred)
    
    return loss, pred

def run_batch(batch_train_func, steps_per_epoch=1000, epochs=1):
    train_loss = keras.metrics.Mean()
    train_accuracy = keras.metrics.SparseCategoricalAccuracy()
    for e in range(epochs):
        batch = 0
        for (data, label) in prep_train_data:
            loss, pred = batch_train_func(data, label, train_loss, train_accuracy)
            
            if batch % 500 == 0:
                print("Batch {} accuracy {} loss {}\n".format(
                    batch, train_accuracy.result(), train_loss.result()))
            if batch >= steps_per_epoch:
                break
            batch = batch + 1

# Basic info
print_info()
avail_device_names = get_available_device_names()
print(avail_device_names)

steps_per_epoch = 1000
epochs = 1
devices = ['/device:CPU:0', '/device:GPU:0']

for device in devices:
    if device not in avail_device_names:
        print("Device {} not available".format(device))
        continue # Device not available

    print("\n-------------------------------------------")
    print("-------------------------------------------")
    print("Starting test for device {}".format(device))
    print("-------------------------------------------")

    # Tests record accuracy, loss, batch number and test type
    # ----------------------------- TEST 1 --------------------------- #
    # ------------ Test Using Keras compile and fit ------------------ #

    print("Begin Benchmarking Test 1")
    print("Test uses keras functions compile and fit")

    # Data prep
    prep_train_data, info = get_and_prepare_data()

    t1_start = time.time()

    with tf.device(device):
        keras_fit_model = create_model()

        keras_fit_model.compile(optimizer='adam', 
                        loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        keras_fit_model.fit(prep_train_data, epochs=epochs, steps_per_epoch=steps_per_epoch,verbose=2)

    t1_end = time.time()
    print("Test 1 Time Elapsed {}\n".format(t1_end-t1_start))

    # ----------------------------- TEST 2 ----------------------------- #
    # --------- Test Using Gradient Tape without tf.function ----------- #
    print("Begin Benchmarking Test 2")
    print("Test uses tf.GradientTape() (does not use tf.function decorator)")

    # Data prep
    prep_train_data, info = get_and_prepare_data()

    t2_start = time.time()

    with tf.device(device):
        eager_model = create_model()
        eager_optimizer = tf.keras.optimizers.Adam()
        eager_loss_func = keras.losses.SparseCategoricalCrossentropy()

        def eager_train_batch(data, label, train_loss, train_accuracy):
            return train_batch(eager_model, data, label, eager_loss_func, eager_optimizer, train_loss, train_accuracy)

        run_batch(eager_train_batch, steps_per_epoch, epochs)

    t2_end = time.time()
    print("Test 2 Time Elapsed {}\n".format(t2_end-t2_start))


    # ----------------------------- TEST 3 ----------------------------------- #
    # --------- Test Using Gradient Tape with tf.function decorator----------- #
    print("Begin Benchmarking Test 3")
    print("Test uses tf.GradientTape() (uses tf.function decorator)")

    # Data prep
    prep_train_data, info = get_and_prepare_data()

    t3_start = time.time()

    with tf.device(device):
        graph_model = create_model()
        graph_optimizer = tf.keras.optimizers.Adam()
        graph_loss_func = keras.losses.SparseCategoricalCrossentropy()

        @tf.function
        def graph_train_batch(data, label, train_loss, train_accuracy):
            return train_batch(graph_model, data, label, graph_loss_func, graph_optimizer, train_loss, train_accuracy)

        run_batch(graph_train_batch, steps_per_epoch, epochs)

    t3_end = time.time()
    print("Test 3 Time Elapsed {}\n".format(t3_end-t3_start))