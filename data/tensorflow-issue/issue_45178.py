from tensorflow.keras import layers
from tensorflow.keras import optimizers

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}
import tensorflow as tf
from tensorflow import keras
import numpy as np

tf.get_logger().setLevel('DEBUG')

def get_compiled_model():
    # Make a simple 2-layer densely-connected neural network.
    inputs = keras.Input(shape=(784,))
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.Dense(256, activation="relu")(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def get_dataset():
    batch_size = 10240
    num_val_samples = 10000

    # Return the MNIST dataset in the form of a `tf.data.Dataset`.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.concatenate([x_train for i in range(50)])    
    y_train = np.concatenate([y_train for i in range(50)])

    # Preprocess the data (these are Numpy arrays)
    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    x_test = x_test.reshape(-1, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Reserve num_val_samples samples for validation
    x_val = x_train[-num_val_samples:]
    y_val = y_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]
    y_train = y_train[:-num_val_samples]
    
    print(len(x_train), len(y_train),len(x_val),len(y_val))    
    return (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size),
    )


# Create a MirroredStrategy.
#strategy = tf.distribute.MirroredStrategy()
# cross_tower_ops = tf.distribute.ReductionToOneDevice()
cross_tower_ops = tf.distribute.HierarchicalCopyAllReduce(num_packs=4)
# Reduce to CPU
#cross_tower_ops = tf.distribute. ReductionToOneDevice(reduce_to_device="/device:CPU:0")
strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0','/gpu:1','/gpu:2','gpu:3'],
                                            cross_device_ops=cross_tower_ops)
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    model = get_compiled_model()
    # model.summary()

# Train the model on all available devices.
train_dataset, val_dataset, test_dataset = get_dataset()
    # train_dataset, val_dataset, test_dataset = (strategy.experimental_distribute_dataset(train_dataset),
    #         strategy.experimental_distribute_dataset(val_dataset), 
    #         strategy.experimental_distribute_dataset(test_dataset))
    # model.summary()
model.fit(train_dataset, epochs=200, validation_data=val_dataset)

# Test the model on all available devices.
model.evaluate(test_dataset)