import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np
import tqdm
import os
import psutil
import matplotlib.pyplot as plt


class TestModel(tf.keras.models.Model):
    def __init__(self, input_shape):
        model = tf.keras.applications.VGG16(input_shape=input_shape[1:], include_top=False)
        output_encoder = model.get_layer('block5_pool').output
        decoder = tf.keras.layers.Conv2D(3, 3, padding='SAME')(output_encoder)
        decoder = tf.keras.layers.UpSampling2D(size=(32,32))(decoder)
        decoder = tf.keras.layers.Concatenate()([model.input, decoder])
        decoder = tf.keras.layers.Conv2D(3, 3, padding='SAME')(decoder)
        super().__init__(model.input, decoder)

def run_predictions(iterations):
    input_shape = 1, 256, 256, 3
    model = TestModel(input_shape)
    inputs = np.random.rand(*input_shape)
    memory_used = []
    for _ in tqdm.tqdm(range(iterations)):
        process = psutil.Process(os.getpid())
        memory_used += [process.memory_info().rss / 1024.0 / 1024.0]
        model.predict(inputs, batch_size=1)
    return memory_used

if __name__ == "__main__":
    memory_used = run_predictions(5000)
    plt.plot(memory_used)
    plt.xlabel("Iteration")
    plt.ylabel("Memroy Usage (MB)")
    plt.show()

def run_predictions(iterations):
    input_shape = 1, 256, 256, 3
    model = TestModel(input_shape)
    inputs = np.random.rand(*input_shape)
    memory_used = []
    for _ in tqdm.tqdm(range(iterations)):
        process = psutil.Process(os.getpid())
        memory_used += [process.memory_info().rss / 1024.0 / 1024.0]
        model(inputs)   # This is the changed line: Call the model directly
    return memory_used

import tensorflow as tf
import numpy as np
import tqdm
import os
import psutil
import matplotlib.pyplot as plt


class MemoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_used = []

    def on_epoch_end(self, epoch, log={}):
        process = psutil.Process(os.getpid())
        self.memory_used += [process.memory_info().rss / 1024.0 / 1024.0]


class TestModel(tf.keras.models.Model):
    def __init__(self, input_shape):
        model = tf.keras.applications.VGG16(input_shape=input_shape[1:], include_top=False)
        output_encoder = model.get_layer('block5_pool').output
        decoder = tf.keras.layers.Conv2D(3, 3, padding='SAME')(output_encoder)
        decoder = tf.keras.layers.UpSampling2D(size=(32,32))(decoder)
        decoder = tf.keras.layers.Concatenate()([model.input, decoder])
        decoder = tf.keras.layers.Conv2D(3, 3, padding='SAME')(decoder)
        super().__init__(model.input, decoder)


def run_training(iterations, epochs):
    input_shape = 1, 256, 256, 3
    model = TestModel(input_shape)
    model.compile(optimizer="Adam", loss="categorical_crossentropy")

    inputs = np.random.rand(*input_shape)
    labels = np.random.rand(*input_shape)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).repeat().batch(2)

    callback = MemoryCallback()

    model.fit(dataset, steps_per_epoch=iterations, epochs=epochs, callbacks=[callback])
    return callback.memory_used

if __name__ == "__main__":
    memory_used = run_training(iterations=20, epochs=100)
    plt.plot(memory_used)
    plt.xlabel("Iteration")
    plt.ylabel("Memroy Usage (MB)")
    plt.show()

import tensorflow as tf
import numpy as np
import tqdm
import os
import psutil
import matplotlib.pyplot as plt


class TestModel(tf.keras.models.Model):
    def __init__(self, input_shape, output_layer_name):
        model = tf.keras.applications.VGG16(input_shape=input_shape[1:], include_top=False)
        output_encoder = model.get_layer(output_layer_name).output
        super().__init__(model.input, output_encoder)

def run_predictions(iterations, output_layer_name):
    input_shape = 1, 256, 256, 3
    model = TestModel(input_shape, output_layer_name)
    inputs = np.random.rand(*input_shape)
    memory_used = []
    for _ in tqdm.tqdm(range(iterations)):
        process = psutil.Process(os.getpid())
        memory_used += [process.memory_info().rss / 1024.0 / 1024.0]
        model.predict(inputs, batch_size=1)
    return memory_used

if __name__ == "__main__":
    memory_used = run_predictions(5000, 'block1_pool')
    plt.plot(memory_used)
    plt.xlabel("Iteration")
    plt.ylabel("Memroy Usage (MB)")
    plt.show()

import tensorflow as tf
import numpy as np
import tqdm
import os
import psutil
import matplotlib.pyplot as plt


class TestModel(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.layer_1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')
        self.layer_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')
		
    def call(self, inputs):
        return self.layer_2(self.layer_1(inputs))

def run_predictions(iterations):
    input_shape = 1, 256, 256, 3
    model = TestModel()
    inputs = np.random.rand(*input_shape)
    memory_used = []
    for _ in tqdm.tqdm(range(iterations)):
        process = psutil.Process(os.getpid())
        memory_used += [process.memory_info().rss / 1024.0 / 1024.0]
        model.predict(inputs, batch_size=1)
    return memory_used

if __name__ == "__main__":
    memory_used = run_predictions(5000)
    plt.plot(memory_used)
    plt.xlabel("Iteration")
    plt.ylabel("Memroy Usage (MB)")
    plt.show()

class TestModel(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.layer_1 = tf.keras.layers.Conv2D(64, (3, 3), activation=None, padding='same', name='block1_conv1')   # Note activations here
        self.layer_2 = tf.keras.layers.Conv2D(64, (3, 3), activation=None, padding='same', name='block1_conv2')   # Note activations here
		
    def call(self, inputs):
        return self.layer_2(self.layer_1(inputs))

import tensorflow as tf
import numpy as np
import tqdm
import os
import psutil
import matplotlib.pyplot as plt


class TestModel(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.layer_1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')
        self.layer_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')
		
    def call(self, inputs):
        return self.layer_2(self.layer_1(inputs))


model = TestModel()


def run_model(inputs):
    return model(inputs)

def run_predictions(iterations):
    input_shape = 1, 256, 256, 3
    inputs = np.random.rand(*input_shape)
    memory_used = []
    for _ in tqdm.tqdm(range(iterations)):
        process = psutil.Process(os.getpid())
        memory_used += [process.memory_info().rss / 1024.0 / 1024.0]
        run_model(inputs)
    return memory_used

if __name__ == "__main__":
    memory_used = run_predictions(5000)
    plt.plot(memory_used)
    plt.xlabel("Iteration")
    plt.ylabel("Memroy Usage (MB)")
    plt.show()

@tf.function
def run_model(inputs):
    return model(inputs)

inputs = np.random.rand(*input_shape)

inputs = tf.convert_to_tensor(np.random.rand(*input_shape))

class TestModel(tf.keras.models.Model):
    def __init__(self, input_shape):
        model = tf.keras.applications.VGG16(input_shape=input_shape[1:], include_top=False)
        output_encoder = model.get_layer('block5_pool').output
        decoder = tf.keras.layers.Conv2D(3, 3, padding='SAME')(output_encoder)
        decoder = tf.keras.layers.UpSampling2D(size=(32,32))(decoder)
        decoder = tf.keras.layers.Concatenate()([model.input, decoder])
        decoder = tf.keras.layers.Conv2D(3, 3, padding='SAME')(decoder)
        super().__init__(model.input, decoder)
		
def run_predictions(iterations):
    input_shape = 1, 256, 256, 3
    model = TestModel(input_shape)
    inputs = np.random.rand(*input_shape)
    memory_used = []
    for _ in tqdm.tqdm(range(iterations)):
        process = psutil.Process(os.getpid())
        memory_used += [process.memory_info().rss / 1024.0 / 1024.0]
        model.predict(inputs, batch_size=1)
    return memory_used

if __name__ == "__main__":
    tracemalloc.start()
    memory_used = run_predictions(1000)
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    for stat in top_stats[:10]:
        print(stat)
    plt.plot(memory_used)
    plt.xlabel("Iteration")
    plt.ylabel("Memroy Usage (MB)")
    plt.show()

import tensorflow as tf
import numpy as np
import tqdm
import os
import psutil
import matplotlib.pyplot as plt


class TestModel(tf.keras.models.Model):
    def __init__(self, input_shape):
        model = tf.keras.applications.VGG16(input_shape=input_shape[1:], include_top=False)
        output_encoder = model.get_layer('block5_pool').output
        decoder = tf.keras.layers.Conv2D(3, 3, padding='SAME')(output_encoder)
        decoder = tf.keras.layers.UpSampling2D(size=(32,32))(decoder)
        decoder = tf.keras.layers.Concatenate()([model.input, decoder])
        decoder = tf.keras.layers.Conv2D(3, 3, padding='SAME')(decoder)
        super().__init__(model.input, decoder)
		
def run_predictions(iterations):
    input_shape = 1, 256, 256, 3
    model = TestModel(input_shape)
    inputs = np.random.rand(*input_shape)
    memory_used = []
    for _ in tqdm.tqdm(range(iterations)):
        process = psutil.Process(os.getpid())
        memory_used += [process.memory_info().rss / 1024.0 / 1024.0]
        model.predict(inputs, batch_size=1)
    return memory_used

if __name__ == "__main__":
    options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 3, python_tracer_level = 1, device_tracer_level = 1)
    tf.profiler.experimental.start('logdir', options = options)
    memory_used = run_predictions(250)
    tf.profiler.experimental.stop()
    plt.plot(memory_used)
    plt.xlabel("Iteration")
    plt.ylabel("Memroy Usage (MB)")
    plt.show()