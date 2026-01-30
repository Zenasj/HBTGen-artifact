import numpy as np
import random
import tensorflow as tf

class DummyTrainer:
    def __init__(self, file_path, samples):
        self.file_path = file_path
        self.samples = samples
        self.index_array = np.random.permutation(self.samples)

    def _calc_image(self, index):
        filename= os.path.join(self.file_path, str(i)+'.png')
        if os.path.isfile(filename):
            img_raw = tf.io.read_file(filename)
            img = tf.io.decode_png(img_raw, channels=1)
            img = tf.dtypes.cast(img, tf.float32) / 256.0
            return img
        else:
            return None

    def _dataset_function(self, index):
        return tuple(tf.numpy_function(
            self._calc_image, [index],
            [tf.float32, tf.uint8]))

    def get_trainer_dataset(self):
        # Reading the images from the numpy array of indexes
        dataset = Dataset.from_tensor_slices(
            self.index_array[:self.trainer_samples])
        return dataset.map(self._dataset_function).shuffle(1)

trainer = DummyTrainer('data', 256)

model.fit(trainer.get_trainer_dataset(), ...)

def mappable_fn(x):
            result_tensors = tf.py_func(func=my_py_func,
                                        inp=[my_py_func_args],
                                        Tout=[my_py_func_output_types])
            result_tensor.set_shape(the_shape_i_know_ahead_of_time)
            return (result_tensor)

def load_image(file, label):
    nifti = np.asarray(nibabel.load(file.numpy().decode('utf-8')).get_fdata()).astype(np.float32)

    xs, ys, zs = np.where(nifti != 0)
    nifti = nifti[min(xs):max(xs) + 1, min(ys):max(ys) + 1, min(zs):max(zs) + 1]
    nifti = nifti[0:100, 0:100, 0:100]
    nifti = np.reshape(nifti, (100, 100, 100, 1))
    return nifti, label


@tf.autograph.experimental.do_not_convert
def load_image_wrapper(file, label):
    result_tensors = tf.py_function(load_image, [file, label], [tf.float64, tf.float64])
    result_tensors[0].set_shape([100, 100, 100, 1])
    result_tensors[1].set_shape([None])
    return result_tensors


dataset = tf.data.Dataset.from_tensor_slices((train, labels))
dataset = dataset.map(load_image_wrapper, num_parallel_calls=24)
dataset = dataset.repeat(50)
dataset = dataset.batch(12, drop_remainder=True)
dataset = dataset.prefetch(buffer_size=6)

dataset = tf.data.Dataset.from_tensor_slices((train, labels))
dataset = dataset.map(load_image_wrapper, num_parallel_calls=12)
dataset = dataset.repeat(50)
dataset = dataset.prefetch(buffer_size=2)
dataset = dataset.apply(tf.data.experimental.prefetch_to_device('/device:GPU:0', 1))
dataset = dataset.batch(12, drop_remainder=True)

def load_image(file, label):
    nifti = np.asarray(nibabel.load(file.numpy().decode('utf-8')).get_fdata()).astype(np.float32)

    xs, ys, zs = np.where(nifti != 0)
    nifti = nifti[min(xs):max(xs) + 1, min(ys):max(ys) + 1, min(zs):max(zs) + 1]
    nifti = nifti[0:100, 0:100, 0:100]
    nifti = np.reshape(nifti, (100, 100, 100, 1))
    return nifti, label


@tf.autograph.experimental.do_not_convert
def load_image_wrapper(file, label):
    return tf.py_function(load_image, [file, label], [tf.float64, tf.float64])


dataset = tf.data.Dataset.from_tensor_slices((train, labels))
dataset = dataset.map(load_image_wrapper, num_parallel_calls=32)
dataset = dataset.prefetch(buffer_size=1)
dataset = dataset.apply(tf.data.experimental.prefetch_to_device('/device:GPU:0', 1))
# Now my dataset size is 522, and for batch, I'm creating a single batch of the entire dataset.
dataset = dataset.batch(522, drop_remainder=True).repeat()

# Initialise iterator
iterator = iter(dataset)

# get x and y
batch_image, batch_label = iterator.get_next()

# Over here, supply model.fit with x & y, and THEN supply your batch size here.
model.fit(batch_image, batch_label, epochs=100, batch_size=12)