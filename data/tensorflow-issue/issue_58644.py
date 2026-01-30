import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def process_single_sample_train(img_path, label):

    # 1. Read image
    img = tf.io.read_file(img_path)

    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=3) # channels = 1 for grayscale

    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)

    # 4. Resize to the desired size
    img = tf.image.resize(img, [32, 128])
    
    # 5. Augment
    rand_ = tf.random.uniform(shape=(), minval=0, maxval=1)
    if rand_ < 0.8:
        img = tf.image.random_contrast(img, lower=0.7, upper=1)
        
        ## THIS LINE CAUSES THE BUG
        img = tf.convert_to_tensor(tf.keras.preprocessing.image.random_rotation(img, rg=5))
        
    return {"image": img, "label": label}

train_dataset = (
    train_dataset.map(
        process_single_sample_train, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

layer_rotate = tf.keras.layers.RandomRotation(factor=(-0.3, 0.2))
image_rotate = layer_rotate(image)

data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomRotation(factor=(-0.3, 0.2))])
img_rotated       = data_augmentation(img)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)
...
...
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(16).map(lambda x, y: (data_augmentation(x), y))