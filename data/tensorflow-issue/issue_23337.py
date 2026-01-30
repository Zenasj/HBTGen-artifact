import tensorflow as tf

def fn_resize_image(self, image):
    im = tf.image.resize(image, [new_height, new_width])
    return im

def read_tfrecord(sample, output_size, num_class):

    features = {
        "image": tf.io.FixedLenFeature([], tf.string), 
        "one_hot_class": tf.io.VarLenFeature(tf.float32),
    }

    sample = tf.io.parse_single_example(sample, features)
    image = tf.image.decode_jpeg(sample['image'], channels=3)
    image = tf.compat.v1.tpu.outside_compilation(fn_resize_image,image)
    image = tf.cast(image, tf.float32) / 255.0 
    image_tensor = tf.reshape(image, [*self.image_size, 3]) # explicit size will be needed for TPU
    one_hot_class = tf.sparse.to_dense(sample['one_hot_class'])
    one_hot_class = tf.reshape(one_hot_class, [num_class])

    return image_tensor, one_hot_class