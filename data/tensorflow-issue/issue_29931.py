import tensorflow as tf
from tensorflow import keras

def build_dataset(boxes_df, data_directory='/content'):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in ['block5_pool']]
    vgg = tf.keras.Model([vgg.input], outputs)

    filenames_ds = tf.data.Dataset.from_tensor_slices(boxes_df['image_name'].apply(lambda path: os.path.join(data_directory, path)))
    x1_ds        = tf.data.Dataset.from_tensor_slices(boxes_df['x_1'])
    x2_ds        = tf.data.Dataset.from_tensor_slices(boxes_df['x_2'])
    y1_ds        = tf.data.Dataset.from_tensor_slices(boxes_df['y_1'])
    y2_ds        = tf.data.Dataset.from_tensor_slices(boxes_df['y_2'])
    tmp_ds       = tf.data.Dataset.zip((filenames_ds, x1_ds, x2_ds, y1_ds, y2_ds))
    #"""
    images_ds    = tmp_ds.map(
        lambda path, x1, x2, y1, y2: tf.image.resize_images(
            tf.image.crop_to_bounding_box(
                tf.image.decode_jpeg(tf.read_file(path)),
                tf.cast(x1, tf.int32),
                tf.cast(y1, tf.int32),
                tf.cast(x2 - x1, tf.int32),
                tf.cast(y2 - y1, tf.int32)
            ),
            (224, 224)
        )
    )
    images_ds = images_ds.map(
        lambda img: tf.keras.applications.vgg19.preprocess_input(img)
    )
    features_ds = images_ds.map(
        lambda img: vgg(tf.expand_dims(img, axis=0)).reshape(7 * 7 * 512)
    )
    return features_ds