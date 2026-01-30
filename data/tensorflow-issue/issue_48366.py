import math
import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import os

import tensorflow as tf


CFG = dict(
            dataset_path=os.path.join('datasets', 'grassclover'),
            train_data_txt='train.txt',
            validation_data_txt='val.txt',
            test_data_txt='val.txt',
            output_dir=os.path.join('outputs', 'grassclover'),
            # network hyperparameters
            batch_size=8,
            n_classes=15,
            epochs=100,
            target_height=512,
            target_width=512,
            n_channels=3,
            replace_with_label=15
        )


class CreateDataset:
    def __init__(self):
        self.cfg = CFG

        self.step = 32  # model required divisible image size
        self.target_height = tf.cast(
            tf.math.multiply(tf.math.ceil(tf.math.divide(self.cfg["target_height"], self.step)), self.step),
            tf.int32
        ).numpy()

        self.target_width = tf.cast(
            tf.math.multiply(tf.math.ceil(tf.math.divide(self.cfg["target_width"], self.step)), self.step),
            tf.int32
        ).numpy()

        self.is_only_resize = True  # normalize resize, False -> random cropping and resize

    @tf.function
    def normalize(self, input_image: tf.Tensor):
        """
        normalizes image pixel values between 0.0 and 1.0
        :param input_image: Tensor containing image : dim [Size, Size, 3]
        :return: normalized image
        """

        input_image = tf.cast(input_image, tf.float32) / 255.0

        return input_image

    @tf.function
    def read_file(self, full_path):
        img = tf.io.read_file(full_path)

        return img

    @tf.function
    def decode_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.uint8)

        return image

    @tf.function
    def decode_mask(self, mask):
        mask = tf.image.decode_png(mask, channels=1)

        return mask

    @tf.function
    def resize_crop(self, image, seed, channels):
        dimension = tf.cond(tf.math.greater(tf.shape(image)[0], tf.shape(image)[1]), lambda: tf.shape(image)[1], lambda: tf.shape(image)[0])

        cropped_image = tf.image.random_crop(
            image,
            size=[dimension, dimension, channels],
            seed=seed
        )
        image = tf.image.resize(
            cropped_image,
            [self.target_height, self.target_width],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

        # OR
        # image = tf.image.resize_with_pad(image, self.target_height, self.target_width)

        return image

    @tf.function
    def resize(self, image):
        # it is much much faster with normal resize
        return tf.image.resize(image, [self.target_height, self.target_width])

    def load_and_preprocess_image(self, path):
        image = self.read_file(path)
        image = self.decode_image(image)

        if self.is_only_resize:
            image = self.resize(image)
        else:
            image = self.resize_crop(image, self.seed, channels=3)

        image = self.normalize(image)

        return image

    def load_and_preprocess_mask(self, path):
        mask = self.read_file(path)
        mask = self.decode_mask(mask)
        if self.is_only_resize:
            mask = self.resize(mask)
        else:
            mask = self.resize_crop(mask, self.seed, channels=1)

        return mask

    def create_dataset_train_val_test(self, path_dataset, txt_training_data, txt_validation_data, txt_test_data):
        """
        Loads training, validation, and test datasets
        Args:
            path_dataset: path to dataset folder
            path_training_data: path to training data
            path_validation_data: path to validation data
            path_test_data: path to test data

        Returns: dictionary containing train, validation, and test datasets

        """
        for set_txt, mode in zip([txt_training_data, txt_validation_data, txt_test_data], ['train', 'val', 'test']):
            with open(os.path.join(self.cfg["dataset_path"], set_txt), 'r') as fr:
                pairs = fr.read().splitlines()

                img_paths, lb_paths = [], []
                for pair in pairs:
                    imgpth, lbpth = pair.split(',')
                    img_paths.append(os.path.join(path_dataset, imgpth))
                    lb_paths.append(os.path.join(path_dataset, lbpth))

                if mode == 'train':
                    assert len(img_paths) == len(lb_paths)
                    print(f"The training dataset contains {len(img_paths)} images.")

                    self.length_training_data = len(img_paths)

                    train_img_ds = tf.data.Dataset.from_tensor_slices(img_paths)
                    train_msk_ds = tf.data.Dataset.from_tensor_slices(lb_paths)

                    self.seed = np.random.randint(0, 100)
                    train_img_ds = train_img_ds.map(self.load_and_preprocess_image)
                    train_msk_ds = train_msk_ds.map(self.load_and_preprocess_mask)

                    train_dataset = tf.data.Dataset.zip((train_img_ds, train_msk_ds))

                elif mode == 'val':
                    assert len(img_paths) == len(lb_paths)
                    print(f"The validation dataset contains {len(img_paths)} images.")

                    self.length_validation_data = len(img_paths)

                    val_img_ds = tf.data.Dataset.from_tensor_slices(img_paths)
                    val_msk_ds = tf.data.Dataset.from_tensor_slices(lb_paths)

                    val_img_ds = val_img_ds.map(self.load_and_preprocess_image)
                    val_msk_ds = val_msk_ds.map(self.load_and_preprocess_mask)

                    val_dataset = tf.data.Dataset.zip((val_img_ds, val_msk_ds))

                else:
                    assert len(img_paths) == len(lb_paths)
                    print(f"The test dataset contains {len(img_paths)} images.")

                    self.length_test_data = len(img_paths)

                    test_img_ds = tf.data.Dataset.from_tensor_slices(img_paths)
                    test_msk_ds = tf.data.Dataset.from_tensor_slices(lb_paths)

                    test_img_ds = test_img_ds.map(self.load_and_preprocess_image)
                    test_msk_ds = test_msk_ds.map(self.load_and_preprocess_mask)

                    test_dataset = tf.data.Dataset.zip((test_img_ds, test_msk_ds))

        train_val_test_dataset = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        }

        return train_val_test_dataset


class TrainUNet:
    def __init__(self):
        self.cfg = CFG
        self.seed = 42

        self.create_dataset = CreateDataset()

        self.train_batch_set, self.val_batch_set, self.test_batch_set = self.prepare_dataset()

    def prepare_dataset(self):
        dataset = self.create_dataset.create_dataset_train_val_test(
            self.cfg["dataset_path"],
            self.cfg["train_data_txt"],
            self.cfg["validation_data_txt"],
            self.cfg["test_data_txt"]
        )

        # -- Training Dataset --#
        train_dataset = dataset['train']
        batch_train_dataset = train_dataset.batch(self.cfg["batch_size"])

        # -- Validation Dataset --#
        val_dataset = dataset['val']
        batch_val_dataset = val_dataset.batch(self.cfg["batch_size"])

        # -- Test Dataset --#
        test_dataset = dataset['test']
        batch_test_dataset = test_dataset.batch(self.cfg["batch_size"])

        return batch_train_dataset, batch_val_dataset, batch_test_dataset

    def get_model(self, input_size, initializer='he_normal'):
        # -- Encoder -- #
        # Block encoder 1
        inputs = tf.keras.layers.Input(shape=input_size)
        conv_enc_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(inputs)
        conv_enc_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_1)

        # Block encoder 2
        max_pool_enc_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_enc_1)
        conv_enc_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(max_pool_enc_2)
        conv_enc_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_2)

        # Block  encoder 3
        max_pool_enc_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_enc_2)
        conv_enc_3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(max_pool_enc_3)
        conv_enc_3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_3)
        # -- Encoder -- #

        # ----------- #
        max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_enc_3)
        conv = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(max_pool)
        conv = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv)
        # ----------- #

        # -- Decoder -- #
        # Block decoder 1
        up_dec_1 = tf.keras.layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=initializer)(tf.keras.layers.UpSampling2D(size=(2, 2))(conv))
        merge_dec_1 = tf.keras.layers.concatenate([conv_enc_3, up_dec_1], axis=3)
        conv_dec_1 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(merge_dec_1)
        conv_dec_1 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_1)

        # Block decoder 2
        up_dec_2 = tf.keras.layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=initializer)(tf.keras.layers.UpSampling2D(size=(2, 2))(conv_dec_1))
        merge_dec_2 = tf.keras.layers.concatenate([conv_enc_2, up_dec_2], axis=3)
        conv_dec_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(merge_dec_2)
        conv_dec_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_2)

        # Block decoder 3
        up_dec_3 = tf.keras.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=initializer)(tf.keras.layers.UpSampling2D(size=(2, 2))(conv_dec_2))
        merge_dec_3 = tf.keras.layers.concatenate([conv_enc_1, up_dec_3], axis=3)
        conv_dec_3 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(merge_dec_3)
        conv_dec_3 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_3)
        conv_dec_3 = tf.keras.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_3)
        # -- Decoder -- #

        output = tf.keras.layers.Conv2D(self.cfg["n_classes"], 1, activation='softmax')(conv_dec_3)

        return tf.keras.Model(inputs=inputs, outputs=output)

    def train(self):
        model = self.get_model([self.create_dataset.target_height, self.create_dataset.target_width, 3])
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer="adam",
            metrics=["accuracy"]
        )

        history = model.fit(
            self.train_batch_set,
            validation_data=self.val_batch_set,
            epochs=self.cfg["epochs"]
        )


def main():
    train_unet = TrainUNet()
    train_unet.train()


if __name__ == '__main__':
    main()