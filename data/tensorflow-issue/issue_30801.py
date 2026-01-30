import numpy as np
import random
import tensorflow as tf
from tensorflow import keras

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, affect_net_dir, data_file, batch_size, aug=True):
        self.affect_net_dir = affect_net_dir
        with FileIO(os.path.join(self.affect_net_dir, data_file), 'r') as fr:
            self.data = pd.read_csv(fr)
        self.batch_size = batch_size
        self.aug = aug
        self.indexes = np.arange(self.data.shape[0])
        np.random.shuffle(self.indexes)

    @property
    def steps(self):
        return int(np.floor(self.data.shape[0] / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        x = np.zeros(shape=(self.batch_size, 224, 224, 3))
        y = np.zeros(shape=(self.batch_size, 11))

        for i, b_index in enumerate(batch_indexes):
            file_path = self.data.iloc[b_index]['subDirectory_filePath']
            with FileIO(os.path.join(self.affect_net_dir, file_path), 'rb') as fr:
                face_img = Image.open(BytesIO(fr.read()))
            face_x = self.data.iloc[b_index]['face_x']
            face_y = self.data.iloc[b_index]['face_y']
            face_width = self.data.iloc[b_index]['face_width']
            face_height = self.data.iloc[b_index]['face_height']
            expression = self.data.iloc[b_index]['expression']
            # crop face from image
            face_img = face_img.crop([face_x, face_y, face_x + face_width, face_y + face_height])
            face_img = face_img.resize(size=(224, 224), resample=Image.LANCZOS)
            face_img = tf.keras.preprocessing.image.img_to_array(face_img)

            # face augmentation if needed
            if self.aug:
                if np.random.rand() >= 0.5:
                    face_img = face_aug.random_brightness(face_img)
                if np.random.rand() >= 0.5:
                    face_img = face_aug.random_hue(face_img)
                if np.random.rand() >= 0.5:
                    face_img = face_aug.random_saturation(face_img)
                if np.random.rand() >= 0.5:
                    face_img = face_aug.random_contrast(face_img)
                if np.random.rand() >= 0.5:
                    face_img = face_aug.horizontal_mirror(face_img)

            x[i] = face_img
            y[i] = tf.keras.utils.to_categorical(expression, num_classes=11)

        return tf.keras.applications.resnet50.preprocess_input(x), y

    def __len__(self):
        return int(np.floor(self.data.shape[0] / self.batch_size))

epochs = 80

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(train_data_gen.n / float(batch_size))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(val_data_gen.n / float(batch_size))),
    use_multiprocessing=True,
    workers=3,
)