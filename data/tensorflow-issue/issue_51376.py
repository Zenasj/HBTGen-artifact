import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_siamese_model(input_shape, conv2d_filts):
    # Define the tensors for the two input images
    # ================================= THE INNER MODEL =================================
    augmentations = Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.70),
            RandomBrightnessLayer(max_delta=0.1, name='RandomBrightness'),
            RandomHSVLayer(hsv_max_amp=[0.05, 0.25, 0], name='RandomHSVPreprocessor'),
        ],
        name=configurations.AUGMENTATIONS_LAYER_NAME
    )

    # ================================= THE INNER MODEL =================================
    # THE PROBLEM IS HERE, WHEN NOT SPECIFING 'batch_size' TO INPUT LAYER.
    left_input = Input(input_shape, name="Input1")
    right_input = Input(input_shape, name="Input2")
    left_input_augmented = augmentations(left_input)
    right_input_augmented = augmentations(right_input)

    # Generate the encodings (feature vectors) for the two images
    body = build_body(input_shape=input_shape, conv2d_filts=conv2d_filts)
    encoded_l = body(left_input_augmented)
    encoded_r = body(right_input_augmented)
    distance = Lambda(lambda embeds: euclidean_distance(embeds), name='Distance')([encoded_l, encoded_r])
    # normed_layer = BatchNormalization()(distance)  # making sure the distances wont be all over the place.
    distance = Dense(1, activation='sigmoid', name='Prediction')(distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=distance)

    return siamese_net


# DataFrameGeneratorClass is a custom pair image generator, since nither Keras or Tensorflow has one.
train_gen, test_gen = DataFrameGeneratorClass.create_train_test_generators(
    csv_path='data.csv',
    validation_split=0.1,
    shuffle=True,
    batch_size=32,
    rescale=1. / 255.,
    img_size=(128, 128),
)
siamese_model = get_siamese_model(IMG_SIZE, conv2d_filts=CONV2D_FILTERS)
siamese_model.summary()

# siamese_model.load_weights('check_points/29-07-21_034351/')
optimizer = Adam()
siamese_model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
history = siamese_model.fit(train_gen, epochs=10, validation_data=test_gen)

class PairDataGenerator(tf.keras.utils.Sequence):
    """
    NOTE: ON model.fit(SHUFFLE=FALSE) -> MUST BE FALSE!
    """

    def __init__(self,
                 df_similar: pd.DataFrame,
                 df_dissimilar: pd.DataFrame,
                 batch_size=256,
                 shuffle=True,
                 rescale: {float, None} = 1. / 255.,
                 target_img_size=(128, 128),
                 preprocess_function=None,
                 rand_preproc_single: {ImageDataGenerator, dict} = None,
                 rand_preproc_batch: list = None):
        # self.batch_counter = 0
        self.last_batch_index = 0
        self.df_similar = df_similar.sample(frac=1).reset_index(drop=True)
        self.df_dissimilar = df_dissimilar.sample(frac=1).reset_index(drop=True)
        self.preprocess_function = preprocess_function
        self.rescale = rescale
        self.target_img_size = target_img_size
        # rounding up batch size to be an even number.
        self.batch_size = batch_size + (batch_size % 2 == 1)
        self.shuffle = shuffle
        # indexes of rows. every batch we draw 2 samples. 1 similar and 1 dissimilar
        assert batch_size <= len(self.df_similar) + len(
            self.df_dissimilar), f"Cannot create batch of size {batch_size} when there " \
                                 f"are only {len(self.df_similar)} samples"
        self.indexes_similar = np.arange(len(self.df_similar))
        self.similar_max_idx = len(self.indexes_similar) // self.batch_size
        self.indexes_dissimilar = np.arange(len(self.df_dissimilar))
        self.dissimilar_max_idx = len(self.indexes_dissimilar) // self.batch_size
        if self.shuffle:
            np.random.shuffle(self.indexes_dissimilar)
            np.random.shuffle(self.indexes_similar)

        self.rand_preproc_single = rand_preproc_single
        self.rand_preproc_batch = rand_preproc_batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return (len(self.df_similar) + len(self.df_dissimilar)) // self.batch_size

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        batch_idx_sim = index % self.similar_max_idx
        indexes_sim = self.indexes_similar[batch_idx_sim * (self.batch_size // 2):
                                           (batch_idx_sim + 1) * (self.batch_size // 2)]

        batch_idx_dissim = index % self.dissimilar_max_idx
        indexes_dissim = self.indexes_dissimilar[batch_idx_dissim * (self.batch_size // 2):
                                                 (batch_idx_dissim + 1) * (self.batch_size // 2)]
        self.last_batch_index = index

        img1 = []
        img2 = []
        labels = [0, 1] * (self.batch_size // 2)  # creating labels list
        np.random.shuffle(labels)
        same_counter = 0
        diff_counter = 0
        for idx, label in enumerate(labels):
            if label == configurations.LABELS['same']:
                img1_path, img2_path, _ = self.df_similar.iloc[indexes_sim[same_counter]]
                same_counter += 1
            else:
                img1_path, img2_path, _ = self.df_dissimilar.iloc[indexes_dissim[diff_counter]]
                diff_counter += 1

            img1.append(self.load_image(img1_path))
            img2.append(self.load_image(img2_path))

        img1 = np.array(img1, dtype='float32')
        img2 = np.array(img2, dtype='float32')
        labels = np.array(labels, dtype='float32')

        if self.rand_preproc_batch is not None:
            for func in self.rand_preproc_batch:
                img1 = func(img1)
                img2 = func(img2)

        return [img1, img2], labels

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if (self.last_batch_index + 1) % self.dissimilar_max_idx == 0 and self.shuffle:
            self.indexes_dissimilar = np.arange(len(self.df_dissimilar))
            np.random.shuffle(self.indexes_dissimilar)

        if (self.last_batch_index + 1) % self.similar_max_idx == 0 and self.shuffle:
            self.indexes_similar = np.arange(len(self.df_similar))
            np.random.shuffle(self.indexes_similar)


    def load_image(self, path):
        """
        loads an image using tensorflow tools
        :param path: absolute path (refers to the project's folder) to the image
        :return: an image array.
        """

        if self.rand_preproc_single is not None:
            if isinstance(self.rand_preproc_single, ImageDataGenerator):
                img_arr = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                img_arr = self.rand_preproc_single.random_transform(img_arr)
                img_arr = cv2.resize(img_arr, self.target_img_size)
            else:
                img_arr = my_utils.image_augmentations(path, **self.rand_preproc_single)
        else:
            img_arr = cv2.imread(path)
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            img_arr = cv2.resize(img_arr, self.target_img_size)
        if self.preprocess_function is not None:
            img_arr = self.preprocess_function(img_arr)
        elif self.rescale is not None:
            img_arr = img_arr * self.rescale
        return img_arr

def create_train_test_generators(csv_path: str,
                                 pair_gen: bool = True,
                                 validation_split: float = 0.1,
                                 shuffle: bool = True,
                                 batch_size: int = 256,
                                 rescale: {float, None} = 1. / 255.,
                                 img_size: tuple = (128, 128),
                                 preprocess_func=None,
                                 rand_preproc_single: {ImageDataGenerator, dict} = None,
                                 rand_preproc_batch: list = None,
                                 ):
    """
    Initialization
    :param rand_preproc_batch: list of functions which augments a whole batch.
    :param rand_preproc_single: an ImageDataGenerator Instance or dictionary for my_utils.image_augmentation function.
    :param pair_gen: boolean. True = Pair Gen. False = Triplets Gen.
    :param rescale: rescaling factor
    :param preprocess_func: a preprocessing function for the network's inputs.
    :param img_size: the size of the output image.
    :param batch_size: batch size
    :param shuffle: whether to shuffle the data before casting to Train Test.
    :param csv_path: the path to the csv file which we create DataFrame object from.
    :param validation_split: how much from the whole data goes to validation.
    """
    from sklearn.model_selection import train_test_split

    # read all the csv file
    df = pd.read_csv(csv_path, index_col=False)

    params = dict(batch_size=batch_size,
                  shuffle=shuffle,
                  rescale=rescale,
                  target_img_size=img_size,
                  preprocess_function=preprocess_func,
                  rand_preproc_single=rand_preproc_single,
                  rand_preproc_batch=rand_preproc_batch)
    if pair_gen:
        # split the rows to similar and dissimilar by label column
        df_similar = df.where(df['labels'] == 1.0).dropna().reset_index(drop=True)
        df_dissimilar = df.where(df['labels'] == 0.0).dropna().reset_index(drop=True)

        print(len(df_dissimilar), len(df_similar))
        print(f"Found {len(df)} pairs.")

        # split similar and dissimilar to train and test (4 groups)
        df_train_similar, df_test_similar = train_test_split(df_similar, test_size=validation_split, shuffle=shuffle)
        df_train_dissimilar, df_test_dissimilar = train_test_split(df_dissimilar, test_size=validation_split,
                                                                   shuffle=shuffle)

        # drop the index column, no need of that.
        df_train_similar = df_train_similar.reset_index(drop=True)
        df_test_similar = df_test_similar.reset_index(drop=True)
        df_train_dissimilar = df_train_dissimilar.reset_index(drop=True)
        df_test_similar = df_test_similar.reset_index(drop=True)

        # print(len(pd.merge(df_train_similar, df_train_dissimilar, how='inner', on=['img1_p', 'img2_p', 'labels'])))

        print(f"Total={len(df)}",
              f"Train={len(df_train_similar)} + {len(df_train_dissimilar)}",
              f"Test={len(df_test_similar)} + {len(df_test_dissimilar)}", sep='\n')
        return PairDataGenerator(df_similar=df_train_similar, df_dissimilar=df_train_dissimilar, **params), \
               PairDataGenerator(df_similar=df_test_similar, df_dissimilar=df_test_dissimilar, **params)

    else:
        print(f"Found {len(df)} triplets.")

        # split similar and dissimilar to train and test (4 groups)
        df_train, df_test = train_test_split(df,
                                             test_size=validation_split,
                                             shuffle=shuffle)

        # drop the index column, no need of that.
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

        # print(len(pd.merge(df_train_similar, df_train_dissimilar, how='inner', on=['img1_p', 'img2_p', 'labels'])))

        print(f"Total={len(df)}",
              f"Train={len(df_train)}",
              f"Test={len(df_test)}", sep='\n')

        return TripletDataGenerator(df=df_train, **params), \
               TripletDataGenerator(df=df_test, **params)