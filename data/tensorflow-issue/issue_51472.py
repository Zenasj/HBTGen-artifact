import numpy as np
import random
import tensorflow as tf
from tensorflow import keras

class TripletDataGenerator(tf.keras.utils.Sequence):
    """
    NOTE: ON model.fit(SHUFFLE=FALSE) -> MUST BE FALSE!
    """

    def __init__(self,
                 df: pd.DataFrame,
                 batch_size=256,
                 shuffle=True,
                 rescale: {float, None} = 1. / 255.,
                 target_img_size=(128, 128),
                 preprocess_function=None,
                 rand_preproc_single: {ImageDataGenerator, dict} = None,
                 rand_preproc_batch: list = None):
        self.batch_counter = 0
        self.last_batch_index = 0
        if shuffle:
            self.triplets_df = df.sample(frac=1).reset_index(drop=True)  # randomizing it
        else:
            self.triplets_df = df.reset_index(drop=True)  # randomizing it
        self.preprocess_function = preprocess_function
        self.rescale = rescale
        self.target_img_size = target_img_size

        assert batch_size > 0, "Minimum batch size is 1, must be positive."
        self.batch_size = batch_size
        self.shuffle = shuffle
        # indexes of rows. every batch we draw 2 samples. 1 similar and 1 dissimilar
        self.indexes = np.arange(len(self.triplets_df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

        self.rand_preproc_single = rand_preproc_single
        self.rand_preproc_batch = rand_preproc_batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.triplets_df) // self.batch_size

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:
                               (index + 1) * self.batch_size]
        self.last_batch_index = index

        anchors = []
        positives = []
        negatives = []
        for idx in indexes:
            a, p, n = self.triplets_df.iloc[idx]
            anchors.append(self.load_image(a))
            positives.append(self.load_image(p))
            negatives.append(self.load_image(n))

        anchors = np.array(anchors, dtype='float32')
        positives = np.array(positives, dtype='float32')
        negatives = np.array(negatives, dtype='float32')
        labels = np.zeros(self.batch_size)
        if self.rand_preproc_batch is not None:
            for func in self.rand_preproc_batch:
                anchors = func(anchors)
                positives = func(positives)
                negatives = func(negatives)

        return [anchors, positives, negatives], labels

    def on_epoch_end(self):
        """Updates indexes after each epoch"""

        self.batch_counter += self.last_batch_index + 1  # indices starts from 0
        if self.batch_counter >= len(self):
            if self.shuffle:
                np.random.shuffle(self.indexes)
                self.triplets_df = self.triplets_df.sample(frac=1).reset_index(drop=True)
            self.batch_counter = 0
        else:
            self.indexes = np.append(self.indexes[self.last_batch_index + 1:],
                                     self.indexes[: self.last_batch_index + 1])

    def load_image(self, path):
        """
        loads an image using tensorflow tools
        :param path: absolute path (refers to the project's folder) to the image
        :return: an image array.
        """
        if self.rand_preproc_single is not None:
            if isinstance(self.rand_preproc, ImageDataGenerator):
                img_arr = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                img_arr = self.rand_preproc.random_transform(img_arr)
                img_arr = cv2.resize(img_arr, self.target_img_size)
            else:
                img_arr = my_utils.image_augmentations(path, **self.rand_preproc)
        else:
            img_arr = cv2.imread(path)
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            img_arr = cv2.resize(img_arr, self.target_img_size)
        if self.preprocess_function is not None:
            img_arr = self.preprocess_function(img_arr)
        elif self.rescale is not None:
            img_arr = img_arr * self.rescale
        return img_arr

def get_siamese_model(input_shape, conv2d_filters):
    # Define the tensors for the two input images
    anchor_input = Input(input_shape, name="Anchor_Input")
    positive_input = Input(input_shape, name="Positive_Input")
    negative_input = Input(input_shape, name="Negative_Input")

    body = build_body(input_shape, conv2d_filters)

    # Generate the encodings (feature vectors) for the two images
    encoded_a = body(anchor_input)
    encoded_p = body(positive_input)
    encoded_n = body(negative_input)

    ap_distance = tf.reduce_sum(tf.square(encoded_a - encoded_p), axis=-1, keepdims=True)
    an_distance = tf.reduce_sum(tf.square(encoded_a - encoded_n), axis=-1, keepdims=True)
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[anchor_input, positive_input, negative_input],
                        outputs=[ap_distance, an_distance])
    return siamese_net

def get_loss(margin=1.0):
    def triplet_loss(y_true, y_pred):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = y_pred

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + margin, 0.0)
        return loss

    return triplet_loss

if __name__ == '__main__':

    EPOCHS = configurations.EPOCHS
    BATCH_SIZE = configurations.BATCH_SIZE
    IMG_SIZE = configurations.IMG_SIZE
    MONITOR = configurations.MONITOR
    PATIENCE = configurations.PATIENCE
    EMBEDDING_NODES = configurations.EMBEDDING_NODES
    LEARNING_RATE = configurations.LEARNING_RATE
    STEPS_PER_EPOCH = configurations.STEPS_PER_EPOCH
    VALIDATION_STEPS = configurations.VALIDATION_STEPS
    CONV2D_FILTERS = configurations.CONV2D_FILTERS
    MARGIN = configurations.MARGIN
    DATA_FILE = 'LFW_triplets.csv'
    augment_params = None
    # augment_params = dict(
    #     resize=IMG_SIZE[:-1],
    #     random_gray_scale=0.2,
    #     random_contrast_range=[0.65, 1.5],
    #     hsv_noise_max_amps=[0.02, 0.2, 0],
    #     max_brightness_delta=0.15,
    #     LR_flip=True,
    #     UD_flip=False,
    #     rotate_range=30,
    #     random_shift=[0.1, 0.1],
    #     random_zoom_range=0.3,
    #     rescale=1. / 255.)

    NOTES = '\n'.join([
        f"batch size={BATCH_SIZE}",
        f"learning_rate={LEARNING_RATE}",
        f"embedding_nodes={EMBEDDING_NODES}",
        f"DataFilePath={DATA_FILE}",
        f"BatchNormalization_used={configurations.ADD_BATCHNORM}",
        f"Conv2D_filters_count={CONV2D_FILTERS}",
        f"Loss=triplet_loss",
        "Augmentations with: HSV, Brightness, Contrast."
    ])

    np.random.seed(42)
    tf.random.set_seed(42)

    t = time.time()
    train_gen, test_gen = DataFrameGeneratorClass.create_train_test_generators(csv_path=DATA_FILE, pair_gen=False,
                                                                               validation_split=0.3, shuffle=True,
                                                                               batch_size=configurations.BATCH_SIZE,
                                                                               rescale=1. / 255.,
                                                                               img_size=configurations.IMG_SIZE[:-1],
                                                                               preprocess_func=None,
                                                                               rand_preproc_single=None,
                                                                               rand_preproc_batch=None)

    dt = time.time() - t
    print(f"TOOK {dt} seconds to create Train Generator with {len(train_gen)} Batches"
          f" and Test Generator with {len(test_gen)} Batches")

    siamese_model = get_siamese_model(input_shape=IMG_SIZE, conv2d_filters=CONV2D_FILTERS)
    siamese_model.summary()
    loss_func = get_loss(margin=MARGIN)
    siamese_model.compile(optimizer=Adam(learning_rate=0.0001),
                          loss=loss_func)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor=MONITOR,
                                                  min_delta=1e-5,
                                                  patience=PATIENCE,
                                                  verbose=1,
                                                  mode='auto',
                                                  restore_best_weights=True)

    date_string = datetime.datetime.today().strftime("%d-%m-%y_%H%M%S")
    os.mkdir(f'check_points/{date_string}/')
    chk_point = tf.keras.callbacks.ModelCheckpoint(f'check_points/{date_string}/',
                                                   monitor=configurations.MONITOR,
                                                   verbose=1,
                                                   save_best_only=True,
                                                   save_weights_only=True)

    call_backs = [early_stop, chk_point, tf.keras.callbacks.TensorBoard(log_dir=f'logs/TRIPLET_{date_string}', write_images=True)]

    history = siamese_model.fit(train_gen,
                                shuffle=False,  # ITS MANDATORY WHEN USING MY CUSTOM GENERATOR
                                epochs=EPOCHS,
                                steps_per_epoch=STEPS_PER_EPOCH,
                                validation_steps=VALIDATION_STEPS,
                                callbacks=call_backs,
                                validation_data=test_gen)

    NOTES += f"\n\n{chk_point.monitor}={chk_point.best}"
    my_utils.save_results(notes=NOTES,
                          history_obj=history,
                          directory_dst='results',
                          model=siamese_model,
                          date_str="TRIPLET_" + date_string)