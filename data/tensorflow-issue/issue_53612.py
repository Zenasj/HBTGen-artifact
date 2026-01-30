import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, train_tf_data, val_tf_data, CLASSES, logs={}, **kwargs):
        super().__init__(**kwargs)
        # self.keras_metric = tf.keras.metrics.Mean("val_f1_after_epoch")
        self.train_tf_data = train_tf_data
        self.val_tf_data = val_tf_data
        # self.model = model
        self.CLASSES = CLASSES

    def on_epoch_end(self, epoch, logs={}):
        # self.keras_metric.reset_state()
        # for train data
        self.train_reports = test_model(model=self.model, data=self.train_tf_data, CLASSES=self.CLASSES)
        self.train_f1_after_epoch = self.train_reports['f1_score']
        self.train_recall_after_epoch = self.train_reports['recall']
        self.train_prec_after_epoch = self.train_reports['precision']

        # for val data
        self.val_reports = test_model(model=self.model, data=self.val_tf_data, CLASSES=self.CLASSES)
        self.val_f1_after_epoch = self.val_reports['f1_score']
        self.val_recall_after_epoch = self.val_reports['recall']
        self.val_prec_after_epoch = self.val_reports['precision']

        # saving train results to log dir
        logs["f1_after_epoch"]=self.train_f1_after_epoch
        logs['precision_after_epoch'] = self.train_prec_after_epoch
        logs['recall_after_epoch'] = self.train_recall_after_epoch
        
        # saving val results to log dir
        logs['val_f1_after_epoch'] = self.val_f1_after_epoch
        logs['val_precision_after_epoch'] = self.val_prec_after_epoch
        logs['val_recall_after_epoch'] = self.val_recall_after_epoch
        # self.keras_metric.update_state(self.val_f1_after_epoch)

        print('reports_after_epoch', self.train_reports)
        print('val_reports_after_epoch', self.val_reports)
        



with strategy.scope():
    pretrained_model = tf.keras.applications.MobileNetV2(
                                                    weights='imagenet',
                                                    include_top=False,
                                                    input_shape=[*IMAGE_SIZE, IMG_CHANNELS])
    pretrained_model.trainable = True #fine tuning
    q_aware_pretrained_model = tf.keras.models.clone_model(pretrained_model,
                                                          clone_function=apply_quantization_to_dense,)
    base_model = tf.keras.Sequential([
                            tf.keras.layers.Lambda(# Convert image from int[0, 255] to the format expect by this base_model
                            lambda data:tf.keras.applications.mobilenet.preprocess_input(
                                tf.cast(data, tf.float32)), input_shape=[*IMAGE_SIZE, 3]),
                            q_aware_pretrained_model,
                            tf.keras.layers.GlobalAveragePooling2D()])
    base_model.layers[1]._name = 'custom_mnet_trainable'
    base_model.add(tf.keras.layers.Dense(64, name='object_dense',kernel_regularizer=tf.keras.regularizers.l2(l2=0.1)))
    base_model.add(tf.keras.layers.BatchNormalization(scale=False, center = False))
    base_model.add(tf.keras.layers.Activation('relu', name='relu_dense_64'))
    base_model.add(tf.keras.layers.Dropout(rate=0.5, name='dropout_dense_64'))
    base_model.add(tf.keras.layers.Dense(32, name='object_dense_2',kernel_regularizer=tf.keras.regularizers.l2(l2=0.1)))
    base_model.add(tf.keras.layers.BatchNormalization(scale=False, center = False))
    base_model.add(tf.keras.layers.Activation('relu', name='relu_dense_32'))
    base_model.add(tf.keras.layers.Dropout(rate=0.4, name='dropout_dense_32'))
    base_model.add(tf.keras.layers.Dense(16, name='object_dense_16', kernel_regularizer=tf.keras.regularizers.l2(l2=0.1)))
    base_model.add(tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax', name='object_prob'))
    m1 = tf.keras.metrics.CategoricalAccuracy()
    m2 = tf.keras.metrics.Recall()
    m3 = tf.keras.metrics.Precision()

    m4 = Metrics(train_tf_data=train_data, val_tf_data=test_data, CLASSES=CLASS_NAMES)


    optimizers = [
        tfa.optimizers.AdamW(learning_rate=lr * .001 , weight_decay=wd),
        tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
            ]

    optimizers_and_layers = [(optimizers[0], base_model.layers[0]), (optimizers[1], base_model.layers[1:])]

    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

    annotated_model = tf.keras.models.clone_model(
        base_model,
        clone_function=apply_quantization_to_dense,
    )


    model = tfmot.quantization.keras.quantize_apply(annotated_model)
    model.compile(
        optimizer= optimizer, loss=tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO),
        metrics=[m1, m2, m3],
        )

tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

checkpoint_name = os.getcwd() + os.sep + CUSTOM_MODEL_PATH + os.sep + "training_chkpts/cp-{epoch:04d}-{val_f1_after_epoch:.2f}.ckpt"
checkpoint_dir_path  = os.getcwd() + os.sep + CUSTOM_MODEL_PATH + os.sep+ "training_chkpts"
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_name, 
                                                    monitor = 'val_f1_after_epoch',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    mode='max',
                                                    save_freq='epoch',
                                                    verbose=1)

checkpoint_cb._supports_tf_logs = False
current_dir = os.getcwd()
history = model.fit(train_data, validation_data=test_data, 
                    epochs=N_EPOCHS,
                    callbacks=[m4, checkpoint_cb, tensorboard_cb])

NUM_TRAINING_IMAGES = count_data_items(train_data)
NUM_TEST_IMAGES = count_data_items(test_data)

#TPU or GPU detection
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
    BUFFER_SIZE = NUM_TRAINING_IMAGES
    BATCH_SIZE_PER_REPLICA = BATCH_SIZE / strategy.num_replicas_in_sync
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
    VALIDATION_STEPS = -(-NUM_TEST_IMAGES // BATCH_SIZE) # The "-(-//)" trick rounds up instead of down :-)

except ValueError:
    strategy = tf.distribute.MirroredStrategy()
    # Defining tf_distribute strategy
    BUFFER_SIZE = NUM_TRAINING_IMAGES
    BATCH_SIZE_PER_REPLICA = BATCH_SIZE / strategy.num_replicas_in_sync
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
    VALIDATION_STEPS = -(-NUM_TEST_IMAGES // BATCH_SIZE) # The "-(-//)" trick rounds up instead of down :-)
    STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE_PER_REPLICA
    VALIDATION_STEPS = -(-NUM_TEST_IMAGES // BATCH_SIZE_PER_REPLICA) # The "-(-//)" trick rounds up instead of down :-)
    print('Dataset: training images, {} test images {}'.format(NUM_TRAINING_IMAGES, NUM_TEST_IMAGES))