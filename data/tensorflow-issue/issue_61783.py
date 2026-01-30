import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

# fine-tunining baseline model for few epochs
history = run_model(args, save_model_file)
    
# load trained weights for quantization aware training
model = setup_pretrained_model(args, save_model_file)
    
def apply_quantization_to_dense(layer):
    if isinstance(layer, tf.keras.layers.Dense):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer

annotated_model = tf.keras.models.clone_model(
    model,
    clone_function=apply_quantization_to_dense,
)

# Build Model
annotated_model.build((None, args.input_dim, args.input_dim ,3))

# Now that the Dense layers are annotated,
# `quantize_apply` actually makes the model quantization aware.
quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)

#     quant_aware_model.summary()
n_sample, train_ds, val_ds = load_data(args, args.input_dim)

# `quantize_model` requires a recompile.
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    args.lr / 10.0,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)
    
# Compile the model
quant_aware_model.compile(optimizer=tf.keras.optimizers.Lion(lr=lr_schedule), 
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
              metrics=['accuracy']
              )

def build_model(args):
    IMG_SHAPE = (args.input_dim, args.input_dim, 3)
    # Transfer learning model with MobileNetV3
    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet',
        minimalistic=True,
#         include_preprocessing=False
    )
    # Freeze the pre-trained model weights
    base_model.trainable = False
    x = tf.keras.layers.GlobalMaxPooling2D()(base_model.output)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
    x = tf.keras.layers.Dense(args.num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(base_model.input, x)

    return model


def setup_pretrained_model(args, ckpt_path):
    """
    Function to load pretrained model
    """
    
    model = build_model(args)
    model.load_weights(ckpt_path)
    
    return model


def run_model(args, ckpt_path):
    # build base model architecture
    model = build_model(args)
    label_list = os.listdir(args.input_path)
    
    # Compile the model    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args.lr,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
    model.compile(optimizer=tf.keras.optimizers.Lion(lr=args.lr), 
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
                  metrics=['accuracy']
                  )
    
    # load data
    n_sample, train_ds, val_ds = load_data(args, args.input_dim)

    
    hist = model.fit(train_ds,
                 epochs=args.epochs,
                 validation_data=val_ds,
                 steps_per_epoch=n_sample // args.batch_size,
                 validation_steps=val_ds.n // args.batch_size,
                 callbacks=[checkpoint_callback(ckpt_path),
                            early_stopping(),
                            logging_callback(args.log_dir),
                            save_metadata_callback(ckpt_path, label_list)],
                 verbose=1)


def fit_model(args):
    save_model_file = args.model_path + '/model_best.h5'
    qat_model_file = args.model_path + '/qat_model_best.h5'
    label_list = os.listdir(args.input_path)
    # fine-tunining baseline model for few epochs
    history = run_model(args, save_model_file)
    
    # load trained weights for quantization aware training
    model = setup_pretrained_model(args, save_model_file)
    
    def apply_quantization_to_dense(layer):
        if isinstance(layer, tf.keras.layers.Dense):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        return layer

    annotated_model = tf.keras.models.clone_model(
        model,
        clone_function=apply_quantization_to_dense,
    )
    
    # Now that the Dense layers are annotated,
    # `quantize_apply` actually makes the model quantization aware.
    quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    n_sample, train_ds, val_ds = load_data(args, args.input_dim)

    # `quantize_model` requires a recompile.
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args.lr / 10.0,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
    
    # Compile the model
    quant_aware_model.compile(optimizer=tf.keras.optimizers.Lion(lr=lr_schedule), 
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
                  metrics=['accuracy']
                  )
    
    hist = quant_aware_model.fit(train_ds,
                                 epochs=args.epochs,
                                 validation_data=val_ds,
                                 steps_per_epoch=n_sample // args.batch_size,
                                 validation_steps=val_ds.n // args.batch_size,
                                 callbacks=[checkpoint_callback(qat_model_file),
                                            early_stopping(),
                                            logging_callback(args.log_dir),
                                            save_metadata_callback(qat_model_file, label_list)],
                                 verbose=1)

    plot_hist(hist)
    quant_aware_model.save(args.model_path + '/qat_model_last.h5')

class SaveMetadataCallback(Callback):
    def __init__(self, filepath, labels):
        super(SaveMetadataCallback, self).__init__()
        self.filepath = filepath
        self.labels = labels

    def on_epoch_end(self, epoch, logs=None):
        # Save after each epoch
        with h5py.File(self.filepath, 'a') as f:
            if 'label_names' not in f:
                label_dataset = f.create_dataset('label_names', data=self.labels)

    def on_train_end(self, logs=None):
        # OR Save after training completion
        with h5py.File(self.filepath, 'a') as f:
            if 'label_names' not in f:
                label_dataset = f.create_dataset('label_names', data=self.labels)


def save_metadata_callback(ckpt_path, labels):
    return SaveMetadataCallback(filepath=ckpt_path, labels=labels)

# Create a callback that saves the model's checkpoints
def checkpoint_callback(ckpt_path):
    return tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=True,
                                              save_weights_only=False,
                                              mode='auto',
                                              save_freq='epoch')



def early_stopping():
    return tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            verbose=1)


def logging_callback(log_dir):
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                          histogram_freq=0,
                                          write_graph=True,
                                          write_images=False,
                                          update_freq='epoch',
                                          profile_batch=2,
                                          embeddings_freq=0,
                                          embeddings_metadata=None)

def load_data(args, input_size):
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
        brightness_range=[0.8, 1.2],
        preprocessing_function=tray_crop,
        validation_split=0.2)