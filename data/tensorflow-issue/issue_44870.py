import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

def build_mobilenet(img_vec):
        print("mobilenet loading..........")
        model = Sequential()
        base_mobilenet_model = MobileNet(input_shape = img_vec.shape[1:], 
                                 include_top = False, weights = None)
        model.add(Input(shape = img_vec.shape[1:], name='input_layer'))
        model.add(base_mobilenet_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(0.5))
        model.add(Dense(len(all_labels), activation = 'sigmoid'))
        METRICS = ["binary_accuracy", "top_k_categorical_accuracy", hn_multilabel_loss, tf.keras.metrics.AUC(), 'mae']
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = METRICS)
        model.summary()
        return model

def build_ensemble(img_vec):
    model_input = Input(shape = img_vec.shape[1:], name='input_layer_main')
    ## define fuctional blocks
    mobilenet1 = keras.models.load_model("models/mobilenet_1_30epochs_multi-label.augmented.h5", \
                                    custom_objects = {"hn_multilabel_loss" : hn_multilabel_loss})
    mobilenet1._name , mobilenet1.trainable = "model_1", False
    ###
    mobilenet2 = keras.models.load_model("models/mobilenet_2_30epochs_multi-label.augmented.h5", \
                                    custom_objects = {"hn_multilabel_loss" : hn_multilabel_loss})
    mobilenet2._name , mobilenet2.trainable = "model_2", False
    ###
    mobilenet3 = keras.models.load_model("models/mobilenet_3_30epochs_multi-label.augmented.h5", \
                                    custom_objects = {"hn_multilabel_loss" : hn_multilabel_loss})
    mobilenet3._name , mobilenet3.trainable = "model_3", False

    #merge 3 models
    model1, model2, model3=(mobilenet1(model_input), mobilenet2(model_input), mobilenet3(model_input))
    merge = concatenate([model1, model2, model3], name="concat_merge_123")
    output_layer = Dense(len(all_labels), activation = 'sigmoid', name = "output_layer")(merge)
    
    model = keras.models.Model(inputs= model_input, outputs= output_layer)## model assign

    OPTIMIZER = Adam(learning_rate=0.001,beta_1=0.9, beta_2=0.999)

    METRICS = ["binary_accuracy", "top_k_categorical_accuracy", hn_multilabel_loss, tf.keras.metrics.AUC(), 'mae']
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = METRICS)
    model.summary()
    return model

weight_path="models/ensemble_model_multilabel.best.h5"

patience_reduce_lr=1
min_lr=1e-8
output_dir="models/"
callbacks_list = [
            ModelCheckpoint(weight_path, monitor='val_accuracy', verbose=1, 
                             save_best_only=True, save_weights_only=False, mode='auto'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_reduce_lr,
                              verbose=1, mode="min", min_lr=min_lr),
            ]
ensemble_model = build_ensemble(t_x)
hist3 = ensemble_model.fit_generator(train_gen, 
                              steps_per_epoch=100,
                              validation_data = (test_X, test_Y), 
                              epochs = 30, 
                              callbacks = callbacks_list)