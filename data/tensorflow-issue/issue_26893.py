import tensorflow as tf
from tensorflow import keras

def seresnext_model(input_shape):
  base_model = SEResNextImageNet(input_shape,include_top = False)
  x = base_model.output
  out1 = GlobalMaxPooling2D()(x)
  out2 = GlobalAveragePooling2D()(x)
  out3 = Flatten()(x)
  out = concatenate([out1,out2,out3])
  out = Dropout(0.3)(out)
  out = Dense(256,activation = 'relu')(out)
  out = Dropout(0.3)(out)
  X = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', bias_initializer='zeros')(out)
  model =  Model(inputs=base_model.input, outputs=X)
  return model


tf.logging.set_verbosity(tf.logging.INFO)
tpu_model = tf.contrib.tpu.keras_to_tpu_model(
    seresnext_model,
    strategy=tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    )
)
tpu_model.compile(optimizer=tf.train.AdamOptimizer(learning_rate = 3e-4), loss=tf.keras.losses.binary_crossentropy, metrics=['acc'])
filepath = '/content/model.h5'
#clr = CyclicLR(base_lr=2e-4, max_lr=0.006,
#                     step_size=1070.)
checkpoint = ModelCheckpoint(filepath,monitor='val_loss', verbose=1, 
                             save_best_only=True)
history = tpu_model.fit_generator(train_gen,steps_per_epoch = train_steps,validation_data = val_gen,validation_steps = val_steps,epochs = 60,callbacks=[checkpoint],
                                  )