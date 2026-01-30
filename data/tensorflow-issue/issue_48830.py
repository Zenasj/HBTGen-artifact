import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_classifier_model(train_dataset):
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing', )
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  pooled = outputs['pooled_output']

  net = tf.keras.layers.Dense(
              HIDDEN_LAYER_DIMS,
              kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.002),
              activation="relu",
              name="pre_classifier"
          )(pooled)  
  
  net = tf.keras.layers.Dropout(DROPOUT, trainable=True)(net)
  net = tf.keras.layers.Dense(2, activation="sigmoid", name='classifier')(net)
  model = tf.keras.Model(text_input, net)

  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  epochs = EPOCHS
  steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
  num_train_steps = steps_per_epoch * epochs
  num_warmup_steps = int(0.1*num_train_steps)

  optimizer = optimization.create_optimizer(init_lr=LEARNING_RATE,
                                            num_train_steps=num_train_steps,
                                            num_warmup_steps=num_warmup_steps,
                                            optimizer_type='adamw')

  model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=['accuracy'])
  
  model.summary()
  
  return model

model = build_classifier_model(train_dataset)

checkpoint = ModelCheckpoint(filepath=SAVE_MODEL_PATH, 
                             verbose=1,
                             save_freq='epoch',
                             monitor='val_accuracy',
                             save_best_only=True, 
                             mode='max', 
                             save_weights_only=True)


# Training model...
history = model.fit(train_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=checkpoint, validation_data=valid_dataset)