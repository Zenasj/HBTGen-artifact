import numpy as np
import tensorflow as tf

use_model = hub.Module("http://tfhub.dev/google/universal-sentence-encoder/1", trainable=True);
sess.run(tf.global_variables_initializer());
sess.run(tf.tables_initializer());

def USEEmbedding(x):
    return use_model(tf.squeeze(tf.cast(x, tf.string)), 
                      signature="default", as_dict=True)["default"]
 
input_text = layers.Input(shape=(1,), dtype=tf.string)
embedding = layers.Lambda(USEEmbedding, output_shape=(512,))(input_text)
dense = layers.Dense(1024, activation='relu')(embedding)
bnorm = layers.BatchNormalization()(dense)
pred = layers.Dense(2000, activation='softmax')(bnorm)

model = Model(inputs=[input_text], outputs=pred)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

callbacks = [keras.callbacks.EarlyStopping(monitor='val_acc',
                                               min_delta=1e-3,
                                               patience=8,
                                               verbose=0,
                                               mode='auto'),
             keras.callbacks.ModelCheckpoint('../models/best-weights.h5',
                                                 monitor='val_acc',
                                                 verbose=1,
                                                 save_best_only=True,
                                                 mode='auto'),
             keras.callbacks.TensorBoard(log_dir='../tb-logs', histogram_freq=0,
                                         write_graph=True, write_images=False)]

train_text = [' '.join(t.split()[0:20]) for t in train_x.sentences.tolist()]
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
valid_text = [' '.join(t.split()[0:20]) for t in valid_x.sentences.tolist()]
valid_text = np.array(valid_text, dtype=object)[:, np.newaxis]

model.fit(train_text , train['labels'].tolist(),
          validation_data=(valid_text, valid['labels'].tolist()),
          epochs=100, batch_size=256,
          callbacks=callbacks, shuffle=True)

input_text = layers.Input(shape=(1,), dtype=tf.string)

input_text = layers.Input(shape=(1,), dtype="string")