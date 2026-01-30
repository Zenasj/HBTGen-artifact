from tensorflow import keras
from tensorflow.keras import layers

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        {"input_1": atr},
        {"ed": wtr, "sd": wbtr},
    )
)
train_dataset = train_dataset.batch(100).repeat(3)

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Input, Model
input1 = tf.keras.layers.Input(shape=(None,1),name="input_1")
x = tf.keras.layers.Conv1D(filters=16, kernel_size=3, strides=1, padding="causal", activation="relu",input_shape=[None,1])(input1)
x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, activation="tanh", return_sequences=True))(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, activation="tanh", return_sequences=True))(x)
x = tf.keras.layers.Dense(128, activation="tanh")(x)
o1 = tf.keras.layers.Dense(1, activation="linear",name="ed")(x)
o2 = tf.keras.layers.Dense(1, activation="sigmoid",name="sd")(x)

model = Model(inputs=[input1], outputs=[o1, o2])

model.compile(loss={'ed': 'mean_squared_error', 
                    'sd': 'binary_crossentropy'},
              loss_weights={'ed':0.4,
                            'sd':0.6},
              optimizer='adam',
              metrics={'ed': tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error", dtype=None),
                       'sd': tfa.metrics.F1Score(name="f1_score",num_classes=2, threshold=0.5)})

history = model.fit(train_dataset,epochs=3,verbose=1,steps_per_epoch= 78)