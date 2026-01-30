import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

model.evaluate(source_gen)
model.evaluate(target_gen)

model.save_weights("models/weights.h5")
model.save("models/test.h5")

model_2 = tf.keras.models.load_model("models/test.h5")
model_2.load_weights("models/weights.h5")
model_2.evaluate(source_gen)
model_2.evaluate(target_gen)

model.save("models/test.h5")
model_2 = tf.keras.models.load_model("models/test.h5")
model_2.evaluate(source_gen)
model_2.evaluate(target_gen)

model.save("models/test", save_format="tf")
model_2 = tf.keras.models.load_model("models/test")
model_2.evaluate(source_gen)
model_2.evaluate(target_gen)

input_range = Input(input_shape_range, name="input_range")
input_doppler = Input(input_shape_doppler, name="input_doppler")

r = Conv2D(filters=8, kernel_size=(3, 3), padding="same", activation="relu", name="c1_r",
               data_format="channels_first")(input_range)
r = Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu", name="c2_r",
               data_format="channels_first")(r)
r = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="Valid", name="m1_r",
                  data_format="channels_first")(r)
r = Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu", name="c3_r",
               data_format="channels_first")(r)
r = Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu", name="c4_r",
               data_format="channels_first")(r)
r = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="Valid", name="m2_r",
                  data_format="channels_first")(r)

features_range = Flatten(name="flatten_r")(r)

d = Conv2D(filters=8, kernel_size=(3, 3), padding="same", activation="relu", name="c1_d",
               data_format="channels_first")(input_doppler)
d = Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu", name="c2_d",
               data_format="channels_first")(d)
d = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="Valid", name="m1_d",
                  data_format="channels_first")(d)
d = Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu", name="c3_d",
               data_format="channels_first")(d)
d = Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu", name="c4_d",
               data_format="channels_first")(d)
d = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="Valid", name="m2_d",
                  data_format="channels_first")(d)

features_doppler = Flatten(name="flatten_d")(d)

features = tf.keras.layers.concatenate([features_range, features_doppler], name="features" ,axis=1)

fc_2 = Dense(128, activation="relu", use_bias=True, name="fc_2",
                 kernel_regularizer= tf.keras.regularizers.l2(0.01))(features)
drop_2 = Dropout(0.2)(fc_2)
fc_3 = Dense(64, activation="relu", use_bias=True, name="fc_3",
                 kernel_regularizer= tf.keras.regularizers.l2(0.01))(drop_2)
drop_3 = Dropout(0.2)(fc_3)
out = Dense(5, use_bias=True, name="out")(drop_3)

model = tf.keras.Model(inputs=[input_range, input_doppler], outputs=out)