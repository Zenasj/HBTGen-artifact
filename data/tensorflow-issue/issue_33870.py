from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

shape = (224, 224, 3)

# sequential model
model1 = tf.keras.Sequential(
            [
                tf.keras.Input(shape=shape, name="input"),
                tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=shape),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(256, activation="relu", name="descriptor"),
                tf.keras.layers.Dense(2, activation="softmax", name="probs"),
            ]
        )

# functional model
base_model2 = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=shape)
inputs = tf.keras.Input(shape=shape, name="input")
x = base_model2(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation="relu", name="descriptor")(x)
outputs = tf.keras.layers.Dense(2, activation="softmax", name="probs")(x)
model2 = tf.keras.Model(inputs=inputs, outputs=outputs)

tf.saved_model.save(model1, "test1")
tf.saved_model.save(model2, "test2")
#model2.save("test2", include_optimizer=False, save_format="tf")

model_1 = tf.keras.models.load_model('test1')

# THIS RAISES exception 
model_2 = tf.keras.models.load_model('test2')

import tensorflow as tf

shape = (224, 224, 3)

# sequential model
model1 = tf.keras.Sequential(
            [
                tf.keras.Input(shape=shape, name="input"),
                tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=shape),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(256, activation="relu", name="descriptor"),
                tf.keras.layers.Dense(2, activation="softmax", name="probs"),
            ]
        )

# functional model
base_model2 = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=shape)
inputs = tf.keras.Input(shape=shape, name="input")
x = base_model2(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation="relu", name="descriptor")(x)
outputs = tf.keras.layers.Dense(2, activation="softmax", name="probs")(x)
model2 = tf.keras.Model(inputs=inputs, outputs=outputs)

tf.saved_model.save(model1, "test1")
tf.saved_model.save(model2, "test2")
#model2.save("test2", include_optimizer=False, save_format="tf")

#model_1 = tf.keras.models.load_model('test1')
model_1 = tf.saved_model.load('test1')

# THIS RAISES exception 
#model_2 = tf.keras.models.load_model('test2')
model_2 = tf.saved_model.load('test2')

import tensorflow as tf

shape = (224, 224, 3)

# sequential model
model1 = tf.keras.Sequential(
    [
        tf.keras.Input(shape=shape, name="input"),
        tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=shape),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation="relu", name="descriptor"),
        tf.keras.layers.Dense(2, activation="softmax", name="probs"),
    ]
)

# functional model
base_model2 = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=shape)
inputs = tf.keras.Input(shape=shape, name="input")
x = base_model2(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation="relu", name="descriptor")(x)
outputs = tf.keras.layers.Dense(2, activation="softmax", name="probs")(x)
model2 = tf.keras.Model(inputs=inputs, outputs=outputs)

tf.keras.models.save_model(model1, "test1.h5", include_optimizer=False)
tf.keras.models.save_model(model2, "test2.h5", include_optimizer=False)

model_1 = tf.keras.models.load_model("test1.h5")

# THIS is passing
model_2 = tf.keras.models.load_model("test2.h5")

import tensorflow as tf

shape = (224, 224, 3)

# sequential model
model1 = tf.keras.Sequential(
    [
        tf.keras.Input(shape=shape, name="input"),
        tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=shape),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation="relu", name="descriptor"),
        tf.keras.layers.Dense(2, activation="softmax", name="probs"),
    ]
)

# functional model
base_model2 = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=shape)
inputs = tf.keras.Input(shape=shape, name="input")
x = base_model2(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation="relu", name="descriptor")(x)
outputs = tf.keras.layers.Dense(2, activation="softmax", name="probs")(x)
model2 = tf.keras.Model(inputs=inputs, outputs=outputs)

tf.keras.models.save_model(model1, "test1", include_optimizer=False)
tf.keras.models.save_model(model2, "test2", include_optimizer=False)

model_1 = tf.keras.models.load_model("test1")

# FAILS
model_2 = tf.keras.models.load_model("test2")