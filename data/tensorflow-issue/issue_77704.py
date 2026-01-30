import tensorflow as tf
from tensorflow import keras

num_classes = 2
base_model = tf.keras.applications.MobileNetV2(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=input_shape,
    pooling=None, 
)
for layer in base_model.layers:
    layer.trainable = False
base_model.summary()
cnn = models.Sequential()
cnn.add(base_model)
cnn.add(layers.GlobalAveragePooling2D())
cnn.add(layers.Dropout(0.2))
base_model.trainable = False
model = models.Sequential()
print(full_input_shape)
model.add(layers.TimeDistributed(cnn, input_shape=full_input_shape))
model.add(layers.LSTM(nbr_frame, return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(nbr_frame, activation='relu')))
model.add(layers.Flatten())
model.add(layers.Dense(164, activation='relu', name="filter1"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(24, activation='sigmoid', name="filter2"))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(num_classes, activation="sigmoid", name="last"))
rms = optimizers.RMSprop()
metrics = [tf.keras.metrics.CategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.CategoricalCrossentropy()
model.compile(
    loss=loss,
    optimizer= rms,
    metrics=metrics
)
#####.....training the model ... ######
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
converter.experimental_new_converter = True
tflite_model = converter.convert()
open("fights.tflite", "wb").write(tflite_model)