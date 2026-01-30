import tensorflow as tf

model = keras.Sequential()
model.add(layers.Embedding(input_dim=1000, output_dim=64))
model.add(layers.LSTM(128))
model.add(layers.Dense(10))
model.summary()

model.save(f'output_models/simple_lstm_saved_model_format_{tf.__version__}', save_format='tf')
model.save(f'output_models/simple_lstm_{tf.__version__}.h5', save_format='h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(f"output_models/simple_lstm_tf_v{tf.__version__}.tflite", 'wb') as f:
    f.write(tflite_model)

converter_saved_model = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
tflite_model_from_saved_model = converter_saved_model.convert()
with open(f"{saved_model_path}_converted_tf_v{tf.__version__}.tflite", 'wb') as f:
    f.write(tflite_model_from_saved_model)