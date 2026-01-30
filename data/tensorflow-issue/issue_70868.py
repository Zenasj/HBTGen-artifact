import tensorflow as tf

def create_state_model(input_dim):
        model = Sequential()
        model.add(Reshape((input_dim, 1), input_shape=(input_dim,)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

def create_temp_model(input_dim, output_dim):
        model = Sequential()
        model.add(Reshape((input_dim, 1), input_shape=(input_dim,)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

# State Model Conversion
converter = tf.lite.TFLiteConverter.from_keras_model(state_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]
state_model_tflite = converter.convert()
with open('state_model.tflite', 'wb') as f:
    f.write(state_model_tflite)

# Temperature Model Conversion
converter = tf.lite.TFLiteConverter.from_keras_model(temp_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]
temp_model_tflite = converter.convert()
with open('temp_model.tflite', 'wb') as f:
    f.write(temp_model_tflite)