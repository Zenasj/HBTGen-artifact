import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
#converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
#converter.allow_custom_ops=True
tflite_model = converter.convert()

open("tf_test.tflite", "wb").write(tflite_model)

new_model = Sequential()
new_model.add(Input(name='input', batch_size=1, shape=(925, 3)) )
new_model.add(Dense(8))
new_model.add(LSTM(64, return_sequences=True))
new_model.add(LSTM(64, return_sequences=True))
new_model.add(LSTM(64, return_sequences=True))
new_model.add(LSTM(64, return_sequences=True))
new_model.add(Activation('softmax', name='softmax'))

new_model.summary()

def equal(tensor1, tensor2):
    for i, j in zip(tensor1.reshape(-1), tensor2.reshape(-1)):
        if abs(i - j) > 0.001:
            return False
    return True

interpreter.reset_all_variables()

def equal(tensor1, tensor2):
    for i, j in zip(tensor1.reshape(-1), tensor2.reshape(-1)):
        if abs(i - j) > 0.001:
            return False
        else: 
            return True