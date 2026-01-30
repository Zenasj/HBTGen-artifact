model = Sequential([
  LSTM(32, return_sequences=True, input_shape=(window_size, 1)),
  Dropout(0.2),
  LSTM(32, return_sequences=True),
  Dropout(0.2),
  LSTM(16, return_sequences=False),
  Dense(n_steps_out),
])
model.save('testttt-2')

from tensorflow import lite

converter = lite.TFLiteConverter.from_saved_model('testttt-2')
converter.optimizations = [lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS, lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

with open('testttt-2.tflite', 'wb') as f:
  f.write(tflite_model)