model = Sequential()
model.add(Input([256], dtype="int32"))
model.add(Embedding(35000, 10))
model.add(GRU(10,))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

# Train
model.fit(train_input, labels, epochs=10, batch_size=128,)

# Convert
import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_enable_resource_variables = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter._experimental_lower_tensor_list_ops = False
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Load interpreter
interpreter = tf.lite.Interpreter(model_path="./gru_0722_tflite/model_2.tflite")
interpreter.allocate_tensors()

# inference
interpreter.set_tensor(input_details[0]['index'], encode_plus_inputs["input_ids"])
interpreter.invoke()

# get result <---- Error occured 
output_data = interpreter.get_tensor(output_details[0]['index'])
output_data