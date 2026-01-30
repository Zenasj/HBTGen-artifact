import numpy as np
import random
import tensorflow as tf

#Converting a simple model.
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(10,6)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=2)

converter = lite.TFLiteConverter.from_keras_model(model)

#The problem is caused by the following line
converter.optimizations = [lite.Optimize.DEFAULT]

tfmodel = converter.convert()
open(PATH+'/model.tflite',"wb").write(tfmodel)

interpreter = tf.lite.Interpreter(model_path=PATH+"/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()