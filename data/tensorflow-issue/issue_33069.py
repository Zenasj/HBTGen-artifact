import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	# Currently, memory growth needs to be the same across GPUs
	try:
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
	except RuntimeError as e:
		print(e)

def gradient_descent(model, inputs, targets):
	with tf.GradientTape() as tape:
		# compute loss value
		y_predict = model(inputs)
		loss_value = tf.keras.losses.categorical_crossentropy(y_true = targets,
		                                                      y_pred = y_predict)
	return loss_value, tape.gradient(loss, model.trainable_variables)

with tf.device('/device:gpu:0'):
    model = tf.keras.Sequential([
		    tf.keras.Input(shape = (sequence_len, num_feature),
		                   name = 'InputLayer'),
		    tf.keras.layers.Masking(mask_value = 0.,
		                            input_shape = (sequence_len, num_feature)),
		    tf.keras.layers.Bidirectional(
			    tf.keras.layers.LSTM(units = 50, return_sequences = True),
			    name = 'BiLSTM-1'),
		    tf.keras.layers.Dense(units = 3, activation = 'softmax',
		                          name = 'Softmax')])

    train_x, train_y = train_dataset.next_batch()
    loss_value, grads = gradient_descent(model, train_x, train_y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))