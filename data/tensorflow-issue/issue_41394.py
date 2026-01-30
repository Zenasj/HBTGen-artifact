import numpy as np
import tensorflow as tf

class Tem_Agg:

	"""
	This class is to aggregate the representations in the temporal dimension using
	an autoencoder architecture
	"""

	## the length of the time stamp
	max_length = 300

	def __init__(self,  length, timestamp, model = "baseline"):


		## the number of timestamp in the batch
		self.timestamp = timestamp
		## the dimension of the vectors to be aggregated
		self.vec_length = length
		## the mode to use: baseline model or dynamic model
		assert model in ["baseline", "dynamic"]
		self.model = model


	def build_model(self):

		"""
		This is an autoecoder model where the input is used to reconstruct the input itself
		and predict the future

		for reference: https://arxiv.org/pdf/1502.04681.pdf
		"""

		## the input layer

		inp = Input(shape = (self.timestamp - 5, self.vec_length))

		if self.model == "dynamic":

			weights = Input(shape = (2, ))

			## mean
			mean_weights = Dense(inp.shape[1], activation = "relu")(weights)
			mean_weights = tf.expand_dims(mean_weights, axis = -1)
			## standard deviation
			std_weights = Dense(inp.shape[1], activation = "relu")(weights)
			std_weights = tf.expand_dims(std_weights, axis = -1)

		## encoder

		enc_first = LSTM(self.vec_length // 2, return_sequences = True)(inp)

		if self.model == "dynamic":
			enc_first = Tem_Agg.FiLM(enc_first, mean_weights, std_weights)


		enc_second = LSTM(self.vec_length // 4, return_sequences = True)(enc_first)


		if self.model == "dynamic":
			enc_second = Tem_Agg.FiLM(enc_second, mean_weights, std_weights)

		## the full representation has the same dimension as the input so that
		## the hierarchy model can use it in different levels;
		enc_second_full = LSTM(self.vec_length, name = "representation")(enc_first)


		## decoder for reconstruction

		dec_first = LSTM(self.vec_length // 2, return_sequences = True)(enc_second)
		recon = LSTM(inp.shape[-1], return_sequences = True)(dec_first)

		## predict the future vectors
		predicted_vec_1 = Dense(inp.shape[-1], activation = "relu")(enc_second_full)
		predicted_vec_2 = Dense(inp.shape[-1], activation = "relu")(predicted_vec_1)
		predicted_vec_3 = Dense(inp.shape[-1], activation = "relu")(predicted_vec_2)
		predicted_vec_4 = Dense(inp.shape[-1], activation = "relu")(predicted_vec_3)
		predicted_vec_5 = Dense(inp.shape[-1], activation = "relu")(predicted_vec_4)


		if self.model == "baseline":
			model = keras.Model(inputs = inp, outputs = [recon, predicted_vec_5])
			model.compile("adam", loss = MeanAbsolutePercentageError())
		else:
		
			model = Dynamic_Loss(inputs = [inp, weights], outputs = [recon, predicted_vec_5, weights])
			

			model.compile("adam")

		return model


	@staticmethod
	def FiLM(tensor, mean_weights, std_weights):
		"""
		This is the Feature-wise Linear Modulation (FiLM) operation from the paper

		Inputs:
			tensor: the tensor from the layer
			mean_weight: weights to be multiplied to the tensor
			std_weights: weights to be added to product of tensor and mean_weight

		Output:
			output: the output tensor after the operation

		"""

		tensor = Multiply()([tensor, mean_weights])
		output = Add()([tensor, std_weights])
		return output


class Dynamic_Loss(keras.Model):

	"""
	This is the class to learn the distribution of the loss for the multi-tasks
	model. We will pass the weight vector in the training and inference stages

	for reference: https://openreview.net/pdf?id=HyxY6JHKwr

	"""

	def train_step(self, data):

		"""
		Overwrite the training loop

		Input:
			data: the data we feed into the fit() function; a tf Dataset object
		"""

		## unpacking the data
		x, y = data

		## unpacking y
		recon, future = y

		## loss
		mape = MeanAbsolutePercentageError()

		trainable_vars = self.trainable_variables


		with tf.GradientTape(persistent = True) as tape_1:

			## the predicted values
			y_pred = self(x, training = True)
			

			recon_pred, future_pred, weights = y_pred
		

			## the two different losses
			recon_loss = mape(recon, recon_pred)
			#recon_gradients = tape_1.gradient(recon_loss, trainable_vars)



			future_loss = mape(future, future_pred)
			

		
		recon_gradients = tape_1.gradient(recon_loss, trainable_vars)
		
		future_gradients = tape_1.gradient(future_loss, trainable_vars)




		## the gradient of the sum is the sum of the gradient
		gradients = weights.numpy()[0] * recon_gradients + weights.numpy()[1] * future_gradients

		## applying gradients
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))