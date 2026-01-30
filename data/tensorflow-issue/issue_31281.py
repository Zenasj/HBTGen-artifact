import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

for learningrate in learningrates:
	for layerdensity in layerdensitys:
		for layer in amount_of_layers:
			################################
			# generate model               #
			################################
			modelname = f"{layer}-layer_{layerdensity}-nodes_selu-adam_{learningrate}-learningrate_{records_per_epoch}-epochsize_{appendix}"
			model = keras.Sequential()
			## layertypes ##
			# dense // general
			# convolutional; max pooling // image classification
			# CuDNNLSTM // long short term memory (cudnn gpu optimized otherwise just lstm) (same as dense just cudnn)
			model.add(Dense(layerdensity, activation=tf.nn.selu, input_dim=15))
			for i in range(layer-1):
				model.add(Dense(layerdensity, activation=tf.nn.selu))
			model.add(Dense(9,activation=tf.nn.softmax, name = "Output"))
			# Compile
			optimizer = tf.keras.optimizers.Adam(lr=learningrate)
			model.compile(
				optimizer=optimizer,
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy'])
			model.summary()
			tensorboard = TensorBoard(log_dir="\\\\drg-fs01\\BigData\\Projects\\Notebooks\\PokerBot\\log\\" + modelname,
				histogram_freq = 100, write_graph = False)
            #cp_callback = tf.keras.callbacks.ModelCheckpoint("\\\\drg-fs01\\BigData\\Projects\\Notebooks\\PokerBot\\checkpoints\\" + modelname, verbose=0)
            ################################
            # train model                  #
            ################################
			model.fit(trainSet, 
				epochs = epochs, 
				steps_per_epoch = trainSteps, 
				shuffle = True, 
				validation_data = testSet, 
				validation_steps = testSteps, 
				validation_freq = int(epochs/maxTestEpochs),
				verbose = verbose, 
				callbacks = [tensorboard])#,cp_callback])
			model.save(basePath+'saved_models/' + modelname + '.h5')