from tensorflow.keras import models
from tensorflow.keras import optimizers

for learningrate in learningrates:
	for layerdensity in layerdensitys:
		for layer in amount_of_layers:
			################################
			# generate model               #
			################################
			modelname = f"{layer}-layer_{layerdensity}-nodes_selu-adam_{learningrate}-learningrate_{records_per_epoch}-epochsize_{appendix}"
			model = keras.Sequential()
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

# dataset modeler
def modelDataset(sourcepath, badgesize, repeat = False, repetitions = 10):
    #get all files
    list = os.listdir(sourcepath)
    pathfiles = [sourcepath+x for x in list]
    
    #get metrics
    rows_per_file = count_lines(sourcepath+"0.csv")
    number_of_files = len(list)
    total_rows = (rows_per_file * number_of_files)
    print(f"records: {total_rows}")
    # get number of steps per Epoch
    steps_per_epoch = int(rows_per_file / badgesize) # 2000 badges per epoch
    epochs = number_of_files
    if badgesize == 1:
        epochs = 1 
    print(f"number epochs: {epochs}")
    # model interleaved dataset
    dataset = (tf.data.Dataset.from_tensor_slices(pathfiles).interleave(lambda x:
        tf.data.TextLineDataset(x).map(parse_csv, num_parallel_calls=4),
        cycle_length=4, block_length=16))
    dataset.columns = CSV_COLUMNS
    
    if badgesize != 1:
        dataset = dataset.shuffle(buffer_size=badgesize)
    if repeat:
        dataset = dataset.repeat(repetitions)
        epochs = epochs * repetitions
    dataset = dataset.batch(badgesize)
    dataset = dataset.prefetch(2)  # prefetch one batch
    return dataset, steps_per_epoch, epochs, badgesize

from __future__ import absolute_import, division, print_function, unicode_literals

from sklearn.model_selection import train_test_split

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import glorot_uniform
from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
import tensorflow_datasets as tfds

# Helper libraries
import numpy as np
import time
import os
from tqdm import tqdm #progressbars

print(tf.__version__)

# count lines
def count_lines(path):
    rownumber = 0
    with open(path, "r",encoding="utf-8",errors='ignore') as f:
        rownumber =  sum(bl.count("\n") for bl in blocks(f))
    return rownumber

def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b

basePath = "//[fileshare]/BigData/Projects/Notebooks/Trainmodel/"
localPath = "C:/temp/Trainmodel/"
                                                                                  
valPath = localPath+"val/"
trainPath = localPath+"train/"
testPath = localPath+"test/"

# csv to tensor parser
def parse_csv(line):
        parsed_line = tf.io.decode_csv(line, [[0.], [0.], [0.],[0.], [0.], [0.], [0.],[0.], [0.], [0.], [0.],[0.], [0.], [0.], [0.], [0]])
        label = parsed_line[-1:]
        del parsed_line[-1]
        features = parsed_line
        return tf.stack(features),label

# dataset modeler
def modelDataset(sourcepath, badgesize, repeat = False, repetitions = 10):
    #get all files
    list = os.listdir(sourcepath)
    pathfiles = [sourcepath+x for x in list]
    
    #get metrics
    rows_per_file = count_lines(sourcepath+"0.csv")
    number_of_files = len(list)
    total_rows = (rows_per_file * number_of_files)
    print(f"records: {total_rows}")
    # get number of steps per Epoch
    steps_per_epoch = int(rows_per_file / badgesize) # 2000 badges per epoch
    epochs = number_of_files
    if badgesize == 1:
        epochs = 1 
    print(f"number epochs: {epochs}")
    # model interleaved dataset
    dataset = (tf.data.Dataset.from_tensor_slices(pathfiles).interleave(lambda x:
        tf.data.TextLineDataset(x).map(parse_csv, num_parallel_calls=4),
        cycle_length=4, block_length=16))
    #dataset.columns = CSV_COLUMNS
    
    if badgesize != 1:
        dataset = dataset.shuffle(buffer_size=badgesize)
    if repeat:
        dataset = dataset.repeat(repetitions)
        epochs = epochs * repetitions
    dataset = dataset.batch(badgesize)
    dataset = dataset.prefetch(2)  # prefetch one batch
    return dataset, steps_per_epoch, epochs, badgesize

# load interleaved dataset
trainSet, trainSteps, maxTrainEpochs, trainBadgeSize = modelDataset(trainPath, 32, True)
print (f"max number of epochs: {maxTrainEpochs}")
testSet, testSteps, maxTestEpochs, testBadgeSize = modelDataset(testPath, 32,True,4)
valSet, valSteps, maxValEpochs, valBadgeSize = modelDataset(valPath, 1)

# compile model
learningrates = [0.0002]
layerdensitys = [40]
amount_of_layers = [5]
appendix = "GPUtest"
epochs = maxTrainEpochs
records_per_epoch = trainSteps * trainBadgeSize
verbose = 0

################################
# train model                  #
################################
modelname = "unspecified"
for learningrate in learningrates:
	for layerdensity in layerdensitys:
		for layer in amount_of_layers:
			################################
			# generate model               #
			################################
			modelname = f"{layer}-layer_{layerdensity}-nodes_selu-adam_{learningrate}-learningrate_{records_per_epoch}-epochsize_{appendix}"
			model = keras.Sequential()
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
			tensorboard = TensorBoard(log_dir="\\\\[fileshare]\\BigData\\Projects\\Notebooks\\Trainmodel\\log\\" + modelname,
				histogram_freq = 100, write_graph = False)
            #cp_callback = tf.keras.callbacks.ModelCheckpoint("\\\\[Fileshare]\\BigData\\Projects\\Notebooks\\Trainmodel\\checkpoints\\" + modelname, verbose=0)
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

0.25,0.846153846153846,0.5,0.615384615384615,0.75,0.461538461538462,0.25,0.461538461538462,0.75,0.769230769230769,0.75,0.384615384615385,0.25,0.538461538461538,1,8
1,0.923076923076923,0.5,0.0769230769230769,1,0.538461538461538,0.25,0.384615384615385,1,0.846153846153846,0.75,0.384615384615385,1,0.692307692307692,0.444444444444444,2
0.75,0.538461538461538,0.25,0.846153846153846,0.75,0.461538461538462,1,0.769230769230769,0.75,1,1,0.461538461538462,0.75,0.307692307692308,0.555555555555556,3
0.5,0.307692307692308,0.75,0.615384615384615,1,0.0769230769230769,0.75,0.0769230769230769,0.5,0.538461538461538,0.75,0.230769230769231,0.25,0.461538461538462,0.888888888888889,4
0.5,0.307692307692308,0.25,0.769230769230769,1,1,1,0.384615384615385,0.25,1,0.75,0.230769230769231,0.75,0.307692307692308,0.444444444444444,1
1,0.923076923076923,0.5,0.846153846153846,0.75,0.153846153846154,0.5,0.769230769230769,0.25,0.230769230769231,1,0.846153846153846,1,0.307692307692308,0.777777777777778,5
0.5,0.769230769230769,1,0.846153846153846,0.25,0.0769230769230769,0.25,1,0.25,0.692307692307692,0.5,0.923076923076923,0.5,0.230769230769231,0.555555555555556,0
0.5,0.230769230769231,0.25,0.923076923076923,0.5,0.846153846153846,0.5,0.384615384615385,1,0.615384615384615,1,0.384615384615385,0.5,0.153846153846154,1,5
0.25,0.230769230769231,0.5,0.769230769230769,0.75,0.692307692307692,1,0.769230769230769,0.25,0.153846153846154,0.5,1,0.25,0.538461538461538,0.666666666666667,2
0.5,0.615384615384615,1,0.923076923076923,0.75,0.307692307692308,1,0.307692307692308,0.5,0.692307692307692,0.25,0.615384615384615,0.5,0.461538461538462,1,3

# csv to tensor parser
def parse_csv(line):
        parsed_line = tf.io.decode_csv(line, [[0.], [0.], [0.],[0.], [0.], [0.], [0.],[0.], [0.], [0.], [0.],[0.], [0.], [0.], [0.], [0]])
        label = parsed_line[-1:]
        del parsed_line[-1]
        features = parsed_line
        return tf.stack(features),label