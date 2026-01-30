import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

'''---- Training -----'''
model=tf.keras.Sequential() # model has 1 hidden layer
model.add(tf.keras.layers.Dense(20,input_shape=(dimVectors,),activation='relu'))
model.add(tf.keras.layers.Dense(5,activation='softmax')) # 5 is ouput shape
optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam'
)
epochs=10
# without padding
wordVectors=tf.Variable(tf.convert_to_tensor(wordVectors))
trainLabels=tf.keras.utils.to_categorical(trainLabels,5)
for epoch in range(epochs):
    sent_count=0
    print("---Epoch no:",epoch,"----")
    #one sentence in each batch
    for sent in train_Sentences:
        if((sent_count%10)==0):
            print("Optimizing sentence number",sent_count+1)
        sent_count+=1
        len_sent=len(sent)
        one_hot=np.zeros((len_sent,total_words))
        label=(trainLabels[sent_count-1]).reshape(-1,1)
        label=tf.convert_to_tensor(label)
        for i,word in enumerate(sent):
            index=token_dict[word]
            one_hot[i,index]=1
        one_hot=tf.convert_to_tensor(one_hot)
        with tf.GradientTape() as tape:
            #tape.watch(wordVectors)
            feature=tf.matmul(one_hot,wordVectors)
            feature_sum=tf.math.reduce_sum(feature,axis=0,keepdims=True)
            y_pred=model(feature_sum)
            loss=tf.losses.MeanSquaredError()(y_pred,label)
        gradients=tape.gradient(loss,[wordVectors]+model.trainable_variables)
        if(((sent_count-1)%10)==0):
                print("Loss: ",loss)
        '''wordVectors.assign(wordVectors-learning_rate*gradients[0])
        (model.trainable_variables[0]).assign(model.trainable_variables[0]-learning_rate*gradients[1])
        (model.trainable_variables[1]).assign(model.trainable_variables[1]-learning_rate*gradients[2])
        (model.trainable_variables[2]).assign(model.trainable_variables[2]-learning_rate*gradients[3])
        (model.trainable_variables[3]).assign(model.trainable_variables[3]-learning_rate*gradients[4])'''
        optimizer.apply_gradients(zip(gradients,[wordVectors]+model.trainable_variables))