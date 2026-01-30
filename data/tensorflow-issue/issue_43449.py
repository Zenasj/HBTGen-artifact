from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class RNN_Model(tf.keras.Model):
    def __init__(self):
        super(RNN_Model, self).__init__()
        self.embed=tf.keras.layers.Embedding(5,1)
        self.d2 = tf.keras.layers.Dense(2)
        self(tf.constant([4,3,2])) # initialize
    
    @tf.function
    def call(self, x):
        x = self.embed(x)
        return self.d2(x)
     
model=RNN_Model()

v=[tf.ones(w.shape) for w in model.trainable_variables]
with tf.autodiff.ForwardAccumulator(primals = model.trainable_variables, tangents = v) as acc:
    loss = tf.reduce_sum(tf.constant([1,0])-model(tf.constant([[2,2,2], [1,1,1]]), training=True))
acc.jvp(loss)

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding,Bidirectional,LSTM

class RNN_Model(tf.keras.Model):
    def __init__(self, dataset):
        super(RNN_Model, self).__init__()
        self.embed=Embedding(maxfeature,64)
        self.blstm = Bidirectional(LSTM(64))
        self.d1 = Dense(64, activation='relu')
        self.d2 = Dense(2)
                
    # Doing this to initialize model.trainable_variables
        for text, labels in dataset:
            self(text)
            break

    def call(self, x):
        x = self.embed(x)
        x = self.blstm(x)
        x = self.d1(x)
        x = self.d2(x)
        return tf.nn.log_softmax(x)
        
## Training Settings
batch_size = 64
maxfeature=100
SEQ_LEN=32

## Load Problems
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=maxfeature)
x_train=tf.keras.preprocessing.sequence.pad_sequences(x_train,maxlen=SEQ_LEN)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(5000).batch(batch_size)

model=RNN_Model(train_ds)

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

# Compute JVP
card = train_ds.cardinality()
step=0
for images, labels in train_ds:
    step+=1
    print("\r%d/%d" % (step , card), end="")
    v=[tf.ones(w.shape) for w in model.trainable_variables]
    with tf.autodiff.ForwardAccumulator(primals = model.trainable_variables, tangents = v) as acc:
        loss = loss_obj(labels, model(images, training=True))
    acc.jvp(loss)