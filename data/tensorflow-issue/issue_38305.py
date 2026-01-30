import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

class VocabLookup(tf.keras.layers.Layer):
    def __init__(self,vocab_path):
        super(VocabLookup, self).__init__(trainable=False,dtype=tf.int64)
        self.vocab_path = vocab_path
    def build(self,input_shape):
        table_init = tf.lookup.TextFileInitializer(self.vocab_path,tf.string,tf.lookup.TextFileIndex.WHOLE_LINE,
                              tf.int64,tf.lookup.TextFileIndex.LINE_NUMBER)
        self.table = tf.lookup.StaticHashTable(table_init,-1)
        self.built=True
        
    def call(self, input_text):
        splitted_text = tf.strings.split(input_text).to_tensor()
        word_ids = self.table.lookup(splitted_text)
        return word_ids
    
    def get_config(self):
        config = super(VocabLookup, self).get_config()
        config.update({'vocab_path': self.vocab_path})
        return config 
input_text = tf.keras.Input(shape=(),dtype=tf.string,name='input_text')
lookup_out = VocabLookup(vocab_path=vocab_path)(input_text)
model_lookup = tf.keras.Model(inputs={'input_text':input_text},outputs=lookup_out)

keys_tensor = tf.constant(range(27), tf.int64)
vals_tensor = tf.constant([0]*11+[1]*9+[2]*6+[3], tf.int64)
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)

inputs = layers.Input(shape=(4))
inputs1 = layers.Input(shape=(4))
x = layers.Dense(4)(inputs)
model = keras.Model([inputs,inputs1], x)

key = tf.argmax(inputs1, axis=1)
matches = table.lookup((key))
matches = tf.one_hot(matches, depth=4)
loss = K.categorical_crossentropy(matches, inputs1)

model.add_loss(loss)
model.compile(optimizer=keras.optimizers.Adam(1e-3))

x_train = tf.ones(shape=[2,4])
y_train = tf.constant([[0,0,0,1],[0,0,1,0]])
model.fit([x_train, y_train])