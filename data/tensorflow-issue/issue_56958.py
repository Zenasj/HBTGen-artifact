from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.optimizers as O
import tensorflow.keras.losses as Los

from sklearn.model_selection import KFold
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

positive_excerpts = train_data[train_data.target.values>=0]
negative_excerpts = train_data[train_data.target.values<0]

positive_len = [len(x) for x in positive_excerpts.excerpt.values]
negative_len = [len(x) for x in negative_excerpts.excerpt.values]

tokenizer = Tokenizer(num_words=500)
tokenizer.fit_on_texts(train_data.excerpt.values)
data_seq = tokenizer.texts_to_sequences(train_data.excerpt.values)

BATCH_SIZE = 16
MAX_LEN = 172
EPOCHS = 10
pad_data_seq = tf.keras.preprocessing.sequence.pad_sequences(data_seq,maxlen=MAX_LEN,padding='post')
def build_model():
    inp = L.Input(shape=(MAX_LEN,))
    emb = L.Embedding(input_dim=500,output_dim = 62)(inp)
    X = L.Bidirectional(L.LSTM(32))(emb)
    X = L.Dense(64,activation='relu')(X)
    X = L.Dense(32,activation='relu')(X)
    out = L.Dense(1)(X)
    
    model = M.Model(inputs=inp,outputs=out)
    model.compile(loss='mse',optimizer='adam',metrics=['acc'])
    return model
model = build_model()
model.summary()
kf = KFold(n_splits=5,random_state=24,shuffle=True)

for index,(t_idx,v_idx) in enumerate(kf.split(pad_data_seq)):
    print(f"######## STEP {index+1} ########")
    train_data_seq = pad_data_seq[t_idx]
    val_data_seq = pad_data_seq[v_idx]
    train_target = train_data.target.values[t_idx]
    val_target = train_data.target.values[v_idx]
    
    history = model.fit(train_data_seq,
                        train_target,
                        validation_data=(val_data_seq,val_target),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE)
test_data_seq = tokenizer.texts_to_sequences(test_data.excerpt.values)
test_data_seq = tf.keras.preprocessing.sequence.pad_sequences(test_data_seq,maxlen=MAX_LEN)
pred = model.predict(pad_data_seq)
pred = model.predict(test_data_seq,verbose=1)
sampl = pd.read_csv('sample_submission.csv')
sampl.target = pred
sampl.to_csv('submission.csv',index=False)

from sklearn.metrics import mean_squared_error
import pandas as pd

y_pred = pd.read_csv('submission.csv')[['target']]
y_target = pd.read_csv('target.csv')[['target']]
print("MSE: ", mean_squared_error(y_target, y_pred))
print("RMSE: ",mean_squared_error(y_target,y_pred,squared=False))