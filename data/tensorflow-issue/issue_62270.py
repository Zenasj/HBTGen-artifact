embedding_dim = 128  
vocab_size = len(words_to_int)  # number of unique words

num_input = keras.Input(shape= (30,3), name="nums")
nmlstm = layers.LSTM(64)(num_input) #num input goes to num lstm
nmdense = layers.Dense(64)(nmlstm) #num lstm goes to num dense

text_input = keras.Input(shape= (30, max_sequence_length), name="text") #basically length of the longest string observation sequence 

text_vec = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(text_input) #text input goes to text embedding
txlstm = layers.LSTM(64)(text_vec) #text embedding goes to text lstm


united = layers.concatenate([nmdense, txlstm]) #concatenating text lstm and num dense
almostlast = layers.Dense(64)(united) #computing united input 
last = layers.Dense(2, name='prediction')(almostlast) #output dense layer

model = keras.Model(inputs=[num_input, text_input], outputs=last)
model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy)
model.fit(
    {"nums": X1_train, "text": X2_train},
    {"prediction": y_train},
    epochs=8,
    batch_size=10)

### Relevant log output