import numpy as np

def define_models(n_input, n_output, n_units):
	# define training encoder
	encoder_inputs = Input(shape=(None, n_input))
	encoder = LSTM(n_units, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	encoder_states = [state_h, state_c]
	# define training decoder
	decoder_inputs = Input(shape=(None, n_output))
	decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = Dense(n_output, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	# define inference encoder
	encoder_model = Model(encoder_inputs, encoder_states)
	# define inference decoder
	decoder_state_input_h = Input(shape=(n_units,))
	decoder_state_input_c = Input(shape=(n_units,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
	# return all models
	return model, encoder_model, decoder_model

with open("training_data_input.txt") as fopen:
  with open("training_data_output.txt") as fopen2:
    for line in fopen:
      myList = line.strip().split()
      myList[0] = myList[0].replace("[","")
      if myList[0] == "":
        myList = myList[1:]
      if "][" in myList[3]:
        j = 0
        print(myList[3])
        myList[3] = myList[3].replace(']][[',"")
        if len(myList[3]) > 3:
          myList[3] = (myList[3][:3])
        myList = myList[:4]
      myList[len(myList)-1] = myList[len(myList)-1].replace("]","")
      x = np.empty((154,45,4),dtype=np.float32)
      i = 0
      j = 0
      if j >=45:
        j = 0
      print(myList)
      x[i][j] = myList
      i+=1
      j+=1
    for line in fopen2:
      myList = line.strip().split()
      x_out = np.empty((154,45,1), dtype=np.float32)
      myList[0] = myList[0].replace("[","")
      if myList[0] == "":
        myList = myList[1:]
      if "][" in myList[0]:
        j = 0
        myList[0] = myList[0].replace(']][[',"")
        if len(myList[0]) > 3:
          myList[0] = (myList[0][:2])
        myList = myList[:1]
      myList[len(myList)-1] = myList[len(myList)-1].replace("]","")
      i = 0
      j = 0
      if j >=45:
        j = 0
      x_out[i][j] = myList
      i+=1
      
print(x.shape)
print(x_out.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, x_out, test_size = 0.2, random_state = 4)
print(x_train.shape)
print(y_train.shape)

model.fit(x_train, y_train, epochs = 50)