import random
import tensorflow as tf

tf.gradients(ct, c0)

### 1. Define model
batch_size = 100
input_length_m= 1403
output_dim= 100

xtr_pad = tf.random.uniform((batch_size, input_length_m), maxval = 500, dtype=tf.int32)
ytr = tf.random.normal((batch_size, input_length_m, 200))
inp= Input(batch_shape= (batch_size, input_length_m), name= 'input') 
emb_out= Embedding(500, output_dim, input_length= input_length_m, trainable= False, name= 'embedding')(inp)

class LSTMCellwithStates(LSTMCell):
    def call(self, inputs, states, training=None):
        real_inputs = inputs[:,:self.units] # decouple [h, c]
        outputs, [h,c] = super().call(real_inputs, states, training=training)
        return tf.concat([h, c], axis=1), [h,c]
    
rnn = RNN(LSTMCellwithStates(200), return_sequences= True, return_state= False, name= 'LSTM') 
h0 = tf.Variable(tf.random.uniform((batch_size, 200)))
c0 = tf.Variable(tf.random.uniform((batch_size, 200)))
rnn_allstates= rnn(emb_out, initial_state=[h0, c0])  
model_lstm_mod = Model(inputs=inp, outputs= rnn_allstates, name= 'model_LSTMCell')
model_lstm_mod.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

### 2. Compute gradients:
ds = tf.data.Dataset.from_tensor_slices((xtr_pad, ytr)).batch(100)

@tf.function
def compute_dct_dc0(ct, c0):
    return tf.gradients(ct, c0)

n_b = int(xtr_pad.shape[0]/ 100)
n_steps = 20   # look up only the first and last 20 steps

dctdc0_all= tf.zeros([n_b, n_steps])
for b, (x_batch_train, y_batch_train) in enumerate(ds):  
    grad_batch= []   
    cell_states= model_lstm_mod(x_batch_train)[:, :, 200:]
    for t in range(n_steps):  
        ct= cell_states[:, t, :]
        print(ct)
        # steps 0,...,19
        dctdc0_b_t = compute_dct_dc0(ct, c0)  # (batch_size, n_units)
        print(dctdc0_b_t)
        grad_t = tf.reduce_mean(abs(dctdc0_b_t[0]), [0,1]) # Scalar dctdc0 at the current batch and step
        print('step', t+1, 'of batch' ,b+1, 'done')
        grad_batch.append(grad_t)
    
    dctdc0_all= tf.concat([dctdc0_all, [grad_batch]], axis = 0)

cell_states= model_lstm_mod(x_batch_train)[:, :, 200:]

tape.watch(h0)

tf.gradients