import numpy as np

no_of_files = len(x_tr_f)
file_batch = 5
loop_iterations = np.floor(no_of_files / file_batch)
init_alpha = 47.05906039370126
mask_value = -1.3371337

def tf_data_generator5(directory = [], batch_size = 5):
    i = 0
    x_t = os.listdir(directory[0])
    y_t = os.listdir(directory[1])
    x_t, y_t = shuffle(x_t, y_t)
    while True:
        if i*batch_size >= len(x_t):
            i = 0
            x_t, y_t = shuffle(x_t, y_t)
        file_chunk = x_t[i*batch_size:(i+1)*batch_size] 
        X_a = []
        Y_a = []
        for fname in file_chunk:
            x_info = np.load(directory[0]+fname)
            y_info = np.load(directory[1]+fname)
            X_a.append(x_info)
            Y_a.append(y_info)
        X_a = np.concatenate(X_a)
        Y_a = np.concatenate(Y_a)
       #just some masking stuff and weight generation, irrelevant to the problem
        tte_mean_train = np.nanmean(Y_a[:,:,0])
        mask_value = -1.3371337
        X_a,Y_a, W_a = nanmask_to_keras_mask(X_a,Y_a,mask_value,tte_mean_train)
        yield X_a,Y_a
        i = i + 1

generated_train_data = tf_data_generator2(['./data/x_train_m/', './data/y_train_m/'], batch_size = 5)

def base_model():
    model = Sequential()
    model.add(Masking(mask_value=mask_value,input_shape=(None, 2)))
    model.add(GRU(3,activation='tanh',return_sequences=True))
    return model
def wtte_rnn():
    model = base_model()
    model.add(TimeDistributed(Dense(2)))
    model.add(Lambda(wtte.output_lambda, 
                     arguments={"init_alpha":init_alpha, 
                                "max_beta_value":4.0,
                                "alpha_kernel_scalefactor":0.5}))

    loss = wtte.loss(kind='discrete',reduce_loss=False).loss_function
    model.compile(loss=loss, optimizer=Adam(lr=.01,clipvalue=0.5),sample_weight_mode='temporal')
    return model

model = wtte_rnn()
model.summary()

K.set_value(model.optimizer.lr, 0.01)
model.fit(generated_train_data,
          epochs=100,
          steps_per_epoch=loop_iterations)