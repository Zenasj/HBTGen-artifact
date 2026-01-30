import numpy as np
import tensorflow as tf
from tensorflow import keras

def mega(A, B, data, batch_size):

    inp = tf.keras.Input(shape = data.shape[1:],batch_size=batch_size)

    
    no_batches = int(inp.shape[0]/4) #model A can take 4 patches at once. 
    hyp_recon = []

    for i in range(no_batches):
        hyp_recon.append(A(inp[i*4:(i+1)*4])) #feed in 4 patches at once
    
    all_recon = tf.stack(hyp_recon) #stack the response

    full_images = batch_combine(all_recon) #this function combined the patches into full images
    
    final = B(full_images) #apply model B to full images

    model = tf.keras.Model(inp,final)
    model._name = 'mega'

    return model