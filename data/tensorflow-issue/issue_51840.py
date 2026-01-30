import random

class WeightedSum(krs.layers.Layer):
    def __init__( self, n_models = 2, name = 'weighted_sum_0' ):
        super( WeightedSum, self ).__init__( name = name)
        self.n_models = n_models
        self.ensemble_weights = []
        self.output_init = tf.Variable(0.,validate_shape=False,trainable=False)

    def build(self,input_shape):
        for i in range(self.n_models):
            self.ensemble_weights.append( self.add_weight(shape=(1,),
                                    initializer = 'ones',
                                    trainable = True) )

    def call(self,inputs):
        new_normalizer = tf.convert_to_tensor(0.,dtype = inputs[0].dtype)
        for i in range(self.n_models):
            new_normalizer = new_normalizer + self.ensemble_weights[i]
        new_normalizer = tf.constant(1.,dtype=new_normalizer.dtype)/new_normalizer
        output = self.output_init

        for i in range(self.n_models):
            output = tf.add(output,tf.multiply(self.ensemble_weights[i],inputs[i]))
        output = tf.multiply( output, new_normalizer )
        return output

krs.models.save_model(linked_model,"test_failed_save.mdl")

import numpy as np
import tensorflow.keras as krs
import tensorflow as tf




class WeightedSum(krs.layers.Layer):
    def __init__( self, n_models = 2, **kwargs):
        super( WeightedSum, self ).__init__( **kwargs)
        self.n_models = n_models
        self.ensemble_weights = []
        self.output_init = tf.Variable(0.,validate_shape=False,trainable=False)

    def build(self,input_shape):
        for i in range(self.n_models):
            self.ensemble_weights.append( self.add_weight(shape=(1,),
                                    initializer = 'ones',
                                    trainable = True) )

    def call(self,inputs):
        new_normalizer = tf.convert_to_tensor(0.,dtype = inputs[0].dtype)
        for i in range(self.n_models):
            new_normalizer = new_normalizer + self.ensemble_weights[i]
        new_normalizer = tf.constant(1.,dtype=new_normalizer.dtype)/new_normalizer
        output = tf.cast(self.output_init,dtype=inputs[0].dtype)

        for i in range(self.n_models):
            output = tf.add(output,tf.multiply(tf.cast(self.ensemble_weights[i],dtype=inputs[i].dtype),inputs[i]))
        output = tf.multiply( output, new_normalizer )
        return output


input_lf = krs.Input((4,))

x = input_lf
x = krs.layers.Dense(10,activation = 'relu')(x)
x = krs.layers.Dense(10,activation = 'relu')(x)
lf_out = krs.layers.Dense(10,activation = 'relu')(x)

lf_mod = krs.Model(input_lf,lf_out,name='lf')

input_hf_lin = krs.Input((14,))
x = input_hf_lin
x = krs.layers.Dense(10)(x)
x = krs.layers.Dense(10)(x)
hf_lin_out = krs.layers.Dense(10,activation = 'relu')(x)

hf_lin_mod = krs.Model(input_hf_lin,hf_lin_out,name='hf_linear')

input_hf_nonlin = krs.Input((14,))

x = input_hf_nonlin
x = krs.layers.Dense(10,activation = 'relu')(x)
x = krs.layers.Dense(10,activation = 'relu')(x)
hf_nonlin_out = krs.layers.Dense(10,activation = 'relu')(x)

hf_nonlin_mod = krs.Model(input_hf_lin,hf_lin_out,name='hf_nonlinear')

input_hf = krs.Input((14,))

x = input_hf
lin = hf_lin_mod(x)
nonlin = hf_nonlin_mod(x)
summed_out = WeightedSum(n_models=2)([lin,nonlin])

hf_mod = krs.Model(input_hf,summed_out,name='hf')

input_full_mod = krs.Input((4,))
x = input_full_mod

low = lf_mod(x)
x = krs.layers.Concatenate()([low,x])
full_out = hf_mod(x)

full_mod = krs.Model(input_full_mod,outputs = {'low_fidelity':low,'high_fidelity':full_out},name='full_model')

opt = krs.optimizers.Adam()
loss = krs.losses.MSE
full_mod.compile(optimizer = opt,loss = loss)

x_train = np.random.uniform(0,10,(20,4))
y_train_low = np.random.uniform(0,10,(20,10))
y_train_high = np.random.uniform(0,10,(20,10))
y = {"low_fidelity": y_train_low,
     "high_fidelity": y_train_high}



full_mod.fit(x_train,y,epochs=5)

krs.models.save_model(full_mod,"test_model.mdl")