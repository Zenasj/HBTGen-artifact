import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Rescale(tf.keras.layers.Layer):
    def __init__(self,scale_limit=np.pi,**kwargs):
        super().__init__(trainable=False,**kwargs)
        self.scale_limit = scale_limit
    
    def call(self,inputs):
        s = tf.reduce_sum(inputs,axis=-1)
        return (inputs/s)*self.scale_limit

class DenseQKan(tf.keras.layers.Layer):
    def __init__(self,units:int,circuit:qml.QNode,layers:int,**kwargs):
        super().__init__(**kwargs)
        self.circuit = circuit
        self.qubits =  len(circuit.device.wires)
        self.units = units
        self.qbatches = None
        self.layers = layers
        
    def build(self,input_shape):
        if input_shape[-1]> self.qubits:
            self.qbatches = np.ceil(input_shape[-1]/self.qubits).astype(np.int32)
        else:
            self.qbatches = 1
        self.layer_weights = []
        for u in range(self.units):
            self.layer_weights.append(self.add_weight(shape=(self.qbatches,input_shape[-1]//self.qbatches,self.layers),
                                   initializer=tf.keras.initializers.RandomUniform(minval=-np.pi, maxval=np.pi, seed=None),
                                   trainable=True))
        self.built = True
        # W = np.random.uniform(low=-np.pi,high=np.pi,size=(self.units,self.qbatches,self.qubits,self.layers))
    # @tf.function(reduce_retracing=True)

    def compute_output_shape(self,input_shape):
        print("Build Input Shape",input_shape)
        return (input_shape[0],self.units)
        
    def call(self,inputs):
        assert self.qbatches != None 
        splits = tf.split(inputs,self.qbatches,-1) 
        out = []
        for u in range(self.units):
            unit_out = 0
            for qb in range(self.qbatches):
                qb_out = tf.reduce_sum(tf.stack(self.circuit(splits[qb],self.layer_weights[u][qb]),axis=-1),axis=-1)
                unit_out = unit_out+qb_out
            out.append(unit_out)
        out = tf.stack(out,axis=-1)
        return out

# def create_model(units,qubits,layers,circuit,input_shape=2):
inp = Input(shape=input_shape)
out = DenseQKan(units,circuit,layers,name="DenseKAN")(inp)
out = Rescale(name="RescalePi")(out)
model = Model(inputs=inp,outputs=out,name="Q_KAN")
model.summary(show_trainable=True)

out = tf.reshape(out,(tf.shape(inputs)[0],self.units))