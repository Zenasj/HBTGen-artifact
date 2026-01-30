from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

@tf.function
def calc_sw3(gamma,gamma2, cutoff_jk):
    E3 = 2.0
    return E3

@tf.function
def calc_sw3_noerr( gamma0, cutoff_jk):
    E3 = 2.0
    return E3

@tf.function # without tf.function this works fine
def energy(coords, gamma):
    xyz_i = coords[0, 0 : 3]
    xyz_j = coords[0, 3 : 6]
    rij = xyz_j - xyz_i
    norm_rij = (rij[0]**2 + rij[1]**2 + rij[2]**2)**0.5
    E3 = calc_sw3( gamma,gamma,norm_rij)    # repeating gamma gives error
    # E3 = calc_sw3_noerr( gamma, norm_rij) # this gives no error
    return E3



class SWLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.gamma = tf.Variable(2.51412, dtype=tf.float32)

    def call(self, coords_all):
        total_conf_energy = energy( coords_all, self.gamma)
        return total_conf_energy
# =============================================================================


SWL = SWLayer()
coords2 = tf.constant([[
                        1.9434,  1.0817,  1.0803,  
                        2.6852,  2.7203,  1.0802,  
                        1.3807,  1.3573,  1.3307]])

with tf.GradientTape() as tape:
    tape.watch(coords2)
    E = SWL( coords2)