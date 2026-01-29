# tf.random.uniform((B, 3)), tf.random.uniform((B, 4)), tf.random.uniform((B, 5)), tf.random.uniform((B, 5)), tf.random.uniform((B, 4)), tf.random.uniform((B, 1)), dtype=tf.float32

import tensorflow as tf
from tensorflow.keras.constraints import non_neg
from tensorflow.keras.layers import Dense, concatenate, Input

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define one Dense layer with sigmoid and non-negative kernel constraint per input branch
        self.dense_1 = Dense(1, activation='sigmoid', kernel_constraint=non_neg())
        self.dense_2 = Dense(1, activation='sigmoid', kernel_constraint=non_neg())
        self.dense_3 = Dense(1, activation='sigmoid', kernel_constraint=non_neg())
        self.dense_4 = Dense(1, activation='sigmoid', kernel_constraint=non_neg())
        self.dense_5 = Dense(1, activation='sigmoid', kernel_constraint=non_neg())
        # Final output Dense with softmax and non-negative constraint
        self.output_dense = Dense(2, activation='softmax', kernel_constraint=non_neg())
    
    def call(self, inputs, training=False):
        # inputs is expected to be a dict of named inputs, matching keys:
        # 'green_fin_const', 'green_fin_inst', 'gov_sup', 'com_act', 'eco_city', 'type'
        # Extract inputs from dict
        x1 = inputs['green_fin_const']  # shape (B, 3)
        x2 = inputs['green_fin_inst']   # shape (B, 4)
        x3 = inputs['gov_sup']          # shape (B, 5)
        x4 = inputs['com_act']          # shape (B, 5)
        x5 = inputs['eco_city']         # shape (B, 4)
        x6 = inputs['type']             # shape (B, 1)
        
        # Pass each through corresponding Dense + sigmoid
        score_1 = self.dense_1(x1)  # (B,1)
        score_2 = self.dense_2(x2)  # (B,1)
        score_3 = self.dense_3(x3)  # (B,1)
        score_4 = self.dense_4(x4)  # (B,1)
        score_5 = self.dense_5(x5)  # (B,1)
        
        # Concatenate scores and type input tensor along last axis
        concatenated = concatenate([score_1, score_2, score_3, score_4, score_5, x6], axis=-1)  # (B, 6)
        
        # Final output layer producing 2-class softmax probabilities
        outputs = self.output_dense(concatenated)  # (B, 2)
        
        return outputs

def my_model_function():
    # Return an instance of MyModel, initialized
    model = MyModel()
    
    # Build the model by calling with sample inputs to create weights (optional but recommended)
    # shape info: batch size arbitrary; here used 1
    sample_input = {
        'green_fin_const': tf.zeros((1, 3), dtype=tf.float32),
        'green_fin_inst': tf.zeros((1, 4), dtype=tf.float32),
        'gov_sup': tf.zeros((1, 5), dtype=tf.float32),
        'com_act': tf.zeros((1, 5), dtype=tf.float32),
        'eco_city': tf.zeros((1, 4), dtype=tf.float32),
        'type': tf.zeros((1, 1), dtype=tf.float32),
    }
    model(sample_input)
    return model

def GetInput():
    # Return a dictionary of inputs matching those expected by MyModel call
    # Here batch size=4 arbitrarily chosen for example
    B = 4
    inputs = {
        'green_fin_const': tf.random.uniform((B, 3), dtype=tf.float32),
        'green_fin_inst': tf.random.uniform((B, 4), dtype=tf.float32),
        'gov_sup': tf.random.uniform((B, 5), dtype=tf.float32),
        'com_act': tf.random.uniform((B, 5), dtype=tf.float32),
        'eco_city': tf.random.uniform((B, 4), dtype=tf.float32),
        'type': tf.random.uniform((B, 1), dtype=tf.float32),
    }
    return inputs

