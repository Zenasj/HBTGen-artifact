# tf.random.uniform((batch_size, timestep, input_1), dtype=tf.float32), tf.random.uniform((batch_size, timestep, input_2, input_3), dtype=tf.float32)
import tensorflow as tf
import collections

# Namedtuples to structure inputs and states
NestedInput = collections.namedtuple("NestedInput", ["feature1", "feature2"])
NestedState = collections.namedtuple("NestedState", ["state1", "state2"])

class MyModel(tf.keras.Model):
    def __init__(self, unit_1=10, unit_2=20, unit_3=30, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.unit_1 = unit_1
        self.unit_2 = unit_2
        self.unit_3 = unit_3
        
        # Define the nested RNN cell with complex state and input structure
        self.cell = NestedCell(unit_1, unit_2, unit_3)
        
        # Stateful=True as per the issue context, to preserve states across batches
        self.rnn = tf.keras.layers.RNN(self.cell, stateful=True)
    
    def call(self, inputs, training=None):
        # inputs is expected to be NestedInput(feature1, feature2)
        # where feature1: (batch, time, input_1)
        #       feature2: (batch, time, input_2, input_3)
        return self.rnn(inputs)

class NestedCell(tf.keras.layers.Layer):
    def __init__(self, unit_1, unit_2, unit_3, **kwargs):
        super(NestedCell, self).__init__(**kwargs)
        self.unit_1 = unit_1
        self.unit_2 = unit_2
        self.unit_3 = unit_3
        
        # Define state sizes for the nested state structure:
        # state1 shape: unit_1 (vector)
        # state2 shape: (unit_2, unit_3) matrix
        self.state_size = NestedState(
            state1=unit_1, 
            state2=tf.TensorShape([unit_2, unit_3])
        )
        # Output size matches the structure of outputs:
        self.output_size = NestedInput(
            feature1=unit_1,
            feature2=tf.TensorShape([unit_2, unit_3])
        )

    def build(self, input_shapes):
        # input_shapes is a NestedInput with shapes:
        # feature1: (batch, input_1)
        # feature2: (batch, input_2, input_3)
        input_1 = input_shapes.feature1[-1]  # last dim of feature1 input
        input_2, input_3 = input_shapes.feature2[-2], input_shapes.feature2[-1]  # last two dims for feature2
        
        # Weight matrices for each input branch
        self.kernel_1 = self.add_weight(
            shape=(input_1, self.unit_1),
            initializer="uniform",
            name="kernel_1"
        )
        # kernel_2_3 shape: (input_2, input_3, unit_2, unit_3)
        # For tensor contraction with input_2 shaped (batch, input_2, input_3)
        self.kernel_2_3 = self.add_weight(
            shape=(input_2, input_3, self.unit_2, self.unit_3),
            initializer="uniform",
            name="kernel_2_3"
        )
        super(NestedCell, self).build(input_shapes)

    def call(self, inputs, states):
        # inputs is NestedInput with tensors:
        # - feature1: (batch, input_1)
        # - feature2: (batch, input_2, input_3)
        input_1, input_2 = tf.nest.flatten(inputs)
        
        # states is NestedState with tensors:
        # - state1: (batch, unit_1)
        # - state2: (batch, unit_2, unit_3)
        s1, s2 = states
        
        # Compute outputs by multiplying inputs with weights
        # output_1: (batch, unit_1)
        output_1 = tf.matmul(input_1, self.kernel_1)
        
        # output_2_3: einsum to contract input_2 and kernel_2_3 tensors
        # input_2 shape: (batch, input_2, input_3)
        # kernel_2_3 shape: (input_2, input_3, unit_2, unit_3)
        # Result is (batch, unit_2, unit_3)
        output_2_3 = tf.einsum("bij,ijkl->bkl", input_2, self.kernel_2_3)
        
        # New states are updated by adding outputs to previous states (simple recurrence)
        state_1 = s1 + output_1
        state_2 = s2 + output_2_3

        # Outputs follow the nested structure
        output = NestedInput(feature1=output_1, feature2=output_2_3)
        new_states = NestedState(state1=state_1, state2=state_2)
        
        return output, new_states

def my_model_function():
    # Return an instance of MyModel with default units as in the original example
    return MyModel(unit_1=10, unit_2=20, unit_3=30)

def GetInput():
    # Returns a NestedInput of two tensors with batch_size, timestep, and respective input shapes
    
    batch_size = 64
    timestep = 50
    input_1 = 32
    input_2 = 64
    input_3 = 32
    
    # Random uniform inputs with float32 dtype
    
    feature1 = tf.random.uniform(
        shape=(batch_size, timestep, input_1),
        dtype=tf.float32
    )
    feature2 = tf.random.uniform(
        shape=(batch_size, timestep, input_2, input_3),
        dtype=tf.float32
    )
    
    return NestedInput(feature1=feature1, feature2=feature2)

