# tf.random.uniform((B, 2), dtype=tf.float32) ← Input shape (batch_size, 2)

import tensorflow as tf
import numpy as np

class Rescale(tf.keras.layers.Layer):
    def __init__(self, scale_limit=np.pi, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.scale_limit = scale_limit

    def call(self, inputs):
        s = tf.reduce_sum(inputs, axis=-1, keepdims=True)
        # avoid division by zero by adding a small epsilon
        s = tf.where(tf.equal(s, 0), tf.ones_like(s) * 1e-7, s)
        return (inputs / s) * self.scale_limit

class DenseQKan(tf.keras.layers.Layer):
    def __init__(self, units: int, circuit, layers: int, **kwargs):
        super().__init__(**kwargs)
        self.circuit = circuit
        self.qubits = len(circuit.device.wires)
        self.units = units
        self.qbatches = None
        self.layers = layers

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if input_dim > self.qubits:
            self.qbatches = np.ceil(input_dim / self.qubits).astype(np.int32)
        else:
            self.qbatches = 1
        self.layer_weights = []
        for u in range(self.units):
            # Weights shape:
            # (qbatches, input_dim_per_batch, layers)
            w_shape = (self.qbatches, input_dim // self.qbatches, self.layers)
            self.layer_weights.append(
                self.add_weight(
                    shape=w_shape,
                    initializer=tf.keras.initializers.RandomUniform(
                        minval=-np.pi, maxval=np.pi
                    ),
                    trainable=True,
                    name=f"weights_unit_{u}"
                )
            )
        self.built = True

    def compute_output_shape(self, input_shape):
        # Output shape is (batch_size, units)
        return (input_shape[0], self.units)

    def call(self, inputs):
        assert self.qbatches is not None
        splits = tf.split(inputs, self.qbatches, axis=-1)  # List of tensors
        out = []
        for u in range(self.units):
            unit_out = 0
            for qb in range(self.qbatches):
                # circuit call: circuit(splits[qb], weights_for_unit_qb)
                # The circuit presumably returns a vector to sum over axis=-1
                qb_out = tf.reduce_sum(
                    tf.stack(self.circuit(splits[qb], self.layer_weights[u][qb]), axis=-1), axis=-1
                )
                unit_out += qb_out
            out.append(unit_out)
        out = tf.stack(out, axis=-1)
        # Fix to enable model.summary() shape inference
        out = tf.reshape(out, (tf.shape(inputs)[0], self.units))
        return out

class MyModel(tf.keras.Model):
    def __init__(self, units, circuit, layers, input_shape=2):
        super().__init__()
        self.dense_qkan = DenseQKan(units, circuit, layers, name="DenseKAN")
        self.rescale = Rescale(name="RescalePi")
        self._input_shape_ = input_shape

    def call(self, inputs):
        x = self.dense_qkan(inputs)
        x = self.rescale(x)
        return x

def my_model_function():
    # For this example, we need to create a dummy circuit object with QNode interface.
    # The real circuit should be provided by the user, but we provide a simplified mock.

    class DummyCircuit:
        def __init__(self, wires):
            # Simulate a QNode device with wires information
            self.device = type("device", (), {"wires": wires})

        def __call__(self, inputs, weights):
            # inputs shape: (input_dim_for_batch,)
            # weights shape: (layers,)
            # We simulate outputs as sin(inputs * sum(weights)) with shape (layers,)
            # to create a tensor we can stack and reduce_sum over in DenseQKan.call

            # Broadcasting weights and inputs to multiply and apply sin
            # weights shape (layers,), inputs shape (?,)
            # We'll return a tensor of shape (layers,), one value per layer
            # For simplicity, sum inputs multiplied by weights per layer -> scalar per layer

            # Using tf.reduce_sum(inputs) * weights as input to sin
            sum_inputs = tf.reduce_sum(inputs)
            val_per_layer = tf.sin(sum_inputs * weights)
            # Return a vector of shape (layers,)
            return val_per_layer

    # Parameters to match example in issue
    units = 10
    circuit = DummyCircuit(wires=[0, 1])  # 2 qubits, aligned to input_dim=2
    layers = 5
    input_shape = 2

    model = MyModel(units=units, circuit=circuit, layers=layers, input_shape=input_shape)
    return model

def GetInput():
    # Generate random input of shape (batch_size, 2)
    batch_size = 4  # arbitrary batch size
    return tf.random.uniform((batch_size, 2), dtype=tf.float32)

