from tensorflow import keras
from tensorflow.keras import layers

class QLSTM(tf.keras.layers.Layer):
    def __init__(self, units, symbols, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.lstm_cell = tf.keras.layers.LSTMCell(units, activation=activation)
        self.expectation = tfq.layers.Expectation()
        self.symbol_names = symbols
        # State size: LSTM hidden state, LSTM outputs, QPU expectancy
        self.state_size = [tf.TensorShape([2]),tf.TensorShape([2]),tf.TensorShape([1])]
        self.output_size = tf.TensorShape([1]) # QPU expectation

    def call(self,inputs,state):
        circuits = inputs[0]
        ops = inputs[1]

        joined_state = tf.keras.layers.concatenate(state[1],state[2])

        output_lstm, hidden_state = self.lstm_cell(joined_state,state[0])
        exp_out = self.expectation(circuits,
            symbol_names=self.symbol_names,
            symbol_values=output_lstm,
            operators=ops
        )
        return exp_out, [hidden_state,output_lstm,exp_out]

rnn = tf.keras.layers.RNN(QLSTM(2,qaoa_symbols),return_sequences=True)

op_inp = tf.keras.Input(shape=(10,1,), dtype=tf.dtypes.string)
circuit_inp = tf.keras.Input(shape=(10,1,), dtype=tf.dtypes.string)

rnn_2 = rnn([circuit_inp,op_inp])

import networkx as nx
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import numpy as np
import random
import sympy

from cirq.contrib.svg import SVGCircuit

random.seed(123)

def maxcut_qaoa_from_graph(graph, p):
    """
    Function to generate a Cirq QAOA circuit for a given
    graph G with depth P.
    This circuit is parametrized using sympy.symbols
    """
    qubits = cirq.GridQubit.rect(1, len(graph.nodes))
    qaoa_circuit = cirq.Circuit()
    # Initial equal superposition
    for qubit in qubits:
        qaoa_circuit += cirq.H(qubit)
    qaoa_symbols = []
    # Stack the parameterized costs and mixers
    for l_num in range(p):
        qaoa_symbols.append(sympy.Symbol("gamma_{}".format(l_num)))
        for e in graph.edges():
            qaoa_circuit.append(cirq.CNOT(control=qubits[e[0]], target=qubits[e[1]]),strategy=cirq.InsertStrategy.NEW)
            qaoa_circuit.append(cirq.rz(qaoa_symbols[-1])(qubits[e[1]]),strategy=cirq.InsertStrategy.NEW)
            qaoa_circuit.append(cirq.CNOT(control=qubits[e[0]], target=qubits[e[1]]),strategy=cirq.InsertStrategy.NEW)
        qaoa_symbols.append(sympy.Symbol("eta_{}".format(l_num)))
        qaoa_circuit.append([cirq.rx(2*qaoa_symbols[-1])(qubits[n]) for n in graph.nodes()], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    # Define the cost as a Cirq PauliSum
    cost_op = None
    for e in graph.edges():
        if cost_op is None:
            cost_op = cirq.Z(qubits[e[0]])*cirq.Z(qubits[e[1]])
        else:
            cost_op += cirq.Z(qubits[e[0]])*cirq.Z(qubits[e[1]])
    return qaoa_circuit, qaoa_symbols, cost_op

def generate_data(n_nodes_min,n_nodes_max,n_points,p):
    """
    This function generates `n_points` number of cirq circuits,
    each one corresponding to one random graph, as well as the
    corresponding cost function for the QAOA problem.
    Finally, it also outputs the parameters of the graphs.
    """
    datapoints = []
    costs = []
    graphs =[] 
    for _ in range(n_points):
        n_nodes = random.randint(n_nodes_min,n_nodes_max)
        random_graph = nx.random_regular_graph(n=n_nodes,d=3)
        circuit, symbols, cost_op = maxcut_qaoa_from_graph(random_graph, p)
        datapoints.append(circuit)
        costs.append([cost_op])
        graphs.append(random_graph)
    return datapoints,symbols,costs,graphs

qaoa_circuit, qaoa_symbols, cost_op = maxcut_qaoa_from_graph(maxcut_graph, P)

class QLSTM(tf.keras.layers.Layer):
    def __init__(self, units, symbols, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.lstm_cell = tf.keras.layers.LSTMCell(units, activation=activation)
        self.expectation = tfq.layers.Expectation()
        self.symbol_names = symbols
        # State size: LSTM hidden state, LSTM outputs, QPU expectancy
        self.state_size = [tf.TensorShape([2]),tf.TensorShape([2]),tf.TensorShape([1])]
        self.output_size = tf.TensorShape([1]) # QPU expectation

    def call(self,inputs,state):
        circuits = inputs[0]
        ops = inputs[1]

        joined_state = tf.keras.layers.concatenate(state[1],state[2])

        output_lstm, hidden_state = self.lstm_cell(joined_state,state[0])
        exp_out = self.expectation(circuits,
            symbol_names=self.symbol_names,
            symbol_values=output_lstm,
            operators=ops
        )
        return exp_out, [hidden_state,output_lstm,exp_out]

# Generate random MaxCut instances as training data.
N_QUBITS_MIN = 8
N_QUBITS_MAX = 8
P = 2
N_POINTS = 2

# For a more accurate optimizer on testing data, increase N_POINTS
circuits, symbols, ops, graphs = generate_data(N_QUBITS_MIN, N_QUBITS_MAX, N_POINTS, P)

TIMESTEPS = 10

circuit_tensor = tfq.convert_to_tensor(circuits)
ops_tensor = tfq.convert_to_tensor(ops)

circuit_tensor_t = tf.expand_dims(circuit_tensor, 1, name=None)
ops_tensor_t = tf.expand_dims(ops_tensor, 1, name=None)

circuit_tensor_rep = tf.repeat(circuit_tensor_t, TIMESTEPS, axis=-1, name=None)
ops_tensor_rep = tf.repeat(ops_tensor_t, TIMESTEPS, axis=-1, name=None)

ops_tensor_rep = tf.reshape(ops_tensor_rep,(-1,TIMESTEPS,1))
circuit_tensor_rep = tf.reshape(circuit_tensor_rep,(-1,TIMESTEPS,1))

op_inp = tf.keras.Input(shape=(10,1,), dtype=tf.dtypes.string)
circuit_inp = tf.keras.Input(shape=(10,1,), dtype=tf.dtypes.string)

rnn = tf.keras.layers.RNN(QLSTM(2,qaoa_symbols),return_sequences=True)

rnn_2 = rnn([circuit_inp,op_inp])