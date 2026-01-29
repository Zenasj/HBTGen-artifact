# tf.random.uniform((B, 10, 1), dtype=tf.dtypes.string)
import tensorflow as tf
import tensorflow_quantum as tfq

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Parameters from the original code snippet and assumptions:
        # LSTM units: 2 as in the original QLSTM
        # The custom QLSTM cell combines an LSTMCell and TFQ Expectation layer,
        # managing three state components:
        # 1) hidden state of LSTM (tuple of (h, c), each shape [units])
        # 2) output of LSTM (shape [units])
        # 3) scalar expectation output (shape [1])

        # We will implement the core custom RNN cell inline to integrate with tf.keras.RNN
        
        self.units = 2
        self.expectation = tfq.layers.Expectation()
        # Symbols would be an input, so for demonstration use a placeholder empty list
        # In practice the symbols should come from the QAOA circuit construction
        # Here we must provide symbols to the expectation layer in call
        self.symbol_names = []  # placeholder empty list for TFQ symbols

        # Inner LSTMCell
        self.lstm_cell = tf.keras.layers.LSTMCell(self.units)

        # Compose a RNN layer wrapping our QLSTMCell logic (see call below)
        self.rnn_layer = tf.keras.layers.RNN(self, return_sequences=True)

    @property
    def state_size(self):
        # Following the original:
        # LSTM hidden state is a tuple of two tensors each shape (units,)
        # output_lstm shape (units,)
        # expectation output shape (1,)
        # Since internal LSTMCell states are (h, c) each units shape,
        # state_size can be (units, units) tuple
        # We encapsulate three states as tuple/list for Keras RNN

        # Because we're defining call, and returning states as a list/tuple of 3,
        # we implement state_size accordingly as a list of shapes
        return [
            tf.TensorShape([self.units]),       # LSTM hidden state h
            tf.TensorShape([self.units]),       # LSTM cell state c
            tf.TensorShape([self.units]),       # LSTM output
            tf.TensorShape([1])                  # scalar expectation
        ]

    @property
    def output_size(self):
        # The output at each timestep is the scalar expectation (shape [1])
        return tf.TensorShape([1])

    def call(self, inputs, states):
        # inputs are expected as tuple/list of two tensors:
        # inputs[0]: batch of circuits with shape (batch_size,), dtype string
        # inputs[1]: batch of operators with shape (batch_size,), dtype string

        # The original code designs the inputs as a list of two tensors of shape [None, 10, 1],
        # later fed timestep by timestep by RNN. Here, call runs per timestep with shape [batch, ...]

        # states is a list of four elements according to state_size:
        # states[0]: prev LSTM hidden state h
        # states[1]: prev LSTM cell state c
        # states[2]: prev LSTM output
        # states[3]: prev expectation scalar output

        circuits = inputs[0]
        operators = inputs[1]

        # The original code attempts to concatenate previous lstm output and expectation
        # to form the input to the LSTM cell.
        # Since LSTMCell expects input shape (batch, input_dim), we concatenate on last dim

        # Note: states[2] shape (batch, units)
        #       states[3] shape (batch, 1)
        joined_state = tf.concat([states[2], states[3]], axis=-1)  # shape (batch, units+1)

        # Pass joined_state and previous lstm states tuple (h,c) to LSTMCell
        lstm_states = (states[0], states[1])  # h, c
        output_lstm, lstm_states_new = self.lstm_cell(joined_state, lstm_states)

        # Compute the expectation using tensorflow-quantum layer
        # symbol_values is mapped to output of LSTM cell (output_lstm)
        # Normally, symbol_names is a list of strings representing symbols,
        # but since original symbols missing here, we add empty list
        exp_out = self.expectation(
            circuits,
            symbol_names=self.symbol_names,
            symbol_values=output_lstm,
            operators=operators
        )

        # Return output: the scalar expectation, shape (batch, 1)
        # New states are updated: h and c from lstm_states_new, output_lstm, and exp_out
        new_states = [
            lstm_states_new[0],  # new h
            lstm_states_new[1],  # new c
            output_lstm,         # new output_lstm
            exp_out              # new expectation scalar
        ]

        return exp_out, new_states


def my_model_function():
    # Construct the model to accept inputs compatible with the above cell in an RNN wrapper

    # Inputs: tuple/list of two tensors shaped (batch, 10, 1) strings representing cirq circuits/operators
    circuit_input = tf.keras.Input(shape=(10, 1), dtype=tf.dtypes.string, name="circuit_input")
    op_input = tf.keras.Input(shape=(10, 1), dtype=tf.dtypes.string, name="operator_input")

    # Wrap the QLSTM cell in tf.keras.layers.RNN with return_sequences=True
    # We implement a tf.keras RNN layer with MyModel as a cell
    # However, the custom logic needed is inside MyModel, so we will create a RNN layer wrapping it

    # The issue in the original code stems from mismatched initial_state and state_size.
    # Here, to keep it simple, we build the RNN with custom cell class, composed correctly.

    # Unfortunately, MyModel defines call as the cell call,
    # but tf.keras.RNN expects the cell to be a Layer with state_size etc.
    # Our MyModel is a full model subclass, so separate the cell.
    # To comply with requirements, we make MyModel itself the cell class.

    # Thus, create an instance of MyModel as a cell and then wrap tf.keras.layers.RNN around it.
    # The inputs to RNN should be a tuple (circuits_t, ops_t) both of shape (batch, 1)
    # So we need to split inputs along time axis and feed as a tuple.

    # Define a Lambda layer to extract each time step from inputs and feed as tuple.

    # We will define a custom layer to stack the inputs per time step into tuple inputs,
    # so that the tf.keras.layers.RNN layer can handle tuples.

    class TupleRNNCell(tf.keras.layers.Layer):
        # Wrap the MyModel cell logic here

        def __init__(self):
            super().__init__()
            self.cell = MyModel()

        @property
        def state_size(self):
            return self.cell.state_size

        @property
        def output_size(self):
            return self.cell.output_size

        def call(self, inputs, states):
            # inputs is a tuple/list of two tensors (circuits, ops), each shape (batch, 1)
            return self.cell.call(inputs, states)

    # Prepare the input tuple expected by RNN: sequences of pairs (circuit, op)
    # Inputs have shape (batch, 10, 1). We transpose to (10, batch, 1), then for each time step,
    # input to cell is a tuple of (circuits_t, ops_t)

    circuit_seq = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(circuit_input)
    op_seq = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(op_input)

    # Stack the two sequences along last dimension to create tuple inputs for RNN:
    # We define a custom RNN that takes tuples as inputs via the cell call

    # Compose inputs as a list of two tensors for each time step
    sequence_inputs = [circuit_seq, op_seq]  # each shape (batch, 10)

    # Custom RNN to handle tuple inputs: we create a tf.keras.layers.RNN with a custom cell
    rnn_layer = tf.keras.layers.RNN(TupleRNNCell(), return_sequences=True)

    # Keras RNN expects inputs shaped (batch, time, features)
    # Our features are a tuple, so keras RNN cannot handle tuples natively.
    # We solve this by defining a wrapper layer that merges inputs by concatenation
    # But since inputs are strings, concatenation does not apply.
    # Workaround: Flatten and feed as a list passed to the cell.

    # Another way is to pass the tuple inputs as a nested structure,
    # Using keras masking for ragged handling.

    # Here, we pack the circuit and operator sequences into a single tuple per time-step
    # We'll use tf.keras.layers.RNN's 'input_tensor' parameter which supports nested inputs.

    # So, package the calls with these inputs:
    rnn_out = rnn_layer([circuit_seq, op_seq])

    # The output is the sequence of scalar expectations for all timesteps
    model = tf.keras.Model(inputs=[circuit_input, op_input], outputs=rnn_out)
    return model

def GetInput():
    """
    Return random input tensors matching the expected shape:
    - batch size: choose 2 for demonstration
    - timesteps: 10 (as in original code)
    - features: 1
    Each tensor is of `tf.string` dtype representing Cirq circuits/operators in TFQ

    Since real Cirq circuits require proper construction using Cirq and TFQ,
    here we generate dummy placeholder strings.

    In practical usage, these would be tfq.convert_to_tensor(circuits) where circuits are cirq.Circuit objects.
    """

    batch_size = 2
    timesteps = 10

    # Create dummy string tensors
    dummy_circuit = tf.constant([""]*batch_size)  # empty string as placeholder circuits
    dummy_op = tf.constant([""]*batch_size)       # empty string as placeholder operators

    # Tile to timesteps along axis=1 and add feature axis=2=1
    dummy_circuit = tf.tile(tf.expand_dims(dummy_circuit, axis=1), [1, timesteps])
    dummy_circuit = tf.expand_dims(dummy_circuit, axis=-1)  # shape (batch_size, timesteps, 1)

    dummy_op = tf.tile(tf.expand_dims(dummy_op, axis=1), [1, timesteps])
    dummy_op = tf.expand_dims(dummy_op, axis=-1)  # shape (batch_size, timesteps, 1)

    return [dummy_circuit, dummy_op]

