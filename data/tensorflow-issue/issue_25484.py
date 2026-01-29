# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← Based on placeholder shape [None, 32, 32, 3]

import tensorflow as tf
import numpy as np


class MyModel(tf.keras.Model):
    def __init__(self, network_architecture=None):
        super().__init__()
        # Use provided network_architecture or defaults
        if network_architecture is None:
            network_architecture = {
                "channels": 10,   # number of output channels
                "num_layers": 6,  # number of convolutional layers after first layer
            }
        self._channels = network_architecture["channels"]
        self._num_layers = network_architecture["num_layers"]
        
        # Parameters per conv2d layer
        self.kernel_shape = (5, 5)
        self.padding = 'SAME'
        
        # Store variable scopes for layers separately for reuse in tf1 style,
        # We implement analogous behavior by using tf.keras.Variable and managing weightnorm manually.
        # Note: This is an inferred adaptation from the original TF1 variable_scope/tf.get_variable code.

        # We build kernel variables, g, and b variables in lists for all layers.
        # For simplicity, we use Variable directly and control initialization with helper methods.
        # We'll simulate the two approaches fused into one model with comparison capability.

        # Lists to store variables per layer for the two approaches
        self.v_vars = []
        self.g_vars = []
        self.b_vars = []

        self.v_init_vars = []
        self.g_init_vars = []
        self.b_init_vars = []

        # Weights for initializers-based method (v_aux tensors, etc)
        self.v_aux_tensors = []

        # Create variables for all layers in __init__ (to imitate TF1 get_variable behavior)
        # Assume input channels = 3 for first layer, else self._channels
        in_channels = 3
        stride_first = (2, 2)
        stride_other = (1, 1)
        
        for i in range(self._num_layers + 1):  # +1 for first layer
            out_channels = self._channels
            stride = stride_first if i == 0 else stride_other
            input_c = in_channels if i == 0 else self._channels
            filter_shape = [self.kernel_shape[0], self.kernel_shape[1], input_c, out_channels]
            
            # Variables for weight-sharing approach
            # v initialized randomly
            v_init = tf.random.normal(filter_shape, mean=0.0, stddev=0.05, dtype=tf.float32)
            v = tf.Variable(v_init, trainable=True, name=f'v_{i}')
            
            # Compute data-dependent initialization for g and b following original logic
            # Normalize v for conv initialization
            v_norm = tf.nn.l2_normalize(v, axis=[0,1,2])  # shape [kh, kw, in_c, out_c]
            
            # Kernel stride shape for conv2d
            stride_shape = [1, stride[0], stride[1], 1]

            # Placeholder input for computing moments in init (we store this for deferred initialization)
            # This is a complication: original code uses x (inputs) to compute moments which requires
            # calling build_net(x) with actual input tensor.
            # We'll store these variables but delay usage to call when input is available.

            # For now, set g and b as variables but initialize later in call() with input-dependent values
            
            # Placeholders to store moments and initial g,b for weight sharing implementation
            # We initialize g and b as trainable variables with initial value of zeros, will assign later
            g = tf.Variable(tf.zeros([out_channels], dtype=tf.float32), trainable=True, name=f'g_{i}')
            b = tf.Variable(tf.zeros([out_channels], dtype=tf.float32), trainable=True, name=f'b_{i}')

            self.v_vars.append(v)
            self.g_vars.append(g)
            self.b_vars.append(b)

            # Variables for initializers-based approach
            # v_aux is a constant tensor sampled once, mimicking initialization from constant tensor.
            v_aux_np = np.random.normal(loc=0, scale=0.05, size=filter_shape).astype(np.float32)
            v_aux = tf.constant(v_aux_np, dtype=tf.float32, name=f'v_aux_{i}')
            self.v_aux_tensors.append(v_aux)
            # g_init, b_init similarly initialized as variables with zeros, updated later in call()
            g_init = tf.Variable(tf.zeros([out_channels], dtype=tf.float32), trainable=True, name=f'g_init_{i}')
            b_init = tf.Variable(tf.zeros([out_channels], dtype=tf.float32), trainable=True, name=f'b_init_{i}')
            self.g_init_vars.append(g_init)
            self.b_init_vars.append(b_init)
            # v_init variable for initializer based method (trainable, initialized from v_aux)
            v_init_var = tf.Variable(v_aux_np, trainable=True, name=f'v_init_var_{i}')
            self.v_init_vars.append(v_init_var)


    def call(self, x):
        """
        Args:
            x: input tensor of shape [B, 32, 32, 3]
        Returns:
            A boolean tensor indicating if the outputs of two approaches are close elementwise.
        """

        # We follow two parallel computations:
        # 1. Weight sharing approach: uses variables v, g, b, compute conv output
        #     g,b are updated to data-dependent initialization based on input x
        # 2. Initializers approach: uses variables initialized from constants (v_aux),
        #    g_init,b_init updated similarly based on input x.
        #
        # After building outputs from both methods, we compare the tensor outputs for equality/tolerance.

        # Internal helper processing for layer i:
        def conv_weightnorm_vgb(inputs, v, g, b, stride):
            # Normalize v
            v_norm = tf.nn.l2_normalize(v, axis=[0,1,2])
            w = tf.reshape(tf.exp(g), [1,1,1,-1]) * v_norm
            stride_shape = [1, stride[0], stride[1], 1]
            b_reshape = tf.reshape(b, [1,1,1,-1])
            r = tf.nn.conv2d(inputs, w, stride_shape, padding=self.padding) + b_reshape
            return r

        def data_dependent_init_vgb(inputs, v):
            """
            Given inputs and v weights, compute moments of conv output and return computed g and b
            following the formula from weight normalization paper.
            """
            stride_shape = [1, 1, 1, 1]  # temporary - real stride depends on layer and provided per call
            stride_shape = None # will be passed from context

            # Function to compute g and b from conv output moments: m_init, v_init
            def compute_g_b(x_init):
                m_init, v_init = tf.nn.moments(x_init, axes=[0,1,2])  # moments across batch and spatial dims
                scale_init = 1.0 / tf.sqrt(v_init + 1e-10)
                g_val = tf.math.log(scale_init) / 3.0
                b_val = -m_init * scale_init
                return g_val, b_val, m_init, scale_init

            return compute_g_b

        # Now construct the two nets for all layers, track inputs for each layer
        hs_weightsharing = []
        hs_initializers = []

        # Initial inputs for first layer conv:
        input_ws = x
        input_init = x

        # Strides handling
        stride_first = (2, 2)
        stride_other = (1, 1)

        # We'll store outputs for comparison of two methods per layer
        for i in range(self._num_layers + 1):
            v_ws = self.v_vars[i]
            g_ws = self.g_vars[i]
            b_ws = self.b_vars[i]

            v_init_var = self.v_init_vars[i]
            g_init_var = self.g_init_vars[i]
            b_init_var = self.b_init_vars[i]
            v_aux = self.v_aux_tensors[i]

            # Set strides
            stride = stride_first if i == 0 else stride_other

            # Weight sharing approach: data-dependent initialization of g,b from input_ws and v_ws
            # Compute x_init as conv2d(input_ws, normalized v_ws)
            v_ws_norm = tf.nn.l2_normalize(v_ws, axis=[0,1,2])
            stride_shape = [1, stride[0], stride[1], 1]
            x_init = tf.nn.conv2d(input_ws, v_ws_norm, stride_shape, padding=self.padding)

            # Compute moments and data dependent g,b (tf operations)
            m_init, v_init = tf.nn.moments(x_init, axes=[0,1,2])  # moments along batch and spatial dims
            scale_init = 1.0 / tf.sqrt(v_init + 1e-10)
            g_val_ws = tf.math.log(scale_init) / 3.0
            b_val_ws = -m_init * scale_init

            # Assign new values to g_ws and b_ws (eager mode, assign update)
            g_ws.assign(g_val_ws)
            b_ws.assign(b_val_ws)

            # Compute output for weight sharing approach
            h_ws = conv_weightnorm_vgb(input_ws, v_ws, g_ws, b_ws, stride)
            hs_weightsharing.append(h_ws)

            # Initializers-based approach:
            # We use v_init_var (trainable), initialized from v_aux
            v_init_norm = tf.nn.l2_normalize(v_init_var, axis=[0,1,2])
            x_init_init = tf.nn.conv2d(input_init, v_init_norm, stride_shape, padding=self.padding)
            # Compute moments likewise
            m_init_init, v_init_init = tf.nn.moments(x_init_init, axes=[0,1,2])
            scale_init_init = 1.0 / tf.sqrt(v_init_init + 1e-10)
            g_val_init = tf.math.log(scale_init_init) / 3.0
            b_val_init = -m_init_init * scale_init_init

            # Assign to g_init_var and b_init_var
            g_init_var.assign(g_val_init)
            b_init_var.assign(b_val_init)

            # Compute output for initializer-based method
            w_init = tf.reshape(tf.exp(g_init_var), [1,1,1,-1]) * v_init_norm
            b_init_reshape = tf.reshape(b_init_var, [1,1,1,-1])
            h_init = tf.nn.conv2d(input_init, w_init, stride_shape, padding=self.padding) + b_init_reshape

            hs_initializers.append(h_init)

            # Inputs for next layer:
            input_ws = h_ws
            input_init = h_init

        # After all layers computed, compare last layer outputs for equality/tolerance
        output_ws = hs_weightsharing[-1]
        output_init = hs_initializers[-1]

        # Compare elementwise closeness within tolerance (e.g., 1e-5)
        tolerance = 1e-5
        close = tf.math.abs(output_ws - output_init) < tolerance
        all_close = tf.reduce_all(close)

        # Return boolean tensor indicating if outputs are close
        return all_close


def my_model_function():
    # Default network architecture used in issue
    network_architecture = {
        "channels": 10,
        "num_layers": 6,
    }
    return MyModel(network_architecture)


def GetInput():
    # Return a random float32 tensor matching the expected input shape [B, 32, 32, 3]
    batch_size = 4  # arbitrary batch size for testing
    return tf.random.uniform((batch_size, 32, 32, 3), dtype=tf.float32)

