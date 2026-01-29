# tf.random.uniform((batch, 3), dtype=tf.float32) ← input shape inferred as (batch_size, 3)

import tensorflow as tf
import math as m

pi = tf.constant(m.pi, dtype=tf.float32)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define 9 hidden layers with 200 units each and tanh activation
        self.hidden_layers = [tf.keras.layers.Dense(200, activation='tanh', dtype='float32') for _ in range(9)]
        # Output layer with linear activation
        self.output_layer = tf.keras.layers.Dense(1, activation='linear', dtype='float32')

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output

def custom_loss(model):
    # The loss function depends on the model predictions and derivatives wrt inputs
    def loss(y_true, y_pred):
        # Generate random points for boundary and domain conditions
        
        # Shapes inferred from original code:
        # x0, y0: shape [2, 100], t0 shape [100, 2]
        # pX0, pX1, pY0, pY1: boundary conditions inputs shape [100,3]
        # pIC: initial condition inputs shape [batch, 3]
        # pF: domain points for PDE residual shape [1000, 3]
        
        # Note: In the original code dimensions had some inconsistencies with transposes,
        # so we clarify shapes here for consistent input to model:
        
        x0 = tf.random.uniform(shape=[2,100], maxval=1, dtype=tf.float32)
        y0 = tf.random.uniform(shape=[2,100], maxval=1, dtype=tf.float32)
        t0 = tf.random.uniform(shape=[100,2], maxval=1, dtype=tf.float32)
        
        # Construct pY0, pY1, pX0, pX1 as 100x3 tensors representing boundary points:
        # These encode points along boundaries with fixed coordinates
        
        pY0 = tf.transpose(tf.concat([
            tf.ones([1, y0.shape[1]]) * y0[0,:],
            tf.zeros([1, y0.shape[1]], dtype=tf.float32),
            tf.ones([1, y0.shape[1]]) * y0[1,:]
        ], axis=0))  # shape (100,3)
        
        pY1 = tf.transpose(tf.concat([
            tf.ones([1, y0.shape[1]]) * y0[0,:],
            tf.ones([1, y0.shape[1]], dtype=tf.float32),
            tf.ones([1, y0.shape[1]]) * y0[1,:]
        ], axis=0))  # shape (100,3)
        
        pX0 = tf.transpose(tf.concat([
            tf.zeros([1, x0.shape[1]], dtype=tf.float32),
            tf.ones([1, x0.shape[1]]) * x0[0,:],
            tf.ones([1, x0.shape[1]]) * x0[1,:]
        ], axis=0))  # shape (100,3)
        
        pX1 = tf.transpose(tf.concat([
            tf.ones([1, x0.shape[1]], dtype=tf.float32),
            tf.ones([1, x0.shape[1]]) * x0[0,:],
            tf.ones([1, x0.shape[1]]) * x0[1,:]
        ], axis=0))  # shape (100,3)
        
        # Initial condition input: concatenate time samples with zero as spatial dimension
        pIC = tf.concat([t0, tf.zeros([t0.shape[0],1], dtype=tf.float32)], axis=1)  # shape (100,3)
        
        # Domain points for PDE residual evaluation
        pF = tf.random.uniform(shape=[1000,3], maxval=1, dtype=tf.float32)
        
        # Model predictions at boundary and initial condition points
        uX0 = model(pX0, training=True)  # shape (100,1)
        uX1 = model(pX1, training=True)
        uY0 = model(pY0, training=True)
        uY1 = model(pY1, training=True)
        uIC = model(pIC, training=True)
        uF = model(pF, training=True)
        
        # Compute derivatives of uF w.r.t inputs to define PDE residuals
        
        # For gradient calculations, we watch pF variables for autodiff
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(pF)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(pF)
                u = model(pF, training=True)  # shape (1000,1)
            u_grad = tape1.gradient(u, pF)  # du/dx, du/dy, du/dt shape (1000,3)
        
        # Extract derivatives
        u_x = u_grad[:,0]
        u_y = u_grad[:,1]
        u_t = u_grad[:,2]
        
        # Second derivatives needed for Laplacian u_xx and u_yy
        u_xx = tape2.gradient(u_x, pF)[:,0]
        u_yy = tape2.gradient(u_y, pF)[:,1]
        
        # PDE residual: u_t - (u_xx + u_yy)
        f = u_t - (u_xx + u_yy)  # shape (1000,)
        
        # Initial condition function value (analytical)
        # sin(pi t) * sin(pi x), but since last dim may be zero for initial condition spatial,
        # the original formula was sin(pi*pIC[:,0]) * sin(pi*pIC[:,1])
        # pIC columns: t, ?, so we assume pIC[:,0] = t and pIC[:,1] = 0 here (spatial coord)
        ic_true = tf.math.sin(pi * pIC[:,0]) * tf.math.sin(pi * pIC[:,1])
        
        # Compute all components of the loss - mean squared errors
        loss_bcs = tf.reduce_mean(tf.square(uX0)) + tf.reduce_mean(tf.square(uX1)) + \
                   tf.reduce_mean(tf.square(uY0)) + tf.reduce_mean(tf.square(uY1))
        
        loss_ic = tf.reduce_mean(tf.square(uIC[:,0] - ic_true))  # uIC shape (100,1) so indexing first col
        
        loss_pde = tf.reduce_mean(tf.square(f))
        
        return loss_bcs + loss_ic + loss_pde
    return loss

def my_model_function():
    # Create and compile the model instance with the custom loss
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=custom_loss(model),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Returns a random input tensor of shape (batch_size, 3)
    # Using batch size 32 as a reasonable default
    batch_size = 32
    # The input values are floats in [0,1) as per the original random.uniform distributions
    return tf.random.uniform(shape=(batch_size, 3), dtype=tf.float32)

