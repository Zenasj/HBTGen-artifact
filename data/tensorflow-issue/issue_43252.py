# tf.random.uniform((N, 1), dtype=tf.float64) for x, z, t inputs 
# tf.random.uniform((N, 3), dtype=tf.float64) for c input
# tf.random.uniform((1, N), dtype=tf.float64) for v1, v2, v3, v4

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Input signature according to the usage in the issue
        self.input_signature = [tf.TensorSpec([None, 1], tf.float64)] * 3 + \
                               [tf.TensorSpec([None, 3], tf.float64)] + \
                               [tf.TensorSpec([1, None], tf.float64)] * 4
        
    @tf.function(input_signature=[tf.TensorSpec([None, 1], tf.float64)] * 3 + 
                                  [tf.TensorSpec([None, 3], tf.float64)] + 
                                  [tf.TensorSpec([1, None], tf.float64)] * 4)
    def call(self, x, z, t, c, v1, v2, v3, v4):
        """
        Computes a set of forward functions and their first and second order directional derivatives 
        using tf.autodiff.ForwardAccumulator.
        
        Inputs:
          - x, z, t: Each is a tensor with shape [batch_size, 1], dtype float64
          - c: [batch_size, 3], float64
          - v1, v2, v3, v4: each [1, batch_size], float64
      
        Returns:
          Tuple of tensors containing:
          (dudx, dudz, dudt, dwdx, dwdz, dwdt,
           dbdx, dbdz, dbdt, dpdx, dpdz,
           d2ud2x, d2ud2z,
           d2wd2x, d2wd2z,
           d2bd2x, d2bd2z)
        """
        # Inner function to compute basic values p,b,u,w
        def ff(xi, zi, ti):
            p = tf.concat([xi, zi, ti], axis=1)  # shape: [batch_size, 3]
            pe = tf.transpose(p[:, :, None], perm=[0, 2, 1])  # shape: [batch, 1, 3]
            ce = tf.transpose(c[:, :, None], perm=[2, 0, 1])  # [3, batch, 1]
            d = ce - pe  # broadcast subtraction, shape: [3, batch, 3] broadcasted ?
            # Actually ce.shape is [3, batch, 1] and pe [batch, 1, 3],
            # Let's carefully check dimensions and broadcasting:
            # ce : [3, batch, 1]
            # pe : [batch, 1, 3]
            # Their difference ce - pe needs to align dims.
            # But the original code just does tf.square(ce - pe) then reduce sum on axis=2.
            # To replicate original logic: r = sum(square(ce - pe), axis=2)
            # We must make ce and pe broadcastable along axis=2.
            #
            # Let's permute ce to [batch, 3, 1] to match shape with pe [batch, 1, 3]
            ce_t = tf.transpose(c, perm=[0, 2, 1])  # [batch, 3, 1]
            d = ce_t - pe  # shape: [batch, 3, 3]
            r = tf.reduce_sum(tf.square(d), axis=2)  # shape: [batch, 3]
            G = tf.exp(-r / 2)  # shape: [batch, 3]
            
            # Compute weighted sums with v1,v2,v3,v4 of shape [1, batch] to apply elementwise with G
            # transpose v_i to shape [batch, 1] for broadcasting
            v1t = tf.transpose(v1)  # [batch, 1]
            v2t = tf.transpose(v2)
            v3t = tf.transpose(v3)
            v4t = tf.transpose(v4)
            
            # G * vi is [batch, 3] * [batch,1], need broadcasting:
            # Actually G is [batch, 3], v_i t is [batch, 1]
            # Broadcasting works along last axis
            # But original code uses reduce_sum(G * v_i, axis=1, keepdims=True)
            # G * v_i multiplies [batch, 3] * [1, batch] original shapes?
            # The original code's v_i are [1, batch], and G is [batch, 3].
            # Multiplication was done as G * v_i which is not shape-compatible (batch,3) * (1,batch)
            # Possibly original code exploits broadcasting wrongly or v_i is understood differently.
            # Since v_i shape is [1, batch], its transpose is [batch,1], so multiply (batch, 3) * (batch,1)
            # will broadcast along axis=1. result is (batch, 3).
            # Summing over axis=1 sums over 3 elements per batch.
            #
            # Ok, let's do element-wise broadcast multiply:
            p = tf.reduce_sum(G * v1t, axis=1, keepdims=True)  # [batch, 1]
            b = tf.reduce_sum(G * v2t, axis=1, keepdims=True)
            u = tf.reduce_sum(G * v3t, axis=1, keepdims=True)
            w = tf.reduce_sum(G * v4t, axis=1, keepdims=True)
            return p, b, u, w
        
        # Helper to compute JVP (Jacobian-vector product) for a function with given primals and tangents
        def _jvp(f, primals, tangents):
            with tf.autodiff.ForwardAccumulator(primals, tangents) as acc:
                primals_out = f(*primals)
            jvp_out = acc.jvp(primals_out, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            return primals_out, jvp_out
        
        primals = [x, z, t]
        tangent_mask = [tf.zeros_like(p) for p in primals]

        # For x partial derivatives and second partials
        with tf.autodiff.ForwardAccumulator(primals=[x], tangents=[tf.ones_like(x)]) as fwd_outer_x:
            i = 0
            # enable tangent only for i-th primal, zeros elsewhere
            inner_tangents = tangent_mask[:i] + [tf.ones_like(primals[i])] + tangent_mask[i + 1:]
            [dpdx, dbdx, dudx, dwdx] = _jvp(lambda a, b, c: ff(a, b, c), primals, inner_tangents)[1]
        [d2bd2x, d2ud2x, d2wd2x] = fwd_outer_x.jvp([dbdx, dudx, dwdx], tf.UnconnectedGradients.ZERO)

        # For z partial derivatives and second partials
        with tf.autodiff.ForwardAccumulator(primals=[z], tangents=[tf.ones_like(z)]) as fwd_outer_z:
            i = 1
            inner_tangents = tangent_mask[:i] + [tf.ones_like(primals[i])] + tangent_mask[i + 1:]
            [dpdz, dbdz, dudz, dwdz] = _jvp(lambda a, b, c: ff(a, b, c), primals, inner_tangents)[1]
        [d2bd2z, d2ud2z, d2wd2z] = fwd_outer_z.jvp([dbdz, dudz, dwdz], tf.UnconnectedGradients.ZERO)

        # For t partial derivatives (no second partials computed for t in original code)
        i = 2
        inner_tangents = tangent_mask[:i] + [tf.ones_like(primals[i])] + tangent_mask[i + 1:]
        _, [dpdt, dbdt, dudt, dwdt] = _jvp(lambda a, b, c: ff(a, b, c), primals, inner_tangents)

        # Returning values in the order expected from original Issue_fwd.f
        return (dudx, dudz, dudt,
                dwdx, dwdz, dwdt,
                dbdx, dbdz, dbdt,
                dpdx, dpdz,
                d2ud2x, d2ud2z,
                d2wd2x, d2wd2z,
                d2bd2x, d2bd2z)

def my_model_function():
    return MyModel()

def GetInput():
    # Create example inputs matching input signature with batch size 10
    n = 10
    x = tf.random.uniform((n, 1), dtype=tf.float64)
    z = tf.random.uniform((n, 1), dtype=tf.float64)
    t = tf.random.uniform((n, 1), dtype=tf.float64)
    c = tf.random.uniform((n, 3), dtype=tf.float64)
    v1 = tf.random.uniform((1, n), dtype=tf.float64)
    v2 = tf.random.uniform((1, n), dtype=tf.float64)
    v3 = tf.random.uniform((1, n), dtype=tf.float64)
    v4 = tf.random.uniform((1, n), dtype=tf.float64)
    return (x, z, t, c, v1, v2, v3, v4)

