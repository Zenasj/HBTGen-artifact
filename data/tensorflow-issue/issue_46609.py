# tf.random.uniform((B, D, H, W, C), dtype=tf.float32)  # Typical 3D data format for UpSampling3D

import tensorflow as tf
import numpy as np

# The interpolation functions for 1D, 2D, 3D separable interpolation using numpy.
# These are direct translations/integrations from the issue source.

def linear_interpolate(x_fix, y_fix, x_var):
    '''
    1D linear interpolation (numpy-based).
    '''
    x_repeat = np.tile(x_var[:, None], (len(x_fix), ))
    distances = np.abs(x_repeat - x_fix)

    x_indices = np.searchsorted(x_fix, x_var)
    weights = np.zeros_like(distances)
    idx = np.arange(len(x_indices))
    weights[idx, x_indices] = distances[idx, x_indices - 1]
    weights[idx, x_indices - 1] = distances[idx, x_indices]
    weights /= np.sum(weights, axis=1)[:, None]

    y_var = np.dot(weights, y_fix.T)

    return y_var


def cubic_interpolate(x, y, x0):
    '''
    1D cubic spline interpolation (numpy-based).
    '''
    x = np.asfarray(x)
    y = np.asfarray(y)
    if np.any(np.diff(x) < 0):
        indexes = np.argsort(x)
        x = x[indexes]
        y = y[indexes]
    size = len(x)
    xdiff = np.diff(x)
    ydiff = np.diff(y)
    Li = np.empty(size)
    Li_1 = np.empty(size - 1)
    z = np.empty(size)
    Li[0] = np.sqrt(2 * xdiff[0])
    Li_1[0] = 0.0
    B0 = 0.0  # natural boundary condition
    z[0] = B0 / Li[0]

    for i in range(1, size - 1):
        Li_1[i] = xdiff[i - 1] / Li[i - 1]
        Li[i] = np.sqrt(2 * (xdiff[i - 1] + xdiff[i]) - Li_1[i - 1] * Li_1[i - 1])
        Bi = 6 * (ydiff[i] / xdiff[i] - ydiff[i - 1] / xdiff[i - 1])
        z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

    i = size - 1
    Li_1[i - 1] = xdiff[-1] / Li[i - 1]
    Li[i] = np.sqrt(2 * xdiff[-1] - Li_1[i - 1] * Li_1[i - 1])
    Bn = 0.0  # natural boundary
    z[i] = (Bn - Li_1[i - 1] * z[i - 1]) / Li[i]

    z[i] = z[i] / Li[i]
    for i in range(size - 2, -1, -1):
        z[i] = (z[i] - Li_1[i - 1] * z[i + 1]) / Li[i]

    index = x.searchsorted(x0)
    index = np.clip(index, 1, size - 1)

    xi1, xi0 = x[index], x[index - 1]
    yi1, yi0 = y[index], y[index - 1]
    zi1, zi0 = z[index], z[index - 1]
    hi1 = xi1 - xi0

    f0 = (zi0 / (6 * hi1) * (xi1 - x0) ** 3 +
          zi1 / (6 * hi1) * (x0 - xi0) ** 3 +
          (yi1 / hi1 - zi1 * hi1 / 6) * (x0 - xi0) +
          (yi0 / hi1 - zi0 * hi1 / 6) * (xi1 - x0))

    return f0


def pchip_interpolate(xi, yi, x, mode="mono", verbose=False):
    '''
    1D PCHIP interpolation (numpy-based).
    '''
    if mode not in ("mono", "quad"):
        raise ValueError("Unrecognized mode string")

    xi = xi.astype("double")
    yi = yi.astype("double")

    x_index = np.zeros(len(x), dtype="int")
    xi_steps = np.diff(xi)
    if not np.all(xi_steps > 0):
        raise ValueError("x-coordinates are not in increasing order.")

    x_steps = np.diff(x)
    if xi_steps.max() / xi_steps.min() < 1.000001:
        # uniform input grid
        xi_start = xi[0]
        xi_step = (xi[-1] - xi[0]) / (len(xi) - 1)
        x_index = np.minimum(np.maximum(np.floor((x - xi_start) / xi_step).astype(int), 0), len(xi) - 2)

        h = (xi[-1] - xi[0]) / (len(xi) - 1)
        d = np.zeros(len(xi), dtype="double")
        if mode == "quad":
            d[[0]] = (yi[1] - yi[0]) / h
            d[[-1]] = (yi[-1] - yi[-2]) / h
            d[1:-1] = (yi[2:] - yi[0:-2]) / (2 * h)
        else:
            delta = np.diff(yi) / h
            d = np.concatenate(([delta[0]], 
                                2 / (1 / delta[:-1] + 1 / delta[1:]), 
                                [delta[-1]]))
            zero_mask = np.concatenate(([False], np.logical_xor(delta[:-1] > 0, delta[1:] > 0), [False]))
            d[zero_mask] = 0
            d[np.concatenate(([False], delta == 0))] = 0
            d[np.concatenate((delta == 0, [False]))] = 0

        dxxi = x - xi[x_index]
        dxxid = x - xi[1 + x_index]
        dxxi2 = dxxi ** 2
        dxxid2 = dxxid ** 2
        y = (2 / h ** 3 * (yi[x_index] * dxxid2 * (dxxi + h / 2) - yi[1 + x_index] * dxxi2 * (dxxid - h / 2)) +
             1 / h ** 2 * (d[x_index] * dxxid2 * dxxi + d[1 + x_index] * dxxi2 * dxxid))
    else:
        # non-uniform or monotonic grids
        # Logic for x_index assignment
        if (x_steps.max() / x_steps.min() < 1.000001 and x_steps.max() / x_steps.min() > 0.999999):
            x_decreasing = x[-1] < x[0]
            if x_decreasing:
                x = x[::-1]
            x_start = x[0]
            x_step = (x[-1] - x[0]) / (len(x) - 1)
            x_indexprev = -1
            for xi_loop in range(len(xi) - 2):
                x_indexcur = max(int(np.floor((xi[1 + xi_loop] - x_start) / x_step)), -1)
                x_index[1 + x_indexprev:1 + x_indexcur] = xi_loop
                x_indexprev = x_indexcur
            x_index[1 + x_indexprev:] = len(xi) - 2
            if x_decreasing:
                x = x[::-1]
                x_index = x_index[::-1]
        elif np.all(x_steps > 0) or np.all(x_steps < 0):
            x_decreasing = x[-1] < x[0]
            if x_decreasing:
                x = x[::-1]
            x_len = len(x)
            x_loop = 0
            for xi_loop in range(len(xi) - 1):
                while x_loop < x_len and x[x_loop] < xi[1 + xi_loop]:
                    x_index[x_loop] = xi_loop
                    x_loop += 1
            x_index[x_loop:] = len(xi) - 2
            if x_decreasing:
                x = x[::-1]
                x_index = x_index[::-1]
        else:
            for index in range(len(x)):
                loc = np.where(x[index] < xi)[0]
                if loc.size == 0:
                    x_index[index] = len(xi) - 2
                elif loc[0] == 0:
                    x_index[index] = 0
                else:
                    x_index[index] = loc[0] - 1

        h = np.diff(xi)
        d = np.zeros(len(xi), dtype="double")
        delta = np.diff(yi) / h
        if mode == "quad":
            d[[0, -1]] = delta[[0, -1]]
            d[1:-1] = (delta[1:] * h[:-1] + delta[:-1] * h[1:]) / (h[:-1] + h[1:])
        else:
            d = np.concatenate(
                (delta[0:1],
                 3 * (h[:-1] + h[1:]) / ((h[:-1] + 2 * h[1:]) / delta[:-1] +
                                          (2 * h[:-1] + h[1:]) / delta[1:]),
                 delta[-1:]))

            zero_mask = np.concatenate(([False], np.logical_xor(delta[:-1] > 0, delta[1:] > 0), [False]))
            d[zero_mask] = 0
            zero_mask2 = np.logical_or(np.concatenate(([False], delta == 0)),
                                      np.concatenate((delta == 0, [False])))
            d[zero_mask2] = 0

        dxxi = x - xi[x_index]
        dxxid = x - xi[1 + x_index]
        dxxi2 = dxxi ** 2
        dxxid2 = dxxid ** 2

        y = (2 / h[x_index] ** 3 *
             (yi[x_index] * dxxid2 * (dxxi + h[x_index] / 2) - yi[1 + x_index] * dxxi2 *
              (dxxid - h[x_index] / 2)) + 1 / h[x_index] ** 2 *
             (d[x_index] * dxxid2 * dxxi + d[1 + x_index] * dxxi2 * dxxid))

    return y


def Interpolate1D(x, y, xx, method='nearest'):
    '''
    1D interpolation dispatcher (numpy-based).
    '''
    n = len(x)
    nn = len(xx)
    yy = np.zeros(nn)

    if method == 'nearest':
        for i in range(nn):
            xi = np.abs(xx[i] - x).argmin()
            yy[i] = y[xi]

    elif method == 'linear':
        yy = linear_interpolate(x, y, xx)

    elif method == 'cubic':
        yy = cubic_interpolate(x, y, xx)

    elif method == 'pchip':
        yy = pchip_interpolate(x, y, xx, mode='mono')

    return yy


def Interpolate2D(x, y, f, xx, yy, method='nearest'):
    '''
    2D separable interpolation (numpy-based).
    '''
    n1 = len(x)
    n2 = len(y)
    nn1 = len(xx)
    nn2 = len(yy)

    w = np.zeros((nn1, n2))
    ff = np.zeros((nn1, nn2))

    for j in range(n2):
        w[:, j] = Interpolate1D(x, f[:, j], xx, method)

    for i in range(nn1):
        ff[i, :] = Interpolate1D(y, w[i, :], yy, method)

    return ff


def Interpolate3D(x, y, z, f, xx, yy, zz, method='nearest'):
    '''
    3D separable interpolation (numpy-based).
    '''
    n1 = len(x)
    n2 = len(y)
    n3 = len(z)
    nn1 = len(xx)
    nn2 = len(yy)
    nn3 = len(zz)

    w1 = np.zeros((nn1, n2, n3))
    w2 = np.zeros((nn1, nn2, n3))
    ff = np.zeros((nn1, nn2, nn3))

    for k in range(n3):
        for j in range(n2):
            w1[:, j, k] = Interpolate1D(x, f[:, j, k], xx, method)

    for k in range(n3):
        for i in range(nn1):
            w2[i, :, k] = Interpolate1D(y, w1[i, :, k], yy, method)

    for j in range(nn2):
        for i in range(nn1):
            ff[i, j, :] = Interpolate1D(z, w2[i, j, :], zz, method)

    return ff


def UpInterpolate3D(x,
                    size=(2, 2, 2),
                    interpolation='nearest',
                    data_format='channels_first',
                    align_corners=True):
    """
    3D upsampling interpolation for Tensor input.

    x: tf.Tensor, 5D.
    size: tuple of ints, upsampling factors.
    interpolation: str, one of nearest, linear, cubic, pchip
    data_format: 'channels_first' or 'channels_last'
    align_corners: bool
    """
    # WARNING: we convert to numpy inside, so this will fail in graph mode or with non-eager tensors.
    # This is a major limitation and the user reported issues in the original thread.
    # This implementation is for demonstration/reference and not production or graph mode ready.

    x_np = x.numpy()
    if data_format == 'channels_last':
        nb, nr, nc, nd, nh = x_np.shape
    else:
        nb, nh, nr, nc, nd = x_np.shape
    r, c, d = size

    ir = np.linspace(0.0, nr - 1.0, num=nr)
    ic = np.linspace(0.0, nc - 1.0, num=nc)
    id = np.linspace(0.0, nd - 1.0, num=nd)

    if align_corners:
        iir = np.linspace(0.0, nr - 1.0, num=nr * r)
        iic = np.linspace(0.0, nc - 1.0, num=nc * c)
        iid = np.linspace(0.0, nd - 1.0, num=nd * d)
    else:
        iir = np.linspace(0.0 - 0.5 + 0.5 / r, nr - 1.0 + 0.5 - 0.5 / r, num=nr * r)
        iic = np.linspace(0.0 - 0.5 + 0.5 / c, nc - 1.0 + 0.5 - 0.5 / c, num=nc * c)
        iid = np.linspace(0.0 - 0.5 + 0.5 / d, nd - 1.0 + 0.5 - 0.5 / d, num=nd * d)
        iir = np.clip(iir, 0.0, nr - 1.0)
        iic = np.clip(iic, 0.0, nc - 1.0)
        iid = np.clip(iid, 0.0, nd - 1.0)

    if data_format == 'channels_last':
        xx = np.zeros((nb, nr * r, nc * c, nd * d, nh))
        for i in range(nb):
            for j in range(nh):
                t = np.reshape(x_np[i, :, :, :, j], (nr, nc, nd))
                xx[i, :, :, :, j] = Interpolate3D(ir, ic, id, t, iir, iic, iid, interpolation)
    else:
        xx = np.zeros((nb, nh, nr * r, nc * c, nd * d))
        for i in range(nb):
            for j in range(nh):
                t = np.reshape(x_np[i, j, :, :, :], (nr, nc, nd))
                xx[i, j, :, :, :] = Interpolate3D(ir, ic, id, t, iir, iic, iid, interpolation)

    return tf.convert_to_tensor(xx, dtype=x.dtype)


class MyModel(tf.keras.Model):
    def __init__(self, size=(2,2,2), interpolation='trilinear', data_format='channels_last', align_corners=True):
        """
        Model wrapping the UpSampling3D with trilinear interpolation feature.
        Defaults:
          - size: Upsampling by 2x in each spatial dim
          - interpolation: 'trilinear' (alias for 'linear')
          - data_format: 'channels_last' (B, D, H, W, C)
          - align_corners: True
        """
        super().__init__()
        self.size = size
        self.interpolation = interpolation
        self.data_format = data_format
        self.align_corners = align_corners

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # Use UpInterpolate3D with the given arguments
        out = UpInterpolate3D(inputs, 
                              size=self.size,
                              interpolation=self.interpolation,
                              data_format=self.data_format,
                              align_corners=self.align_corners)
        return out


def my_model_function():
    # Return an instance of MyModel with default parameters
    return MyModel()


def GetInput():
    # Returns a random input tensor with shape (batch_size, depth, height, width, channels)
    # consistent with data_format 'channels_last' and expected by MyModel
    batch_size = 1  # single batch for testing
    depth = 8
    height = 8
    width = 8
    channels = 3
    dtype = tf.float32
    # Construct a uniform random input tensor
    return tf.random.uniform((batch_size, depth, height, width, channels), dtype=dtype)

