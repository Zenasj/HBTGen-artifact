# tf.random.normal([B, N, H, W, C], dtype=tf.float32) ← inferred input shape from the issue example and test code

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, nframes):
        super(MyModel, self).__init__()
        self.nframes = nframes
        self.center = nframes // 2
        # Attention layers as per the sample code
        self.tAtt_1 = tf.keras.layers.Conv2D(4, (3, 3), (1, 1), padding="same")
        self.tAtt_2 = tf.keras.layers.Conv2D(4, (3, 3), (1, 1), padding="same")

    def call(self, aligned_fea):
        """
        aligned_fea: Tensor of shape [B, N, H, W, C]
        Returns stacked tensor of shape [N, B, H, W] corresponding to attention correlations.
        
        This method illustrates the TensorArray size bug when passing a tf.Tensor size.
        The workaround is to rely on python int `self.nframes` rather than dynamic tensor N.
        """
        aligned_fea_shape = tf.shape(aligned_fea)  # dynamic shape: [B,N,H,W,C]
        B = aligned_fea_shape[0]
        N = aligned_fea_shape[1]
        H = aligned_fea_shape[2]
        W = aligned_fea_shape[3]
        C = aligned_fea_shape[4]

        # Reference embedding for center frame
        emb_ref = self.tAtt_1(aligned_fea[:, self.center, :, :, :])  # [B, H, W, 4]

        emb = tf.reshape(aligned_fea, [-1, H, W, C])  # [B*N, H, W, C]
        emb = self.tAtt_2(emb)                         # [B*N, H, W, 4]
        emb = tf.reshape(emb, [B, N, H, W, -1])       # [B, N, H, W, 4]

        # Use python int nframes for TensorArray size to avoid None shape bug
        cor_l = tf.TensorArray(dtype=tf.float32, size=self.nframes)  # IMPORTANT: size as python int!

        def cond(i, nframes, input_emb, arr):
            return tf.less(i, nframes)

        def body(i, nframes, input_emb, arr):
            emb_nbr = input_emb[:, i, :, :, :]  # [B, H, W, channels]
            cor_tmp = tf.reduce_sum(emb_nbr * emb_ref, axis=3)  # [B, H, W] correlation per spatial location
            arr = arr.write(i, cor_tmp)  # Write correlation for frame i
            i = tf.add(i, 1)
            return i, nframes, input_emb, arr

        i0 = tf.constant(0)
        i_final, _, _, cor_l = tf.while_loop(cond, body, [i0, self.nframes, emb, cor_l])
        # cor_l stacked shape will be [N, B, H, W], with N=self.nframes known statically

        cor_l_stacked = cor_l.stack()  # [N, B, H, W]

        # Return stacked tensor directly; could transpose if needed.
        return cor_l_stacked


def my_model_function():
    # Instantiate MyModel with reasonable fixed number of frames (e.g., 5)
    return MyModel(nframes=5)


def GetInput():
    # Return a random input tensor matching shape [B,N,H,W,C], compatible with model call
    # Use example shape based on issue: B=4, N=5, H=32, W=32, C=3
    B = 4
    N = 5
    H = 32
    W = 32
    C = 3
    input_tensor = tf.random.normal([B, N, H, W, C], dtype=tf.float32)
    return input_tensor

