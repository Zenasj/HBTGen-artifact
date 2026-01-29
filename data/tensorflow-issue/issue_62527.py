# tf.random.uniform((5, 1), dtype=tf.float16), tf.random.uniform((5, 16), dtype=tf.float16)
import tensorflow as tf

class Model1(tf.keras.Model):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def call(self, inp, inp2):
        # Concatenate inputs along axis 1: shapes inp (5,1), inp2 (5,16) -> concat (5,17)
        concat = tf.concat([inp, inp2], axis=1)
        # Slice from index -17 to 17 exclusive, step 4 along columns; effectively select columns [0,4,8,12,16]
        sliced = concat[:, -17:17:4]
        matmul = tf.matmul(sliced, sliced)
        return matmul

class Model2(tf.keras.Model):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def call(self, inp, inp2):
        concat = tf.concat([inp, inp2], axis=1)
        transposed = tf.transpose(concat, perm=[1, 0])
        sliced = concat[:, -17:17:4]
        matmul = tf.matmul(sliced, sliced)
        return matmul, transposed

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model1 = Model1()
        self.model2 = Model2()

    @tf.function(jit_compile=True)
    def call(self, inp, inp2):
        # Run both models:
        matmul1 = self.model1(inp, inp2)  # shape (5,5), dtype float16
        matmul2, transposed2 = self.model2(inp, inp2)  # matmul2 same shape, transposed2 shape (17,5)

        # Compute elementwise difference norm between matmul results to check consistency
        diff = tf.abs(matmul1 - matmul2)

        # Produce a boolean tensor of matching within tolerance (rtol=0.001, atol=0.001)
        # Since input and output types are float16, use float32 for stable numeric comparisons
        matmul1_f32 = tf.cast(matmul1, tf.float32)
        matmul2_f32 = tf.cast(matmul2, tf.float32)
        is_close = tf.experimental.numpy.isclose(matmul1_f32, matmul2_f32, rtol=0.001, atol=0.001)

        # Combine results in output dict:
        # 'matmul1', 'matmul2', 'transposed', 'is_close' for external verification
        return {
            'matmul1': matmul1,
            'matmul2': matmul2,
            'transposed': transposed2,
            'is_close': is_close
        }

def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Return a tuple of two inputs consistent with the original:
    # inp: shape (5,1), dtype float16
    # inp2: shape (5,16), dtype float16
    inp = tf.random.uniform(shape=[5, 1], dtype=tf.float16)
    inp2 = tf.random.uniform(shape=[5, 16], dtype=tf.float16)
    return inp, inp2

