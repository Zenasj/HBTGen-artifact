# tf.random.uniform((B,)) ← Input is a 1D tensor of any batch size (vector), dtype float32 by default
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, siz):
        super(MyModel, self).__init__()
        self.siz = siz
        self.buildFoo()

    def call(self, in_data):
        # Compute Foo0 = in_data * FooTns0 (scalar multiplication)
        Foo0 = tf.multiply(in_data, self.FooTns0)
        FooList = [Foo0]
        # Iteratively multiply by each FooTns[i]
        for i in range(self.siz):
            tmp = tf.multiply(FooList[i], self.FooTns[i])
            FooList.append(tmp)
        # Return the last element representing the "power" of multiplications
        return FooList[self.siz]

    def buildFoo(self):
        # Use add_weight instead of raw tf.Variable to ensure proper tracking by Keras
        # Initialize the scalar variables with float32 dtype for proper multiplication with input tensor
        self.FooTns0 = self.add_weight(
            name="TNS0",
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True
        )
        self.FooTns = []
        for i in range(self.siz):
            # Initialize these weights with float values equal to their index (0,1,2,...)
            var = self.add_weight(
                name="TNS"+str(i+1),
                shape=(),
                initializer=tf.keras.initializers.Constant(float(i)),
                trainable=True
            )
            self.FooTns.append(var)

def my_model_function():
    # Create a MyModel instance with siz=5 (matches example from issue)
    return MyModel(siz=5)

def GetInput():
    # Generate a random input tensor matching the expected input shape (1D tensor)
    # Since the model multiplies elementwise, shape can be (batch_size,)
    # We use batch size 4 for demonstration, dtype float32 as default
    return tf.random.uniform(shape=(4,), dtype=tf.float32)

