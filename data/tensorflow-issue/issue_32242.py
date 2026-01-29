# tf.random.uniform(()) ← The input shape is a scalar as in the example, model deals with scalars and lists containing Tensors, Variables and Python objects.

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model mimics the behavior of the example tf.Module 'A' given in the issue:
    - It contains a scalar float (python float),
    - a tf.Variable,
    - a tf.Tensor constant,
    - and a list containing a tf.Tensor, a python float, and a tf.Variable.
    
    Since tf.train.Checkpoint does not save/restore python objects or constant tensors,
    only tf.Variable are tracked and saved/restored by checkpoint.
    
    This model exposes these attributes and thus can be saved/restored using tf.train.Checkpoint,
    but only Variables values actually persist.
    """
    def __init__(self):
        super().__init__()
        # Python scalar float - not tracked by checkpoint
        self.scalar = 1.0
        # Variable - tracked and saved/restored by checkpoint
        self.variable = tf.Variable(1.0)
        # Constant tensor - not tracked by checkpoint
        self.tensor = tf.constant(1.0)
        # List containing tf.Tensor, python float, and Variable
        # Note: Variables inside lists are not tracked by the checkpoint.
        self.list = [tf.constant(10.0), 20.0, tf.Variable(0.0)]

    def __str__(self):
        # Display current values similar to the issue printout
        # For the list, represent each element clearly
        # Variables and tensors show their numpy values
        list_repr = []
        for elem in self.list:
            if isinstance(elem, tf.Variable):
                list_repr.append(f"Variable({elem.numpy()})")
            elif isinstance(elem, tf.Tensor):
                list_repr.append(f"Tensor({elem.numpy()})")
            else:
                list_repr.append(repr(elem))
        list_str = "[" + ", ".join(list_repr) + "]"
        return (f"scalar={self.scalar}, variable={self.variable.numpy()}, tensor={self.tensor.numpy()}, list={list_str}")

def my_model_function():
    # Instantiate and return the MyModel object
    return MyModel()

def GetInput():
    """
    Returns a dummy input for the model.
    The model does not accept external input for forward pass in the original issue,
    so we return a dummy scalar tensor input for compatibility.
    
    This matches the minimal shape as the variables are scalars.
    """
    return tf.random.uniform(())  # scalar input (shape ())


# The model is not designed to operate on input tensors through __call__ or call,
# but benchmarks or checkpointing happens on the variables inside the model.
# So the forward method is left out because the original example doesn't have it.

