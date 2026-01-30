from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

tf.config.run_functions_eagerly(True)


def build_model_fail():
    input = tf.keras.Input(dtype=tf.int32, shape=(), batch_size=1)
    output = tf.keras.layers.Lambda(lambda_fn)(input)
    return tf.keras.Model(inputs=input, outputs=output)


def build_model_success():
    input = tf.keras.Input(dtype=tf.int32, shape=(), batch_size=1)

    # temporarily setting off the eager execution
    # allows the lambda layer to infer the output spec.
    tf.config.run_functions_eagerly(False)
    output = tf.keras.layers.Lambda(lambda_fn)(input)

    # switching back to eager for runtime debugging
    tf.config.run_functions_eagerly(True)

    return tf.keras.Model(inputs=input, outputs=output)


@tf.function
def lambda_fn(input):
    i = tf.constant(0, dtype=tf.int32)
    while i < input:
        tf.print("loop iteration", i)
        i = i + 1
    return input


if __name__ == "__main__":

    # this works
    model = build_model_success()
    model(5)

    # this doesn't work
    model = build_model_fail()
    model(5)

def outer_factory():

    def inner_factory(ag__):

        def tf__lambda_fn(input):
            with ag__.FunctionScope('lambda_fn', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                i = ag__.converted_call(ag__.ld(tf).constant, (0,), dict(dtype=ag__.ld(tf).int32), fscope)

                def get_state():
                    return (i,)

                def set_state(vars_):
                    nonlocal i
                    (i,) = vars_

                def loop_body():
                    nonlocal i
                    ag__.ld(print)('eager iteration', ag__.ld(i))
                    ag__.converted_call(ag__.ld(tf).print, ('loop iteration', ag__.ld(i)), None, fscope)
                    i = (ag__.ld(i) + 1)

                def loop_test():
                    return (ag__.ld(i) < ag__.ld(input))
                ag__.while_stmt(loop_test, loop_body, get_state, set_state, ('i',), {})
                try:
                    do_return = True
                    retval_ = ag__.ld(input)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__lambda_fn
    return inner_factory

import tensorflow as tf
from tensorflow.python.framework import func_graph
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

tf.config.run_functions_eagerly(True)


def build_model():
    input = tf.keras.Input(dtype=tf.int32, shape=(), batch_size=1)
    output_shape = infer_output(lambda_fn, input)
    output = tf.keras.layers.Lambda(lambda_fn,
                                    output_shape=output_shape,
                                    dynamic=True)(input)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model


def infer_output(func, input: KerasTensor):
    scratch_graph = func_graph.FuncGraph(__name__ + '_scratch_graph')
    _func = tf.autograph.to_graph(func.python_function)
    with scratch_graph.as_default():
        output = _func(input._to_placeholder())
    return output.shape


@tf.function
def lambda_fn(input):
    i = tf.constant(0, dtype=tf.int32)
    while i < input:
        print("eager iteration", i)
        tf.print("loop iteration", i)
        i = i + 1
    return input


if __name__ == "__main__":
    model = build_model()
    model(5)

import tensorflow as tf

tf.config.run_functions_eagerly(True)


def _infer_output_signature(self, inputs, args, kwargs, input_masks):
    function_fn = self.function
    self.function = tf.autograph.to_graph(self.function.python_function)
    output = super(type(self),
                   self)._infer_output_signature(inputs, args, kwargs,
                                                 input_masks)
    self.function = function_fn
    return output


tf.keras.layers.Lambda._infer_output_signature = _infer_output_signature


def build_model():
    input = tf.keras.Input(dtype=tf.int32, shape=(), batch_size=1)
    output = tf.keras.layers.Lambda(lambda_fn)(input)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model


@tf.function
def lambda_fn(input):
    i = tf.constant(0, dtype=tf.int32)
    while i < input:
        print("eager iteration", i)
        tf.print("loop iteration", i)
        i = i + 1
    return input


if __name__ == "__main__":
    model = build_model()
    model(5)

import tensorflow as tf
print(tf.version.GIT_VERSION, tf.version.VERSION, flush=True)
print(tf.config.list_physical_devices(), flush=True)


tf.config.run_functions_eagerly(True)


def build_model_fail():
    input = tf.keras.Input(dtype=tf.int32, shape=(), batch_size=1)
    output = tf.compat.v1.keras.layers.Lambda(lambda_fn)(input)
    return tf.keras.Model(inputs=input, outputs=output)


def build_model_success():
    input = tf.keras.Input(dtype=tf.int32, shape=(), batch_size=1)

    # temporarily setting off the eager execution
    # allows the lambda layer to infer the output spec.
    tf.config.run_functions_eagerly(False)
    output = tf.compat.v1.keras.layers.Lambda(lambda_fn)(input)

    # switching back to eager for runtime debugging
    tf.config.run_functions_eagerly(True)

    return tf.keras.Model(inputs=input, outputs=output)


@tf.function
def lambda_fn(input):
    i = tf.constant(0, dtype=tf.int32)
    while i < input:
        tf.print("loop iteration", i)
        i = i + 1
    return input


if __name__ == "__main__":

    # this works
    model = build_model_success()
    model(5)

    # this doesn't work
    model = build_model_fail()
    model(5)