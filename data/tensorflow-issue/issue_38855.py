from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf
import logging

LOGGER = logging.getLogger("tensorflow")

# Build model.
def build_model(name, random_bn_params=None):
    if name == "ResNet50":
        model = tf.keras.applications.ResNet50(
            input_shape=(224, 224, 3),
            include_top=False,
            pooling="avg")
    elif name == "MobileNetV2_1.0":
        model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            pooling="avg")
    elif name == "VGG16":
        model = tf.keras.applications.VGG16(
            input_shape=(224, 224, 3),
            include_top=False,
            pooling="avg")
    else:
        if random_bn_params:
            params = {"beta_initializer": tf.random_normal_initializer(),
                      "gamma_initializer": tf.random_normal_initializer(),
                      "moving_mean_initializer": tf.random_normal_initializer(),
                      "moving_variance_initializer": tf.random_normal_initializer()}
        else:
            params = dict()
        layer = tf.keras.layers.Conv2D(1, (3, 3), use_bias=False)
        layer_bn = tf.compat.v1.keras.layers.BatchNormalization(**params)
        inp = tf.keras.layers.Input((224, 224, 3))
        out = layer(inp)
        out = layer_bn(out)
        model = tf.keras.Model(inputs=inp,
                               outputs=out)
    return model

# Model params.
MODEL_NAME = "VGG161"  # ResNet50、VGG16、MobileNetV2_1.0 are available, pre-trained models are loaded. Name not in the above three will build a Conv2D + BN network.
RANDOM_BN_PARAMS = True  # Whether to initialize BN  by randomized params when using a customized Conv2D + BN network.
INPUT_TENSOR_FUNC = lambda: tf.ones((1, 224, 224, 3))  # Fixed model input.

# Use Keras model to predict.
model = build_model(MODEL_NAME, RANDOM_BN_PARAMS)
model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0, decay=1.0),
              loss='mean_squared_error')
model_output = model.predict(INPUT_TENSOR_FUNC())

# Use Estimator transformed from Keras model to predict.
estimator = tf.keras.estimator.model_to_estimator(model)
estimator_output = next(estimator.predict(INPUT_TENSOR_FUNC))

# Use a handmake model equivalent to Keras model to predict.
def _init_variables_from_checkpoint(checkpoint_path, model_dir):
    flags_checkpoint_path = checkpoint_path
    # Warn the user if a checkpoint exists in the model_dir. Then ignore.
    if tf.compat.v1.train.latest_checkpoint(model_dir):
        LOGGER.info(
            "Ignoring model_init_name because a checkpoint already exists in %s." % model_dir)
        return None
    if flags_checkpoint_path is "":
        return None

    # Gather all trainable variables to initialize.
    variables_to_init = tf.compat.v1.trainable_variables()

    variables_to_init_dict = {var.name.rsplit(":", 1)[0]: var for var in variables_to_init}

    if tf.compat.v1.gfile.IsDirectory(flags_checkpoint_path):
        checkpoint_path = tf.compat.v1.train.latest_checkpoint(flags_checkpoint_path)
    else:
        checkpoint_path = flags_checkpoint_path

    LOGGER.info("Fine-tuning from %s." % checkpoint_path)

    # Gather all available variables to initialize.
    available_var_map = _get_variables_available_in_checkpoint(variables_to_init_dict,
                                                               checkpoint_path)

    init_op = tf.compat.v1.train.init_from_checkpoint(checkpoint_path, available_var_map)
    LOGGER.info("%d/%d variables in checkpoint has been restored." % (len(available_var_map),
                                                                      len(variables_to_init)))

    return tf.compat.v1.train.Scaffold(init_op=init_op)


def _get_variables_available_in_checkpoint(variables,
                                           checkpoint_path,
                                           include_global_step=False):
    """Returns the subset of variables in the checkpoint.

    Inspects given checkpoint and returns the subset of variables that are
    available in it.

    Args:
        variables: A dictionary of variables to find in checkpoint.
        checkpoint_path: Path to the checkpoint to restore variables from.
        include_global_step: Whether to include `global_step` variable, if it
            exists. Default True.

    Returns:
        A dictionary of variables.

    Raises:
        ValueError: If `variables` is not a dict.
    """
    if not isinstance(variables, dict):
        raise ValueError("`variables` is expected to be a dict.")

    # Available variables
    ckpt_reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
    ckpt_vars_to_shape_map = ckpt_reader.get_variable_to_shape_map()
    if not include_global_step:
        ckpt_vars_to_shape_map.pop(tf.compat.v1.GraphKeys.GLOBAL_STEP, None)
    vars_in_ckpt = {}

    # for key in ckpt_vars_to_shape_map:
    #     LOGGER.info("Available variable name: %s", key)

    for variable_name, variable in sorted(variables.items()):
        if variable_name in ckpt_vars_to_shape_map:
            if ckpt_vars_to_shape_map[variable_name] == variable.shape.as_list():
                vars_in_ckpt[variable_name] = variable
            else:
                LOGGER.warning("Variable [%s] is available in checkpoint, but has an incompatible "
                               "shape with model variable. Checkpoint shape: [%s], model variable "
                               "shape: [%s]. This variable will not be initialized from the "
                               "checkpoint.",
                               variable_name,
                               ckpt_vars_to_shape_map[variable_name],
                               variable.shape.as_list())
        else:
            LOGGER.warning("Variable [%s] is not available in checkpoint", variable_name)
    return vars_in_ckpt

def model_fn(features, labels, mode):
    # Set Keras learning phase for alter BatchNorm and Dropout performance.
    tf.keras.backend.set_learning_phase(mode == tf.estimator.ModeKeys.TRAIN)

    m = build_model(MODEL_NAME, RANDOM_BN_PARAMS)
    predictions = m(features)
    
    scaffold = None
    if MODEL_NAME in {"ResNet50", "VGG16", "MobileNetV2_1.0"}:
        scaffold = _init_variables_from_checkpoint("w:/keras/%s.ckpt" % MODEL_NAME, ".")

    # Create estimator_spec for Estimator.
    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        scaffold = scaffold
    )
    return estimator_spec
estimator_handmake = tf.estimator.Estimator(model_fn=model_fn)
estimator_handmake_output = next(estimator_handmake.predict(INPUT_TENSOR_FUNC))

# Compare the output range.
print(np.max(model_output), np.min(model_output))
print(np.max(estimator_output[list(estimator_output.keys())[0]]), np.min(estimator_output[list(estimator_output.keys())[0]]))
print(np.max(estimator_handmake_output), np.min(estimator_handmake_output))