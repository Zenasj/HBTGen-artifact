import tensorflow as tf

class MemoryLogHook(tf.train.SessionRunHook):
    def begin(self):
        K._GRAPH_LEARNING_PHASES = {}
        K._GRAPH_UID_DICTS = {}

def model_fn(features, labels, mode):
    """model_fn for keras Estimator."""

    # Clear graphs before doing anything.
    # Cannot call K.clear_session because the graph is read_only.
    K._GRAPH_LEARNING_PHASES = {}
    K._GRAPH_UID_DICTS = {}