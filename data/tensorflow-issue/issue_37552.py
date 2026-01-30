import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

class TensorBoard(tf.keras.callbacks.TensorBoard):
    def _collect_learning_rate(self, logs):
        logs = logs or {}
        lr_schedule = getattr(self.model.optimizer, "lr", None)
        if isinstance(lr_schedule, tf.keras.optimizers.schedules.LearningRateSchedule):
            logs["learning_rate"] = tf.keras.backend.get_value(
                lr_schedule(self.model.optimizer.iterations)
            )
        return logs

    def _log_metrics(self, logs, prefix, step):
        super()._log_metrics(self._collect_learning_rate(logs), prefix, step)