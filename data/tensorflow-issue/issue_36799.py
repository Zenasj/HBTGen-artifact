python
class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, val_data, val_labels, fmetrics):
        super().__init__()
        self.val_data = val_data
        self.val_labels = val_labels
        self.fmetrics = fmetrics
        # TODO: more checks

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        predict_results = self.model.predict(self.val_data)

        for fm in self.fmetrics:
            metric_name, value = fm(self.val_labels, predict_results)
            logs[metric_name] = value