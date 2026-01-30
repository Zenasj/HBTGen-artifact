class FailModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lstm = layers.LSTM(64, use_bias=True)

    def call(self, input, training=False):
        return self.lstm(input)