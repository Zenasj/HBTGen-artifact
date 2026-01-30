class CD(keras.Model):
    def __init__(self, rbm, mcsteps = 5, **kwargs):
        super().__init__(self, **kwargs)