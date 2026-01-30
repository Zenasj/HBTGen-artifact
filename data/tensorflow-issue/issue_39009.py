class SumConstraint(Constraint):
    def __init__(self, axis):
        super(SumConstraint, self).__init__()
        self.axis = axis
    def __call__(self, w):
        w = w * K.cast(K.greater_equal(w, 0.), K.floatx())
        s = K.sum(w, axis = self.axis)
        s = s.numpy()
        for i in range(6):
            w[i]/=s[i]
        return w
    def get_config(self):
        return {'axis': self.axis}

x = Dense(3, kernel_initializer = SumConstraint(axis = 1))