if self.momentum is None:
        exponential_average_factor = 0.0
else:
        exponential_average_factor = self.momentum

temp = self.momentum
exponential_average_factor = 0.0 if temp is None else temp