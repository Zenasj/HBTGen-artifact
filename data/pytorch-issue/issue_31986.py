def reset_parameters(self):
    for module in self.children():
         module.reset_parameters()