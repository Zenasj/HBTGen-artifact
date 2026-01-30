if self.num_bad_epochs > self.patience:
    self._reduce_lr(epoch)
    self.cooldown_counter = self.cooldown
    self.num_bad_epochs = 0

if self.num_bad_epochs >= self.patience:
    self._reduce_lr(epoch)
    self.cooldown_counter = self.cooldown
    self.num_bad_epochs = 0