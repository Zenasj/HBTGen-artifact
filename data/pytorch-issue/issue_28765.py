import torch

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="/tmp/tb_logs")
writer.add_hparams({"a": 0, "b": 0}, {"hparam/test_accuracy": 0.5})
writer.add_hparams({"a": 0, "b": 1}, {"hparam/test_accuracy": 0.6})
writer.add_hparams({"a": 1, "b": 0}, {"hparam/test_accuracy": 0.61})
writer.add_hparams({"a": 1, "b": 1}, {"hparam/test_accuracy": 0.4})
writer.add_hparams({"a": 2.0, "b": 1.5}, {"hparam/test_accuracy": 0.7})

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="/tmp/tb_logs")
writer.add_hparams({"a": 0, "b": 0}, {"hparam/test accuracy": 0.5})
writer.add_hparams({"a": 0, "b": 1}, {"hparam/test accuracy": 0.6})
writer.add_hparams({"a": 1, "b": 0}, {"hparam/test accuracy": 0.61})
writer.add_hparams({"a": 1, "b": 1}, {"hparam/test accuracy": 0.4})
writer.add_hparams({"a": 2.0, "b": 1.5}, {"hparam/test accuracy": 0.7})