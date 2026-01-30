import torch

self.longMessage = True
self.assertEqual(torch.int64, torch.int32, msg="boo")

py
msg = lambda orig_msg: f"{orig_msg}\n\n{msg}" if msg and self.longMessage else msg