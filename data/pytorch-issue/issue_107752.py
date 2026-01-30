import torch

self.orig_gm: torch.fx.GraphModule = gm.__copy__()

V.debug.draw_orig_fx_graph(self.orig_gm, self.scheduler.nodes)