import torch.nn as nn

class InterNodeMoELayerOutTest(nn.Module):
    def __init__(self, shared_module, hidden_size, local_rank, pg):
        super().__init__()
        self.shared_module = shared_module
        # self.fc3 = nn.Linear(hidden_size, hidden_size).cuda(local_rank)  <---- when delete this line, we can reproduce the error.
        self.pg = pg

    def forward(self, x):
        logging.info("Test: output_of_intra_node_moe_tensor = {}".format(x))
        x = _AllToAll.apply(self.pg, x)
        return x

class InterNodeMoELayerOutTest(nn.Module):
    def __init__(self, shared_module, hidden_size, local_rank, pg):
        super().__init__()
        self.shared_module = shared_module
        self.dummy_tensor = nn.Linear(1, 1).cuda(local_rank)
        self.pg = pg

    def forward(self, x):
        logging.info("Test: output_of_intra_node_moe_tensor = {}".format(x))
        x = _AllToAll.apply(self.pg, x)
        return x