import torch.nn as nn
import torch
class Smelu(nn.Module):
    def __init__(self, beta: float = 1.0):
        super(Smelu, self).__init__()
        self.beta = beta

    def forward(self, x):
        return torch.where(torch.abs(x) <= self.beta, ((x + self.beta) ** 2) / (4 * self.beta), nn.functional.relu(x))


class GateNN(nn.Module):
    def __init__(self, input_dim, hidden_unit, output_unit):
        super(GateNN, self).__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_unit),
            nn.ReLU()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_unit, output_unit),
            nn.Sigmoid()
        )

    def forward(self, inputs, training=True):
        hidden = self.hidden_layer(inputs)
        output = 2 * self.output_layer(hidden)
        return output


class LHUC(nn.Module):
    def __init__(self, input_dim, hidden_units, gate_units):
        super(LHUC, self).__init__()
        self.hidden_units = hidden_units
        self.gate_units = gate_units
        self.gate_nn = nn.ModuleList()
        self.dense_layers = nn.ModuleList()
        self.input_dim = input_dim

        # 初始化dense layers
        dense_input = self.input_dim
        for i in range(len(self.hidden_units) - 1):
            layer = nn.Sequential(
                nn.Linear(dense_input, self.hidden_units[i]),
                Smelu()
            )
            dense_input = self.hidden_units[i]
            self.dense_layers.append(layer)
        layer = nn.Linear(self.hidden_units[-2], self.hidden_units[-1])
        self.dense_layers.append(layer)
        # 870, 400, 870
        self.gate_nn.append(GateNN(self.input_dim, self.gate_units, self.input_dim))
        input_dim = self.input_dim
        for i, unit_num in enumerate(self.hidden_units[:-1]):
            self.gate_nn.append(GateNN(self.input_dim, self.gate_units, unit_num))

    def forward(self, inputs):
        #2560, 870
        origin_embedding, auxiliary_embedding = inputs
        hidden = origin_embedding
        for i in range(len(self.hidden_units)):
            gate = self.gate_nn[i](auxiliary_embedding)
            hidden = hidden * gate
            hidden = self.dense_layers[i](hidden)
        output = hidden
        return output


batch_size = 2560


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dht_bn = nn.BatchNorm1d(870)
        self.bias_bn = nn.BatchNorm1d(80)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.dht_tower = LHUC(
            input_dim=870,
            hidden_units=[1024, 512, 256, 64, 1],
            gate_units=512)
        # [96, 64, 1]
        self.bias_mlp = nn.Sequential()
        bias_input_dim = 80
        bias_list = [96, 64, 1]
        for dim in bias_list:
            self.bias_mlp.append(nn.Linear(bias_input_dim, dim))
            self.bias_mlp.append(nn.BatchNorm1d(dim))
            self.bias_mlp.append(nn.Dropout(0.1))
            bias_input_dim = dim

    def forward(self, inputs, training=True):
        # 2560, 87, 10
        dht_table = inputs["dht_table"]
        # 2560, 870
        dht_table = dht_table.reshape([batch_size, -1])
        dht_table = self.dht_bn(dht_table)
        # 2560, 8, 10
        bias_table = inputs["bias_table"]
        bias_table = bias_table.reshape([batch_size, -1])
        bias_table = self.bias_bn(bias_table)
        features = [dht_table, dht_table]
        main_logits = self.dht_tower(features)
        main_pred = nn.functional.sigmoid(main_logits)
        bias_logits = self.bias_mlp(bias_table)
        bias_pred = nn.functional.sigmoid(bias_logits)
        pred = main_pred * bias_pred
        return {"combine_ctr_pred": pred, "dht_ctr_pred": main_pred}
    def compute_loss_and_metrics(self, labels, model_outputs, sample_weights) -> tuple[torch.Tensor, dict]:
        loss = self.criterion(model_outputs["combine_ctr_pred"].reshape(-1), labels["label"].reshape(-1))
        return loss, {"auc": [labels["label"], model_outputs["combine_ctr_pred"]], "loss": loss}
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net().to(device=device)
    batch_dim = torch.export.Dim("batch", min=1, max=65536)
    example_inputs={"lhuc_table":torch.rand(2560, 15, 10).to('cuda'), "bias_table":torch.rand(2560, 8, 10).to('cuda'), "dht_table":torch.rand(2560, 87, 10).to('cuda')}
    inputs = {'inputs':{'lhuc_table':{0:batch_dim,1:torch.export.Dim.AUTO,2:torch.export.Dim.AUTO},'bias_table':{0:batch_dim,1:torch.export.Dim.AUTO,2:torch.export.Dim.AUTO}, 'dht_table':{0:batch_dim,1:torch.export.Dim.AUTO,2:torch.export.Dim.AUTO}}}
    example_tuple = tuple([example_inputs[x] for x in example_inputs.keys()])
    exported = torch.export.export(model, (example_inputs,), dynamic_shapes=inputs)