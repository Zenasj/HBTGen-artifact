import torch

torch.onnx.export(
    model=model,
    args=(x_dict, edge_index_dict, edge_attr_dict, {}),
    f=save_path,
    verbose=False,
    input_names=["x_dict", "edge_index_dict", "edge_attr_dict"],
    output_names=["out"],
)

def forward(self,
    argument_1: Dict[str, Tensor],
    argument_2: Dict[str, Tensor],
    argument_3: Dict[str, Tensor]) -> Dict[str, Tensor]:
  state_encoder = self.state_encoder
  x = argument_1["game_vertex"]
  x0 = argument_1["state_vertex"]
  edge_index = argument_2["game_vertex to game_vertex"]
  edge_index0 = argument_2["game_vertex in state_vertex"]
  edge_index1 = argument_2["game_vertex history state_vertex"]
  edge_index2 = argument_2["state_vertex parent_of state_vertex"]
  edge_weight = argument_3["game_vertex history state_vertex"]
  _0 = (state_encoder).forward(x, edge_index, x0, edge_index2, edge_index1, edge_weight, edge_index0, )
  _1 = {"state_vertex": _0, "game_vertex": x}
  return _1