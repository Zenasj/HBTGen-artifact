import onnx
import torch
import torch.nn as nn

class Embedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
    ):
        super(Embedding, self).__init__(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self._normal_init()

        if padding_idx is not None:
            self.weight.data[self.padding_idx].zero_()

    def _normal_init(self, std=0.02):
        nn.init.normal_(self.weight, mean=0.0, std=std)

class EdgeFeature(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(
        self,
        pair_dim,
        num_edge,
        num_spatial,
    ):
        super(EdgeFeature, self).__init__()
        self.pair_dim = pair_dim

        self.edge_encoder = Embedding(num_edge, pair_dim, padding_idx=0)
        self.shorest_path_encoder = Embedding(num_spatial, pair_dim, padding_idx=0)
        self.vnode_virtual_distance = Embedding(1, pair_dim)

    def forward(self, shortest_path, edge_feat, graph_attn_bias):
        shortest_path = shortest_path
        edge_input = edge_feat

        graph_attn_bias[:, 1:, 1:, :] = self.shorest_path_encoder(shortest_path)

        # reset spatial pos here
        t = self.vnode_virtual_distance.weight.view(1, 1, self.pair_dim)
        graph_attn_bias[:, 1:, 0, :] = t
        graph_attn_bias[:, 0, :, :] = t

        edge_input = self.edge_encoder(edge_input).mean(-2)
        graph_attn_bias[:, 1:, 1:, :] = graph_attn_bias[:, 1:, 1:, :] + edge_input
        return graph_attn_bias

if __name__ == "__main__":
    edge_feature = EdgeFeature(
                pair_dim=256,
                num_edge=64,
                num_spatial=512,
            ).float()
    attn_bias = torch.rand((64, 20, 20, 256),dtype=torch.float32)
    shortest_path = torch.ones((64, 19, 19),dtype=torch.int64)
    edge_feat = torch.ones((64, 19, 19, 3),dtype=torch.int64)
    torch.onnx.export(edge_feature, 
                        (shortest_path, edge_feat, attn_bias),
                        "edge_feature.onnx",
                        input_names=["shortest_path", "edge_feat", "attn_bias"],
                        # verbose=True,
                        opset_version=14,
                        output_names=["graph_attn_bias"],
                        dynamic_axes={
                            "attn_bias":{0: "batch_size", 1: "seq_len_1", 2: "seq_len_1"},
                            "shortest_path":{0: "batch_size", 1: "seq_len", 2: "seq_len"},
                            "edge_feat":{0: "batch_size", 2: "seq_len", 3: "seq_len"},
                            "graph_attn_bias":{0: "batch_size", 1: "seq_len_1", 2: "seq_len_1"}
                        }
                        )
    from onnxsim import simplify
    model = onnx.load("edge_feature.onnx")
    # convert model
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, "edge_feature_modified.onnx")