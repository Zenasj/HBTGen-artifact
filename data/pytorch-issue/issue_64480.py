import torch
import torch.nn as nn
import numpy as np

class AutoIntModel(nn.Module):

    def __init__(self, num_features, num_feature_dims,
                embedding_size, num_hid_layers, hidden_sizes, # Embedding layer params
                num_att_layers, num_att_heads, att_embedding_size, use_residual, # attention params
                feature_stds, feature_means, neg_sample_rate, p_dropout, device):

        super(AutoIntModel, self).__init__()

        assert(len(num_feature_dims) == 1)
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        self.neg_sample_rate = neg_sample_rate
        self.num_features = num_features
        self.num_feature_dims = num_feature_dims[0]
        self.device = device
        self.att_layers = nn.ModuleList()

        # Converting our continuous numerical values into an embedding

        self.emb = DeepEmbeddingLayer(num_feature_dims[0], num_hid_layers, hidden_sizes, embedding_size, p_dropout)

        for i in range(num_att_layers):
            layer_input_size = embedding_size if i == 0 else num_att_heads*att_embedding_size
            self.att_layers.append(AttentionLayer(layer_input_size, att_embedding_size, num_att_heads, p_dropout, use_residual))

        self.pred_layer = nn.Linear(num_features*num_att_heads*att_embedding_size, 1)

        self.dropout = nn.Dropout(p_dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.001)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_f, mask, labels=None):

        # Size [N, D*M] where D is number of features and M is number of feature dims
        features = torch.div((input_f - self.feature_means),self.feature_stds)

        # Add a third dimension to make it [N, D, 1]
        if self.num_feature_dims == 1:
            features = torch.unsqueeze(features, -1)
        # Make it [N, D/M, M]
        else:
            bs, d_m = features.shape
            new_size = (bs, int(d_m/self.num_feature_dims), self.num_feature_dims)
            if mask == [-5]:
                features = features.reshape(new_size)
            else:
                tmp_features = torch.tensor(np.zeros(new_size), dtype=torch.float)
                for i in range(len(mask)):
                    m = mask[i]
                    ind = torch.where(m != 0)[0]
                    tmp_features[:, i, :] = features[:, ind]

                features = tmp_features
                features = features.to(self.device)

        embeddings = self.emb(features)

        for layer in self.att_layers:
            embeddings = layer(embeddings)

        embeddings = torch.flatten(embeddings, start_dim=1)

        p_click = self.sigmoid(self.pred_layer(embeddings))
        output = (p_click,)

        if labels is not None:
            loss_func = BCELoss()
            loss = loss_func(p_click, labels)
            output = (loss, ) + output

        return output

def verify_onnx_model(output_dir, model, dummy_in, score):
    test_session = onnxruntime.InferenceSession(os.path.join(output_dir, 'autoint.onnx'))
    test_inputs = {}
    cnt = 0
    for inp in dummy_in:
        test_inputs[test_session.get_inputs()[cnt].name] = to_numpy(inp)
        cnt += 1
    test_outs = test_session.run(None, test_inputs)

    print("pytorch output: ", to_numpy(score), " onnx: ", test_outs)
    assert np.absolute(to_numpy(score)-test_outs)<1e-6
    print("test pass")

# Assumes model has no embeddings as input
def save_onnx_ini_model(model, dataloader, mask, output_dir):

    model.eval()
    mask = mask.to('cuda')
    epoch_iterator = tqdm(dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to('cuda') for t in batch)
       # Batch[0] shape = [bs, 67], batch[1] shape = [bs, 1]
        inputs = {"input_f": batch[0], "mask": mask}
        score = model(**inputs)
        break

    dummy_in = (batch[0], mask)
    # Save .onnx model
    torch.onnx.export(model, dummy_in, os.path.join(output_dir, 'autoint.onnx'), opset_version=11,  export_params=True, do_constant_folding=True, input_names=['input_f', 'mask'], output_names=['prob'], dynamic_axes={'input_f' : {0: 'batch'}, 'prob' : {0: 'batch'}})

    # Verify .onnx model saved gives the same output as pytorch model
    verify_onnx_model(output_dir, model, dummy_in, score)


model = AutoIntModel(num_value_features, num_feature_dims, args.embedding_size, args.num_hid_layers, args.hidden_sizes,
                    args.num_att_layers, args.num_att_heads, args.att_embedding_size, args.use_residual, 
                    feature_stds, feature_means, neg_sample_rate, args.dropout, args.device)
model.to(args.device)
mask = np.ones((23, 3))
save_onnx_ini_model(model, dataloader, mask, './')