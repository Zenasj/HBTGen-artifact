import torch
import torch.nn as nn

class Attention(nn.Module):

    def __init__(self, input_dim,
                 hidden_dim):
        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.tanh = tanh
        self.softmax = nn.Softmax(dim=-1)
        nn.init.uniform_(self.V, -1, 1)

    def forward(self,
                input,
                context,
                mask):

        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))

        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
        inf = self._inf.unsqueeze(1).expand(mask.size())
        att[mask] = inf[mask]
        alpha = self.softmax(att)
        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)
        return hidden_state, alpha


class Decoder(nn.Module):

    def __init__(self, embedding_dim,
                 hidden_dim):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = Attention(hidden_dim, hidden_dim)
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

    def step(self, x, dec_h, dec_c, context, mask):
        h = dec_h
        c = dec_c

        gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
        input, forget, cell, out = gates.chunk(4, 1)

        input = sigmoid(input)
        forget = sigmoid(forget)
        cell = tanh(cell)
        out = sigmoid(out)
        c_t = (forget * c) + (input * cell)
        h_t = out * tanh(c_t)

        hidden_t, output = self.att(h_t, context, torch.eq(mask, 0))
        hidden_t = tanh(self.hidden_out(torch.cat((hidden_t, h_t), 1)))
        return hidden_t, c_t, output

    def forward(self, embedded_inputs,
                decoder_input,
                dec_h,
                dec_c,
                context):

        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs = []
        pointers = []

        # Recurrence loop
        for _ in range(input_length):
            h_t, c_t, outs = self.step(decoder_input, dec_h, dec_c, context, mask)
            (dec_h, dec_c) = (h_t, c_t)

            masked_outs = outs * mask
            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()
            mask  = mask * (1 - one_hot_pointers)
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim)
            embed_mask = embedding_mask == 1
            decoder_input = embedded_inputs[embed_mask].view(batch_size, self.embedding_dim)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)

        return (outputs, pointers), dec_h, dec_c

torch.onnx.export(
            decoder_scripted,
            (dec_inp, dec_inp0, dec_h0, dec_c0, enc_out),
            decoder_path,
            export_params=True,
            verbose=True,
            input_names=['embedded_inputs', 'dec_input', 'dec_h0', 'dec_c0', 'enc_outs'],
            example_outputs=(dec_out, pt, dec_hn, dec_cn),
            output_names=['dec_output', 'pointer', 'dec_hn', 'dec_cn'],
            do_constant_folding=False,
            dynamic_axes={
                'embedded_inputs': {0: 'batch_size', 1: 'sequence'},
                'dec_input': {0: 'batch_size'},
                'dec_h0': {0: 'batch_size'},
                'dec_c0': {0: 'batch_size'},
                'enc_outs': {0: 'batch', 1: 'sequence'},
            },
            opset_version=11,
        )