from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask):

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=True)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)

        return tuple([tuple(encoded_layers), pooled_output])


class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.traced_bert = self.bert

        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)
        self.traced = False

    def trace_bert(self, input_ids, token_type_ids, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if not self.traced:
            self.traced_bert = torch.jit.trace(self.bert, tuple([input_ids, token_type_ids, attention_mask]))
            self.traced = True

        sequence_output, pooled_output = self.traced_bert(input_ids, token_type_ids, attention_mask)
        return sequence_output, pooled_output



    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        sequence_output_all, pooled_output = self.trace_bert(input_ids, token_type_ids, attention_mask)
        sequence_output = sequence_output_all[-1]

        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score

model = BertForPreTraining.from_pretrained('path_to_pretrained_model')
loss = model(input_ids, token_type_ids, attention_mask, lm_labels, next_sent)
loss.backward()

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(64, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.contiguous()  # INSERTED
        return F.log_softmax(x, dim=1)

m = Net()
x = torch.randn(64, 1, 28, 28, requires_grad=True)
traced_net = torch.jit.trace(m, x)
traced_output = traced_net(x)
tgt = torch.randn(traced_output.size())
traced_output.backward(tgt)