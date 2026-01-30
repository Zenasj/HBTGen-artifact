tag_space = self.classifier(lstm_out[:,-1,:])

from argparse import ArgumentParser

import torchmetrics
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):

    def __init__(self, 
        num_classes, 
        batch_size=10,
        embedding_dim=100, 
        hidden_dim=50, 
        vocab_size=128):

        super(LSTMClassifier, self).__init__()

        initrange = 0.1

        self.num_labels = num_classes
        n = len(self.num_labels)
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)  # !
        
        print("# !")
        
        bi_grus = torch.nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        reverse_gru = torch.nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=False)
        
        self.lstm.weight_ih_l0_reverse = bi_grus.weight_ih_l0_reverse
        self.lstm.weight_hh_l0_reverse = bi_grus.weight_hh_l0_reverse
        self.lstm.bias_ih_l0_reverse = bi_grus.bias_ih_l0_reverse
        self.lstm.bias_hh_l0_reverse = bi_grus.bias_hh_l0_reverse
        
        bi_output, bi_hidden = bi_grus()
        reverse_output, reverse_hidden = reverse_gru()
        
        print("# !")

        # self.classifier = nn.Linear(hidden_dim, self.num_labels[0])
        self.classifier = nn.Linear(2 * hidden_dim, self.num_labels[0])  # !


    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)


    def forward(self, sentence, labels=None):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)  # lstm_out - 2 tensors, _ - hidden layer
        print(lstm_out[:,-1,:])
        tag_space = self.classifier(lstm_out[:,-1,:] + lstm_out[:,-1,:])  # !  # lstm_out[:,-1,:] - 1 tensor
        logits = F.log_softmax(tag_space, dim=1)
        loss = None
        if labels:
            loss = F.cross_entropy(logits.view(-1, self.num_labels[0]), labels[0].view(-1))
        return loss, logits


class LSTMTaggerModel(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        class_map,
        from_checkpoint=False,
        model_name='last.ckpt',
        learning_rate=3e-6,
        **kwargs,
    ):

        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = LSTMClassifier(num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_name), strict=False)  # !
        self.class_map = class_map
        self.num_classes = num_classes
        self.valid_acc = torchmetrics.Accuracy()
        self.valid_f1 = torchmetrics.F1()


    def forward(self, *input, **kwargs):
        return self.model(*input, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        loss, _ = self(x, labels=y_true)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        _, y_pred = self(x, labels=y_true)
        preds = torch.argmax(y_pred, axis=1)
        self.valid_acc(preds, y_true[0])
        self.log('val_acc', self.valid_acc, prog_bar=True)
        self.valid_f1(preds, y_true[0])
        self.log('f1', self.valid_f1, prog_bar=True)     

    def configure_optimizers(self):
        'Prepare optimizer and schedule (linear warmup and decay)'
        opt = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.tensor([x['loss']
                                 for x in training_step_outputs]).mean()
        self.log('train_loss', avg_loss)
        print(f'###score: train_loss### {avg_loss}')

    def validation_epoch_end(self, val_step_outputs):
        acc = self.valid_acc.compute()
        f1 = self.valid_f1.compute()
        self.log('val_score', acc)
        self.log('f1', f1)
        print(f'###score: val_score### {acc}')

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("OntologyTaggerModel")       
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", default=2e-3, type=float)
        return parent_parser