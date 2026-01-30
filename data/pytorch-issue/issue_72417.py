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
        
        self.num_layers = 1

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True)  # !
        #self.classifier = nn.Linear(hidden_dim, self.num_labels[0])
        self.classifier = nn.Linear(2 * hidden_dim, self.num_labels[0])  # !


    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)


    def forward(self, sentence, labels=None):
        embeds = self.word_embeddings(sentence)
        # lstm_out, _ = self.lstm(embeds)  # lstm_out - 2 tensors, _ - hidden layer
        lstm_out, hidden = self.lstm(embeds)
        
        # Calculate number of directions
        self.num_directions = 2 if self.lstm.bidirectional == True else 1
        
        # Extract last hidden state
        # final_state = hidden.view(self.num_layers, self.num_directions, self.batch_size, self.hidden_dim)[-1]
        final_state = hidden[0].view(self.num_layers, self.num_directions, self.batch_size, self.hidden_dim)[-1]
        # Handle directions
        final_hidden_state = None
        if self.num_directions == 1:
            final_hidden_state = final_state.squeeze(0)
        elif self.num_directions == 2:
            h_1, h_2 = final_state[0], final_state[1]
            # final_hidden_state = h_1 + h_2               # Add both states (requires changes to the input size of first linear layer + attention layer)
            final_hidden_state = torch.cat((h_1, h_2), 1)  # Concatenate both states
        
        self.linear_dims = [0]
        
        # Define set of fully connected layers (Linear Layer + Activation Layer) * #layers
        self.linears = nn.ModuleList()
        for i in range(0, len(self.linear_dims)-1):
            linear_layer = nn.Linear(self.linear_dims[i], self.linear_dims[i+1])
            self.init_weights(linear_layer)
            self.linears.append(linear_layer)
            if i == len(self.linear_dims) - 1:
                break  # no activation after output layer!!!
            self.linears.append(nn.ReLU())
        
        X = final_hidden_state
        
        # Push through linear layers
        for l in self.linears:
            X = l(X)

        print('type(X)', type(X))
        print('len(X)', len(X))
        print('X.shape', X.shape)
        
        print('type(labels)', type(labels))
        print('len(labels)', len(labels))
        print('labels', labels)
        
        print('type(hidden[0])', type(hidden[0]))  # hidden[0] - tensor
        print('len(hidden[0])', len(hidden[0]))
        print('hidden[0].shape', hidden[0].shape)
        print('hidden[0]', labels)

        
        logits = self.classifier(X)  # !  # torch.flip(lstm_out[:,-1,:], [0, 1]) - 1 tensor
        print('type(logits)', type(logits))
        print('len(logits)', len(logits))
        print('logits.shape', logits.shape)
        
        loss = None
        if labels:
            print("len(self.num_labels)", len(self.num_labels))
            print("self.num_labels[0]", self.num_labels[0])
            print("len(labels[0].view(-1))", len(labels[0].view(-1)))
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
        # self.model.load_state_dict(torch.load(model_name), strict=False)  # !
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