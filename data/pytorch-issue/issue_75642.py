import torch
import torch.nn as nn
import pytorch_lightning as pl

class MovieLensDummyDataset(torch.utils.data.Dataset):
    #dummy version of the dataset with synthetic data
    def __init__(self, n, n_users, n_movies):

        self.users = torch.randint(0,n_users,(n,)).type(torch.int32)
        self.items = torch.randint(0,n_users,(n,)).type(torch.int32)
        self.labels = torch.randint(0,2,(n,)).type(torch.uint8)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def __len__(self):
        return self.users.shape[0]


class NCF(pl.LightningModule):
    """ Neural Collaborative Filtering (NCF)
    
        Args:
            num_users (int): Number of unique users
            num_items (int): Number of unique items
            ratings (pd.DataFrame): Dataframe containing the movie ratings for training
            all_movieIds (list): List containing all movieIds (train + test)
    """
    
    def __init__(self, num_users, num_items):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=8)
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc1_activation = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc2_activation = nn.ReLU()
        self.output = nn.Linear(in_features=32, out_features=1)
        self.out_activation = nn.Sigmoid()

        self.loss_func = nn.BCELoss()
        
    def forward(self, user_input, item_input):
        
        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        # Concat the two embedding layers
        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        # Pass through dense layer
        vector = self.fc1_activation(self.fc1(vector))
        vector = self.fc2_activation(self.fc2(vector))

        # Output layer
        pred = self.out_activation(self.output(vector))

        return pred
    
    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = self.loss_func(predicted_labels, labels.view(-1, 1).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return self._train_dataloader

    def set_train_dataloader(self, dl):
        self._train_dataloader = dl

    def set_test_dataloader(self, dl):
        self._test_dataloader = dl

num_users=13849
num_items=19103
num_samples=10142520
#the bug is here - the whole thing works if num_users supplied to the dataset is <= num_users supplied to the model - the issue is about the size of the embedding.
#on nvidia hardware, this bug causes an error and the program stops. on amd hardware, supplying this leads to a GPU hang/reset
train_dummy_ds = MovieLensDummyDataset(num_samples, 10*num_users, num_items)
train_dummy_dl = torch.utils.data.DataLoader(train_dummy_ds, batch_size=2048, num_workers=12)
model = NCF(num_users, num_items)

model.set_train_dataloader(train_dummy_dl)

trainer = pl.Trainer(max_epochs=5,
                     gpus=1,
                     progress_bar_refresh_rate=50,
                     logger=False,
                     checkpoint_callback=False,
                     amp_backend='native')

trainer.fit(model)

import torch
z = torch.nn.Embedding(3,256).cuda()
p = torch.Tensor([64]).type(torch.int32).cuda()
o = z(p)

import torch
z = torch.nn.Embedding(3,256).cuda()
p = torch.Tensor([64]).type(torch.int32).cuda()
o = z(p)