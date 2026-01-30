import torch
import torch.nn as nn

def forward(self, Q, D):
        return self.score(self.query(Q), self.doc(D))

def query(self, queries):
        queries = [["[unused0]"] + self._tokenize(q) for q in queries]

        input_ids, attention_mask = zip(
            *[self._encode(x, self.query_maxlen) for x in queries]
        )
        input_ids, attention_mask = (
            self._tensorize(input_ids),
            self._tensorize(attention_mask),
        )

        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)
        Q = torch.nn.functional.normalize(Q, p=2, dim=2)

        return Q

def _encode(self, x, max_length):
        input_ids = self.tokenizer.encode(
            x,
            add_special_tokens=True,
            max_length=max_length,
            truncation="longest_first",
        )

        padding_length = max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + [103] * padding_length

        return input_ids, attention_mask

def _tensorize(self, l):
        return torch.tensor(
            l,
            dtype=torch.long,
            device=self.bert.embeddings.word_embeddings.weight.device,
        )

class TripleTextDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_parquet(data_path)
        self.dims = {col: self.data[col].values.shape for col in self.data.columns}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        Q, PD, ND = (
            row.query,
            row.pos,
            row.neg,
        )
        return {"query": Q, "pos": PD, "neg": ND}


class TripleTextDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.data_dir = hparams.dataset.dir
        self.batch_size = hparams.train.batch_size
        self.num_workers = hparams.dataset.num_workers
        self.use_small = hparams.dataset.use_small

    # def prepare_data(self):
    #     pass

    def setup(self, stage=None):
        self.dims = {}

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if self.use_small:
                self._train_dataset = self._init_dataset("train-sm")
            else:
                self._train_dataset = self._init_dataset("train")
            self._val_dataset = self._init_dataset("val")
            self.dims.update(
                {
                    "train": self._train_dataset.dims,
                    "val": self._val_dataset.dims,
                }
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self._test_dataset = self._init_dataset("test")
            self.dims.update(
                {
                    "test": self._test_dataset.dims,
                }
            )

    def _init_dataset(self, dset_type):
        data_path = Path(self.data_dir) / f"{dset_type}.parquet"
        return TripleTextDataset(data_path)

    def _get_dataloader(self, dataset):
        # num_workers = int(cpu_count() / 2)
        num_workers = self.num_workers
        batch_size = self.batch_size
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._get_dataloader(self._train_dataset)

    def val_dataloader(self):
        return self._get_dataloader(self._val_dataset)

    def test_dataloader(self):
        return self._get_dataloader(self._test_dataset)