import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn
import torch.optim as optim
import math
import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import sys

# Hyperparameters
BATCH_SIZE = 16  # Adjust as needed for 2 GPUs
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
D_MODEL = 512
NHEAD = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1
WEIGHT_DECAY = 0.0001
CLIP_GRAD_NORM = 1.0
ACCUMULATION_STEPS = 4  # Adjust as needed

# Step 1: Define the Dataset Class
class TranslationDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.src_tokenizer = get_tokenizer('basic_english')
        self.tgt_tokenizer = get_tokenizer('basic_english')

        print(f"Loading dataset from {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = 0
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            src, tgt = line.split('","')
                            src = src[2:]
                            tgt = tgt[:-3]
                            self.data.append((src, tgt))
                        except ValueError:
                            print(f"Skipping malformed line: {line}")
                    line_count += 1
                    if line_count % 10000 == 0:
                        print(f"Processed {line_count} lines...")
            print(f"Dataset loaded successfully. {len(self.data)} samples found.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise e

        print("Building source vocabulary...")
        self.src_vocab = build_vocab_from_iterator(self._yield_tokens(self.data, 0), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
        print("Building target vocabulary...")
        self.tgt_vocab = build_vocab_from_iterator(self._yield_tokens(self.data, 1), specials=["<unk>", "<pad>", "<bos>", "<eos>"])

        print("Setting default indices for unknown tokens...")
        self.src_vocab.set_default_index(self.src_vocab["<unk>"])
        self.tgt_vocab.set_default_index(self.tgt_vocab["<unk>"])

        self.SRC_VOCAB_SIZE = len(self.src_vocab)
        self.TGT_VOCAB_SIZE = len(self.tgt_vocab)
        print(f"Source vocabulary size: {self.SRC_VOCAB_SIZE}")
        print(f"Target vocabulary size: {self.TGT_VOCAB_SIZE}")

    def _yield_tokens(self, data, index):
        for src, tgt in data:
            yield self.src_tokenizer(src) if index == 0 else self.tgt_tokenizer(tgt)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_tokens = [self.src_vocab["<bos>"]] + [self.src_vocab[token] for token in self.src_tokenizer(src)] + [self.src_vocab["<eos>"]]
        tgt_tokens = [self.tgt_vocab["<bos>"]] + [self.tgt_vocab[token] for token in self.tgt_tokenizer(tgt)] + [self.tgt_vocab["<eos>"]]
        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)

# Step 2: Define the Transformer Model 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self.d_model = d_model
        self._reset_parameters()

    def _reset_parameters(self):
        print("Initializing model parameters...")
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        print("Performing forward pass...")
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        memory = self.transformer(src, tgt, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        output = self.fc_out(memory)
        return output

# Step 3: Distributed Training Setup 
def setup(rank, world_size):
    os.environ['NCCL_DEBUG'] = 'INFO'
    if torch.cuda.is_available():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        print(f"Initializing process group with world size: {world_size}")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        print("CUDA not available, running on CPU.")

def cleanup():
    if dist.is_initialized():
        print("Cleaning up distributed environment...")
        dist.destroy_process_group()

def create_mask(src, tgt, dataset):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)
    src_padding_mask = (src == dataset.src_vocab["<pad>"]).transpose(0, 1)
    tgt_padding_mask = (tgt == dataset.tgt_vocab["<pad>"]).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def collate_fn(batch, dataset):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=dataset.src_vocab["<pad>"])
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=dataset.tgt_vocab["<pad>"])
    return src_batch, tgt_batch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def evaluate(model, dataloader, criterion, dataset):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_tokens = 0

    print("Evaluating model...")
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(model.module.src_embedding.weight.device)
            tgt = tgt.to(model.module.src_embedding.weight.device)
            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, dataset)

            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = tgt[1:, :]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

            total_loss += loss.item()

            _, predicted = torch.max(logits, dim=-1)
            correct = (predicted == tgt_out).masked_select(tgt_out.ne(dataset.tgt_vocab["<pad>"]))
            total_accuracy += correct.sum().item()
            total_tokens += tgt_out.ne(dataset.tgt_vocab["<pad>"]).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_accuracy / total_tokens
    print(f"Evaluation results: Loss - {avg_loss:.4f}, Accuracy - {accuracy:.4f}")
    return avg_loss, accuracy

def train_epoch(model, dataloader, optimizer, criterion, accumulation_steps, scaler, dataset):
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_tokens = 0

    print("Training epoch...")
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for i, (src, tgt) in enumerate(progress_bar):
        src = src.to(model.module.src_embedding.weight.device)
        tgt = tgt.to(model.module.src_embedding.weight.device)
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, dataset)

        with torch.cuda.amp.autocast():
            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            tgt_out = tgt[1:, :]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        l2_reg = torch.tensor(0., requires_grad=True)
        for name, param in model.named_parameters():
            if 'weight' in name:
                l2_reg = l2_reg + torch.norm(param, 2)
        loss = loss + WEIGHT_DECAY * l2_reg

        loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        _, predicted = torch.max(logits, dim=-1)
        correct = (predicted == tgt_out).masked_select(tgt_out.ne(dataset.tgt_vocab["<pad>"]))
        total_accuracy += correct.sum().item()
        total_tokens += tgt_out.ne(dataset.tgt_vocab["<pad>"]).sum().item()

        progress_bar.set_postfix({
            'Loss': f'{total_loss / (i + 1):.4f}',
            'Accuracy': f'{total_accuracy / total_tokens:.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = total_accuracy / total_tokens
    print(f"Training results: Loss - {avg_loss:.4f}, Accuracy - {accuracy:.4f}")
    return avg_loss, accuracy

def translate(model, src_sentence, src_vocab, tgt_vocab, device, max_len=50):
    model.eval()
    src_tokenizer = get_tokenizer('basic_english')
    src_tokens = [src_vocab["<bos>"]] + [src_vocab[token] for token in src_tokenizer(src_sentence)] + [src_vocab["<eos>"]]
    src_tensor = torch.LongTensor(src_tokens).unsqueeze(1).to(device)

    src_mask = torch.zeros((src_tensor.size(0), src_tensor.size(0)), device=device).type(torch.bool)

    print("Translating...")
    with torch.inference_mode():
        memory = model.transformer.encoder(model.pos_encoder(model.src_embedding(src_tensor) * math.sqrt(model.d_model)), src_mask)

    ys = torch.ones(1, 1).fill_(tgt_vocab["<bos>"]).type(torch.long).to(device)
    for i in range(max_len-1):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(0)).to(device)

        with torch.inference_mode():
            out = model.transformer.decoder(model.pos_encoder(model.tgt_embedding(ys) * math.sqrt(model.d_model)),
                                            memory, tgt_mask)
            out = model.fc_out(out)

        prob = out[-1].detach()

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src_tensor.data).fill_(next_word)], dim=0)
        if next_word == tgt_vocab["<eos>"]:
            break

    ys = ys.flatten()
    translated_tokens = [tgt_vocab.get_itos()[token] for token in ys if token not in [tgt_vocab["<bos>"], tgt_vocab["<eos>"], tgt_vocab["<pad>"]]]
    return " ".join(translated_tokens)

# Main Function
def main():
    try:
        if 'WORLD_SIZE' in os.environ:
            world_size = int(os.environ['WORLD_SIZE'])
        else:
            world_size = torch.cuda.device_count()  # Assuming 1 GPU per process

        if 'RANK' in os.environ:
            rank = int(os.environ['RANK'])
        else:
            rank = 0  # Default rank if not running in a distributed environment

        print(f"Setting up distributed environment with rank {rank} and world size {world_size}...")
        setup(rank, world_size)

        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Load dataset (adjust path as needed)
        file_path = '/kaggle/input/24bdata/newcode15M.txt'
        dataset = TranslationDataset(file_path)
        print("Dataset loaded successfully!")

        # Get vocab sizes from dataset
        SRC_VOCAB_SIZE = dataset.SRC_VOCAB_SIZE
        TGT_VOCAB_SIZE = dataset.TGT_VOCAB_SIZE

        # Split dataset into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        print(f"Splitting dataset: Train size - {train_size}, Validation size - {val_size}")
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=lambda x: collate_fn(x, dataset))
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, collate_fn=lambda x: collate_fn(x, dataset))

        print(f"Number of train batches: {len(train_dataloader)}")
        print(f"Number of validation batches: {len(val_dataloader)}")

        # Initialize model, optimizer, and loss function
        print("Initializing model...")
        model = TransformerModel(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT).to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

        print("Initializing optimizer...")
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        print("Initializing loss function...")
        criterion = nn.CrossEntropyLoss(ignore_index=dataset.tgt_vocab["<pad>"])
        print("Initializing learning rate scheduler...")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

        print("Initializing gradient scaler...")
        scaler = torch.cuda.amp.GradScaler()

        early_stopping = EarlyStopping(patience=7, min_delta=0.01)
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        sample_sentence = "Hello, how are you?"  # Sample sentence for translation

        dist.barrier()  # Add a barrier to synchronize processes

        for epoch in range(NUM_EPOCHS):
            start_time = time.time()

            print(f"Epoch {epoch+1}/{NUM_EPOCHS} started...")
            print("Starting training epoch...")
            train_loss, train_accuracy = train_epoch(model, train_dataloader, optimizer, criterion, ACCUMULATION_STEPS, scaler, dataset)
            print("Starting evaluation epoch...")
            val_loss, val_accuracy = evaluate(model, val_dataloader, criterion, dataset)

            scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Time: {epoch_mins:.0f}m {epoch_secs:.2f}s")

            # Generate sample translation
            if rank == 0:  # Only rank 0 prints to avoid clutter
                translation = translate(model.module, sample_sentence, dataset.src_vocab, dataset.tgt_vocab, device)
                print(f"Sample Translation:")
                print(f"Input: {sample_sentence}")
                print(f"Output: {translation}")
                print()

            # Save the model for each epoch (only rank 0 saves to avoid conflicts)
            if rank == 0:
                torch.save(model.module.state_dict(), f'translation_model_epoch_{epoch+1}.pth')
                print(f"Model saved for epoch {epoch+1}")

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        if rank == 0:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')

            plt.subplot(1, 2, 2)
            plt.plot(train_accuracies, label='Train Accuracy')
            plt.plot(val_accuracies, label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Training and Validation Accuracy')

            plt.tight_layout()
            plt.savefig('training_plots.png')
            plt.show()

            torch.save(model.module.state_dict(), 'final_translation_model.pth')
            print("Final model saved.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cleanup()

if __name__ == "__main__" or ('jupyter' in sys.modules):
    main()
    print("Starting main()")