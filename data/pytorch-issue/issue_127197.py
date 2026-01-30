import torch
import math

class RandomDataset(Dataset):
    """Dataset with random input_ids and labels, attention_mask all ones."""
    
    def __init__(self, args: TrainingArguments, size: int):
        super(RandomDataset, self).__init__()
        self.size = size
        self.seq_size = get_sequence_parallel_size()
        self.model_max_length = int(args.model_max_length / self.seq_size) 
        
        vocab_size = 1000
        # Randomly generating input_ids and labels
        self.input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(self.size, self.model_max_length),
            dtype=torch.long
        )
        self.labels = torch.randint(
            low=0,
            high=vocab_size,
            size=(self.size, self.model_max_length),
            dtype=torch.long
        )
        # attention_mask all ones
        self.attention_mask = torch.ones(
            (self.size, self.model_max_length),
            dtype=torch.long
        )

    def __len__(self):
        return self.size

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if isinstance(i, list):
            assert False, f"bs >1 not supported: {i}"
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, training_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = RandomDataset(training_args, 256)
    rank0_print("Loading data...")

    #train_json = json.load(open(data_args.data_path, "r"))
    #train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    #if data_args.eval_data_path:
    #    eval_json = json.load(open(data_args.eval_data_path, "r"))
    #    eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    #else:
    eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )#.to(dtype=torch.float16)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )

    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        trainer_save_model_safe(trainer)


if __name__ == "__main__":
    train()

print("devices:", [[t.device if t is not None else None for t in tt] for tt in tensorlistlist])
print("dtypes:", [[t.dtype if t is not None else None for t in tt] for tt in tensorlistlist])

torch.set_default_device("cuda")