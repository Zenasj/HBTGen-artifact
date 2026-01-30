import torch
import torch.nn as nn

class config:
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 8
    TEST_BATCH_SIZE=8
    EPOCHS = 3
    BASE_MODEL_PATH = "bert-base-uncased"
    MODEL_PATH = "model.bin"
    TRAINING_FILE = "/home/mayank.s/ProjectTensorRT/dataset/ner_dataset.csv"
    TOKENIZER = transformers.BertTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        do_lower_case=True
    )


class EntityModel(nn.Module):
    def __init__(self, num_tag, num_pos):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.num_pos = num_pos
        self.bert = transformers.BertModel.from_pretrained(
            config.BASE_MODEL_PATH
        )
        self.bert_drop_1 = nn.Dropout(0.3)
        self.bert_drop_2 = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag)
        self.out_pos = nn.Linear(768, self.num_pos)
    
    def forward(
        self, 
        ids, 
        mask, 
        token_type_ids, 
        target_pos, 
        target_tag
    ):
        
        o1 = self.bert(
            ids, 
            attention_mask=mask, 
            token_type_ids=token_type_ids
        )

        last_hidden_state, pooled_output = o1.to_tuple()
        
        bo_tag = self.bert_drop_1(last_hidden_state)
        bo_pos = self.bert_drop_2(last_hidden_state)

        tag = self.out_tag(bo_tag)
        pos = self.out_pos(bo_pos)

        loss_tag = loss_fn(tag, target_tag, mask, self.num_tag)
        loss_pos = loss_fn(pos, target_pos, mask, self.num_pos)

        loss = (loss_tag + loss_pos) / 2

        return tag, pos, loss


class EntityDataset:
    def __init__(self, texts, pos, tags):
        self.texts = texts
        self.pos = pos
        self.tags = tags
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = self.texts[item]
        pos = self.pos[item]
        tags = self.tags[item]

        ids = []
        target_pos = []
        target_tag =[]

        for i, s in enumerate(text):
            inputs = config.TOKENIZER.encode(
                s,
                add_special_tokens=False
            )
            
            input_len = len(inputs)
            ids.extend(inputs)
            target_pos.extend([pos[i]] * input_len)
            target_tag.extend([tags[i]] * input_len)

        ids = ids[:config.MAX_LEN - 2]
        target_pos = target_pos[:config.MAX_LEN - 2]
        target_tag = target_tag[:config.MAX_LEN - 2]

        ids = [101] + ids + [102]
        target_pos = [0] + target_pos + [0]
        target_tag = [0] + target_tag + [0]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = config.MAX_LEN - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_pos = target_pos + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_pos": torch.tensor(target_pos, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
        }



testDataset = EntityDataset(
    texts=test_sents, pos=test_pos, tags=test_tag
)

test_dataloader = torch.utils.data.DataLoader(
    testDataset, batch_size=config.TEST_BATCH_SIZE, num_workers=0
)

inputs = next(iter(test_dataloader))
ids = inputs['ids']
mask = inputs['mask']
token_type_ids = inputs['token_type_ids']
target_pos = inputs['target_pos']
target_tag = inputs['target_tag']

num_tag = len(list(enc_tag.classes_))
num_pos = len(list(enc_pos.classes_))

device = torch.device("cuda")
fintune_model = EntityModel(num_tag=num_tag, num_pos=num_pos)
fintune_model.load_state_dict(torch.load(config.MODEL_PATH))
fintune_model.to(device)

traced_torch_model = torch.jit.trace(torch_model, [input_ids, attention_mask, token_type_ids, target_pos, target_tag])

trt_model = torch_tensorrt.compile(traced_torch_model, 
    inputs= [torch_tensorrt.Input(shape=[batch_size, 128], dtype=torch.int32),  # input_ids
             torch_tensorrt.Input(shape=[batch_size, 128], dtype=torch.int32),  # attention_mask
             torch_tensorrt.Input(shape=[batch_size, 128], dtype=torch.int32), # token_type_ids/segment_ids
             torch_tensorrt.Input(shape=[batch_size, 128], dtype=torch.int32), # target_pos
             torch_tensorrt.Input(shape=[batch_size, 128], dtype=torch.int32)], # target_tag
    enabled_precisions= {torch.float32}, # Run with 32-bit precision
#     workspace_size=2000000000,
    truncate_long_and_double=True
)

trt_model = torch_tensorrt.compile(traced_torch_model, 
    inputs= [torch_tensorrt.Input(shape=[batch_size, 128], dtype=torch.int32),  # input_ids
             torch_tensorrt.Input(shape=[batch_size, 128], dtype=torch.int32),  # attention_mask
             torch_tensorrt.Input(shape=[batch_size, 128], dtype=torch.int32), # token_type_ids/segment_ids
             torch_tensorrt.Input(shape=[batch_size, 128], dtype=torch.int32), # target_pos
             torch_tensorrt.Input(shape=[batch_size, 128], dtype=torch.int32)], # target_tag
    enabled_precisions= {torch.float32}, # Run with 32-bit precision
#     workspace_size=2000000000,
    truncate_long_and_double=True
)