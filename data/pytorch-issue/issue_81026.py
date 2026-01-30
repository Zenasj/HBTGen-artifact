import torch.nn as nn

import os
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForPreTraining, BertConfig

# paths
proj_dir = '/scratch/ddegenaro'
def in_proj_dir(dir):
    return os.path.join(proj_dir, dir)
pretraining_test = in_proj_dir('pretraining_test.txt')
pretraining_txt = in_proj_dir('pretraining.txt')
inits = in_proj_dir('inits')
ckpts = in_proj_dir('ckpts')
trained = in_proj_dir('trained')

print('Getting tokenizer.')
# get tokenizer and initialize teacher model mBERT
tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
print('Done.')
print('Getting mBERT.')
# this line will complain that decoder bias was not in the checkpoint
mBERT = BertForPreTraining.from_pretrained("bert-base-multilingual-cased")
print('Done.')

teacher = mBERT # first network to copy from
MSELoss = torch.nn.MSELoss() # loss between logits of two models
batch_size = 65536 # batch size
epochs = 1 # num epochs - SHOULD BE 32

class BertData(Dataset):
    def __init__(self):
        print('Reading in corpus. Warning: requires ~ 50 GB of RAM.')
        self.corpus = open(pretraining_test).readlines()
        print('Done.')
    def __len__(self):
        return len(self.corpus)
    def __getitem__(self, idx):
      return tokenizer(self.corpus[idx], return_tensors='pt', padding='max_length', truncation=True, max_length=512)

dataset = BertData()

data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1, pin_memory=True) # should have num_workers=8 or 12

for i in reversed(range(11,12)): # TA builder loop - SHOULD BE 2, 12

  teacher_state_dict = teacher.state_dict()

  # create a BertConfig with a multilingual vocabulary for the next TA
  config_obj = BertConfig(vocab_size=119547, num_hidden_layers=i)

  student = BertForPreTraining(config_obj) # initialize next model and state dict
  student_state_dict = OrderedDict()

  torch.cuda.empty_cache()

  teacher.to('cuda') # use GPU
  student.to('cuda')

  print('Building student.')
  for key in teacher_state_dict: # copy architecture and weights besides top layer
    if str(i) not in key:
      student_state_dict[key] = deepcopy(teacher_state_dict[key])
  print('Done.')

  # save init for this TA
  print('Saving student.')
  torch.save(student_state_dict, os.path.join(inits, 'ta' + str(i)))
  print('Done.')

  # load next state dict into the next model
  student.load_state_dict(student_state_dict)

  student.train() # ensure training mode

  # generate Adam optimizer close to mBERT's
  optimizer = torch.optim.Adam(student.parameters(), lr=(batch_size/256*1e-4),
                             betas=(0.9, 0.999), eps=1e-06, weight_decay=0)

  optimizer.zero_grad(set_to_none=True) # just to be sure

  with torch.set_grad_enabled(True):

    for k in range(epochs):

      start = datetime.now()

      print(f'Begin epoch {k+1}/{epochs}. Current time: {datetime.now()}.')

      loss = 0 # initialize

      for batch_idx, inputs in enumerate(data_loader):

        for j in inputs:
          inputs[j] = inputs[j][0]
        inputs = inputs.to('cuda')

        # get teacher and student predictions
        teacher_logits = teacher(**inputs).prediction_logits
        student_logits = student(**inputs).prediction_logits
        
        # calculate the loss between them and update
        loss = MSELoss(teacher_logits, student_logits) / batch_size
      
        # learning step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss = 0
        print(batch_idx+1, (datetime.now()-start)/(batch_idx+1))
    
      torch.save(student.state_dict(), os.path.join(ckpts, 'ta' + str(i) + '_ckpt' + str(k)))

  # save trained model for this TA
  torch.save(student.state_dict(), os.path.join(trained, 'ta' + str(i)))

  teacher = student # prepare to initialize next network

# end for