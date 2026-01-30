# train.py
import time
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from image_dataset import ImageDataset


class CustomProcess(object):
  def __init__(self, dataset, model):
    self.dataset = dataset
    self.model = model

  def run(self):
    mp.spawn(self.train, nprocs=4)

  def train(self, id):
    pytorch_loader = DataLoader(dataset=self.dataset,
                                batch_size=64,
                                shuffle=False,
                                sampler=None,
                                batch_sampler=None,
                                num_workers=4,
                                collate_fn=None,
                                pin_memory=True,
                                drop_last=False,
                                timeout=0,
                                worker_init_fn=None,
                                multiprocessing_context=None)

    train_time = 0
    stime = time.time()
    # model.share_memory()
    for data in pytorch_loader:
      # train
      output = model(data)
      train_time = train_time + 1
      if train_time % 100 == 0:
        etime = time.time()
        print("train time: {}, cost time: {}".format(train_time, etime - stime))
        stime = etime


if __name__ == "__main__":
  process = CustomProcess(ImageDataset("./coco/train"))
  process.run()

import time
from torch.multiprocessing import Process

from image_dataset import ImageDataset
from torch.utils.data import DataLoader


def image_reader():
  pytorch_loader = DataLoader(dataset=ImageDataset("./coco/train"),
                              batch_size=64,
                              shuffle=False,
                              sampler=None,
                              batch_sampler=None,
                              num_workers=4,
                              collate_fn=None,
                              pin_memory=True,
                              drop_last=False,
                              timeout=0,
                              worker_init_fn=None,
                              multiprocessing_context=None)

  read_time = 0
  stime = time.time()
  for _ in pytorch_loader:
    read_time = read_time + 1
    if read_time % 100 == 0:
      etime = time.time()
      print("read time: {}, cost time: {}".format(read_time, etime - stime))
      stime = etime


def run():
  plist = []
  for j in range(5):
    p = Process(target=image_reader)
    p.start()
    plist.append(p)

  for p in plist:
    p.join()


if __name__ == "__main__":
  run()