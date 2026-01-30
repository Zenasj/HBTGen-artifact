import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import gc

def create_and_save_image():
    image = torch.randn(30, 384, 384)
    ckpt = {"image": image}
    torch.save(ckpt, "test_image.pt")

@profile
def load_and_del():
    ckpt = torch.load("test_image.pt")
    del ckpt
    gc.collect()

if __name__ == '__main__':
    create_and_save_image()
    load_and_del()

for i in range(1_000_000):
  load_and_del()