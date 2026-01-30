import torch
import torchvision
from torchvision import transforms as tf

dtype = torch.float32
device = torch.device("cuda:0")

transforms = tf.Compose([tf.ToTensor()])

mnist_data = torchvision.datasets.MNIST("./mnist", transform=transforms, download=True)
batch_size = 500
data_loader = torch.utils.data.DataLoader(mnist_data,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=1)

first_batch = next(data_loader.__iter__())