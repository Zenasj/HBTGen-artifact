model.eval()
logger.warning("Set model to {0} mode".format('train' if model.training else 'eval'))
valid_bleu = evalModel(model, translator, validData)
model.train()
logger.warning("Set model to {0} mode".format('train' if model.training else 'eval'))
logger.info('Validation Score: %g' % (valid_bleu * 100))
if valid_bleu >= optim.best_metric:
    saveModel(valid_bleu)

if model is None:
    checkpoint = torch.load(opt.model)

    model_opt = checkpoint['opt']
    self.src_dict = checkpoint['dicts']['src']
    self.tgt_dict = checkpoint['dicts']['tgt']

    self.enc_rnn_size = model_opt.enc_rnn_size
    self.dec_rnn_size = model_opt.dec_rnn_size
    model = some_model()

    model.load_state_dict(checkpoint['model'])
    generator.load_state_dict(checkpoint['generator'])

    if opt.cuda:
        model.cuda()
        generator.cuda()
    else:
        model.cpu()
        generator.cpu()

    model.generator = generator
else:
    self.src_dict = dataset['dicts']['src']
    self.tgt_dict = dataset['dicts']['tgt']

    self.enc_rnn_size = opt.enc_rnn_size
    self.dec_rnn_size = opt.dec_rnn_size
    self.opt.cuda = True if len(opt.gpus) >= 1 else False
    self.opt.n_best = 1
    self.opt.replace_unk = False

self.model = model
self.model.eval()

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
import shutil

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume-type', default='', type=str, metavar='PATH',
                    help='load from best/last checkpoint')
parser.add_argument('--exp_name', type=str, default='dummy',
                    help='experiment name to used across everything', metavar='exp_name', required=True)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if args.resume_type:
    checkpoint_file = 'weights/' + args.exp_name + '/'
    checkpoint_file += 'checkpoint.pth.tar' if args.resume_type == 'last' else 'model_best.pth.tar'
    if os.path.isfile(checkpoint_file):
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_file, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_file))


def save_checkpoint(state, is_best):
    exp_weights_root_dir = 'weights/' + args.exp_name + '/'
    os.makedirs(exp_weights_root_dir, exist_ok=True)
    filename = exp_weights_root_dir + 'checkpoint.pth.tar'
    torch.save(state, filename)
    if is_best:
        print('best beaten')
        shutil.copyfile(filename, exp_weights_root_dir + 'model_best.pth.tar')


for epoch in range(args.start_epoch, args.epochs + args.start_epoch):
    train(epoch)
    test()
    save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_loss': 0,
                        'optimizer': optimizer.state_dict(),
                    }, True)