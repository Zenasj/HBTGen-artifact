import torch.nn as nn

import torch as t
import time

print(f'torch version: {t.__version__}')

def main():
    class Net(t.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv3d = t.nn.Conv3d(3, 3, (2, 3, 3))
            
        def forward(self, x):
            x = self.conv3d(x)
            return x
            
    t.cuda.manual_seed(0)
    net = Net().cuda()
    net.eval()

    video = t.empty(16, 3, 40, 320, 320, device=t.device('cuda')).normal_()

    for i in range(10):
        t1 = time.time()
        for j in range(100):
            with t.no_grad():
                output = net(video)
        t2 = time.time()
        print(f'trial #{i}: {t2 - t1}')
    
if __name__ == '__main__':
    main()

import torch as t
import time

print(f'torch version: {t.__version__}')

def main():
    class Net(t.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv2d = t.nn.Conv2d(3, 3, (3, 3))
            
        def forward(self, x):
            x = self.conv2d(x)
            return x
            
    t.cuda.manual_seed(0)
    net = Net().cuda()
    net.eval()

    image = t.empty(16, 3, 320, 320, device=t.device('cuda')).normal_()

    for i in range(20):
        t1 = time.time()
        for j in range(100):
            with t.no_grad():
                output = net(image)
        t2 = time.time()
        print(f'trial #{i}: {t2 - t1}')
    
if __name__ == '__main__':
    main()

import torch as t
t.backends.cudnn.enabled=True
t.backends.cudnn.benchmark=False

import time

print(f'torch version: {t.__version__}')

def main():
    class Net(t.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv3d = t.nn.Conv3d(3, 3, (2, 3, 3))
            
        def forward(self, x):
            x = self.conv3d(x)
            return x
            
    t.cuda.manual_seed(0)
    net = Net().cuda()
    net.eval()

    video = t.empty(16, 3, 40, 320, 320, device=t.device('cuda')).normal_()

    for i in range(5):
        t1 = time.time()
        for j in range(100):
            with t.no_grad():
                net(video)
        t2 = time.time()
        print(f'trial #{i}: {t2 - t1}')
    
    print('start sleeping')
    time.sleep(120)
    
if __name__ == '__main__':
    main()