import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
import time
import pdb

def hv(loss, model, v):
	s = time.time()
	grad = autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
	e1 = time.time()
	Hv = autograd.grad(grad, model.parameters(), grad_outputs=v, retain_graph=True)
	e2 = time.time()
	print('1st back prop: {} sec. 2nd back prop: {} sec'.format(e1-s, e2-e1))
	return Hv

class CNN_ReLU(nn.Module):
	def __init__(self):
		super(CNN_ReLU, self).__init__()

		self.conv1 = nn.Sequential(
									nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
									nn.BatchNorm2d(32)
									)
									# N x 32 x 32 x 32

		self.conv2 = nn.Sequential(
									nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
									nn.BatchNorm2d(64)
									)
									# N x 64 x 32 x 32

		self.conv3 = nn.Sequential(
									nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
									nn.BatchNorm2d(64),
									nn.MaxPool2d(kernel_size=2, stride=None)
									)
									# N x 128 x 16 x 16

		self.conv4 = nn.Sequential(
									nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
									nn.BatchNorm2d(64),
									nn.MaxPool2d(kernel_size=2, stride=None)
									)
									# N x 64 x 8 x 8

		self.FC_1 = nn.Linear(4096, 1024)
		self.FC_2 = nn.Linear(1024, 256)
		self.FC_3 = nn.Linear(256, 10)

	def forward(self, x):
		N = x.shape[0]

		x = self.conv1(x)
		x = F.relu(x)

		x = self.conv2(x)
		x = F.relu(x)

		x = self.conv3(x)
		x = F.relu(x)		

		x = self.conv4(x)
		x = F.relu(x)

		x = x.view(N, -1)

		x = self.FC_1(x)
		x = F.relu(x)

		x = self.FC_2(x)
		x = F.relu(x)

		out = self.FC_3(x)

		return out

if __name__ == '__main__':
	
	use_gpu = torch.cuda.is_available()
	model = CNN_ReLU().cuda() if use_gpu else CNN_ReLU()
	criterion = nn.CrossEntropyLoss()
	images = torch.randn(256, 3, 32, 32)
	targets = torch.randint(0, 9, (256, ))

	if use_gpu:
		targets = targets.cuda()
		images = images.cuda()

	outputs = (model(images))
	loss = criterion(outputs, targets)
			
	v = [torch.ones_like(p, requires_grad=True) for p in model.parameters()]

	end2 = time.time()
	hv_ = hv(loss, model, v)
	end3 = time.time()
	torch.autograd.grad(loss, model.parameters(), retain_graph=True)
	end4 = time.time()

	print('time: Hv {} sec| grad {} sec'.format(end3-end2, end4-end3))

import tensorflow as tf

def hessian_vec_bk(ys, xs, vs, grads=None):
  """Implements Hessian vector product using backward on backward AD.
  Args:
    ys: Loss function.
    xs: Weights, list of tensors.
    vs: List of tensors to multiply, for each weight tensor.
  Returns:
    Hv: Hessian vector product, same size, same shape as xs.
  """
  # Validate the input
  if type(xs) == list:
    if len(vs) != len(xs):
      raise ValueError("xs and vs must have the same length.")

  if grads is None:
    grads = tf.gradients(ys, xs, gate_gradients=True)
  return tf.gradients(grads, xs, vs, gate_gradients=True)