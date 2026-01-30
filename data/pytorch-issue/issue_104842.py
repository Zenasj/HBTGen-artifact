# -*- coding: utf-8 -*-

import torch
import math
import time 


class PolynomialRegression:
    def __init__(self, degree=3, learning_rate=1e-6, device=torch.device("cpu")):
        self.degree = degree
        self.learning_rate = learning_rate
        self.device = device
        self.dtype = torch.float
        self.a = torch.randn((), device=self.device, dtype=self.dtype)
        self.b = torch.randn((), device=self.device, dtype=self.dtype)
        self.c = torch.randn((), device=self.device, dtype=self.dtype)
        self.d = torch.randn((), device=self.device, dtype=self.dtype)

    def forward(self, x):
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

    def train(self, x, y, num_epochs=2000):
        for t in range(num_epochs):
            # Forward pass: compute predicted y
            y_pred = self.forward(x)

            # Compute and print loss
            loss = (y_pred - y).pow(2).sum().item()
            if t % 100 == 99:
                print(t, loss)

            # Backprop to compute gradients of a, b, c, d with respect to loss
            grad_y_pred = 2.0 * (y_pred - y)
            grad_a = grad_y_pred.sum()
            grad_b = (grad_y_pred * x).sum()
            grad_c = (grad_y_pred * x ** 2).sum()
            grad_d = (grad_y_pred * x ** 3).sum()

            # Update weights using gradient descent
            self.a -= self.learning_rate * grad_a
            self.b -= self.learning_rate * grad_b
            self.c -= self.learning_rate * grad_c
            self.d -= self.learning_rate * grad_d

        print(f'Result: y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3')



def test(device):
    x = torch.linspace(-math.pi, math.pi, 2000, device=torch.device(str(device)), dtype=torch.float)
    y = torch.sin(x)

    model = PolynomialRegression(degree=3, learning_rate=1e-6, device=torch.device(str(device)))
    start_time = time.time()
    model.train(x, y, num_epochs=2000)
    end_time = time.time()

    print(f"✅ Training took {int((end_time - start_time) * 1000000)} microseconds")



if __name__ == '__main__':
    
    test("cpu")
    test("mps")