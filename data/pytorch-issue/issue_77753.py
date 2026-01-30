import time

import torch
import torchvision
import torchvision.transforms as transforms


device = "cpu"

train_set = torchvision.datasets.MNIST(root="./data", download=True, train=True, transform=transforms.ToTensor())
loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)

model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 32, 4, 2, 1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(32, 64, 4, 2, 1),
    torch.nn.ReLU(),
    torch.nn.ConvTranspose2d(64, 32, 4, 2, 1),
    torch.nn.ReLU(),
    torch.nn.ConvTranspose2d(32, 1, 4, 2, 1)
)
model.to(device)

optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())
criterion = torch.nn.MSELoss()

for e in range(30):
    avg_loss = 0.0
    start = time.time()
    for i, (x, _) in enumerate(loader):
        x = x.to(device)
        x_bar = model(x)
        loss = criterion(x_bar, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
    avg_loss /= (i+1)
    end = time.time()

    print(f"epoch={e+1}, loss={avg_loss:.5f}, time={end-start:.3f}")

import time

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, _), _ = mnist.load_data()
x_train = (x_train / 255.0).reshape(-1, 28, 28, 1)

# Add a channels dimension
print(x_train.shape)
train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(256)

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(28, 28, 1,)))
model.add(tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding="same"))

mse = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')


@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        x_bar = model(x)
        loss = mse(x, x_bar)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)


print(model.summary())
for e in range(30):
    start = time.time()
    train_loss.reset_states()
    for x in train_ds:
        train_step(x)
    end = time.time()
    print(f"epoch={e+1}, loss={train_loss.result():.5f}, time={end-start:.3f}")

import time
import torch
import torchvision
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
import torch.optim as optim
from torchvision.models.resnet import ResNet, BasicBlock

BATCH_SZ = 512
MAX_EPOCHS = 10
device = torch.device("mps")

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(64),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_ds = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SZ, shuffle=True, num_workers=0, pin_memory=True)

model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=10).to(device)
optimizer = optim.Adam([*model.parameters()])
criterion = nn.CrossEntropyLoss()

time_intervals = []
step_index = []

step_count = 0
for epoch in range(MAX_EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    avg_loss = []
    start = time.time()
    prev_time = time.time()
    with tqdm(total=len(train_loader), leave=False) as pbar:
        for i, data in enumerate(train_loader, 0):
            pbar.update()
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss.append(loss.item())
            del inputs, labels
            step_count += 1
            if i % 10 == 9:
                pbar.set_description(f"[{epoch + 1}/{MAX_EPOCHS}]: loss {running_loss/10:.5f}")
                pbar.refresh()
                curr_time = time.time()
                time_intervals.append((curr_time - prev_time)/10)
                step_index.append(step_count)
                prev_time = time.time()
                running_loss = 0.0
    end = time.time()
    print(f"epoch={epoch + 1}, loss={np.mean(avg_loss):.5f}, time={end - start:.3f}s")

plt.style.use('ggplot')
plt.figure(figsize=(14, 7))
plt.title("Average step time interval (s)")
plt.plot(step_index, time_intervals)
plt.show()

import time
import torch
import torchvision
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.models.resnet import ResNet, BasicBlock

BATCH_SZ = 512
MAX_EPOCHS = 10
PRINT_ITER = 5

# str_device = "mps" if torch.backends.mps.is_available() else "cpu"

loss_hist = {}
interval_hist = {}

for str_device in ["cpu", "mps"]:
    loss_hist[str_device] = []
    interval_hist[str_device] = []
    
    device = torch.device(str_device)
    print(device)

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(64),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_ds = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SZ, shuffle=True, num_workers=0, pin_memory=True)

    model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=10).to(device)
    model.train()
    optimizer = optim.Adam([*model.parameters()])
    criterion = nn.CrossEntropyLoss()

    time_intervals = []
    step_index = []

    step_count = 0
    for epoch in range(MAX_EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        avg_loss = []
        start = time.time()
        prev_time = time.time()
        with tqdm(total=len(train_loader), leave=False) as pbar:
            for i, data in enumerate(train_loader, 0):
                pbar.update()
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                avg_loss.append(loss.item())
                loss_hist[str_device].append(loss.item())
                del inputs, labels
                step_count += 1
                if i % PRINT_ITER == PRINT_ITER - 1:
                    pbar.set_description(f"[{epoch + 1}/{MAX_EPOCHS}]: loss {running_loss/PRINT_ITER:.5f}")
                    pbar.refresh()
                    curr_time = time.time()
                    time_intervals.append((curr_time - prev_time)/PRINT_ITER)
                    step_index.append(step_count)
                    interval_hist[str_device].append((step_index[-1], time_intervals[-1]))
                    prev_time = time.time()
                    running_loss = 0.0
        end = time.time()
        print(f"epoch={epoch + 1}, loss={np.mean(avg_loss):.5f}, time={end - start:.3f}s")

plt.style.use('ggplot')
plt.figure(figsize=(10, 10))

ax1 = plt.subplot(2, 1, 1)
for str_device in ["cpu", "mps"]:
    x_ticks = [x[0] for x in interval_hist[str_device]]
    y_ticks = [x[1] for x in interval_hist[str_device]]
    ax1.plot(x_ticks, y_ticks, label=str_device)
ax1.legend(loc="best")
ax1.set_title("Average step time interval (s)")

ax2 = plt.subplot(2, 1, 2)
for str_device in ["cpu", "mps"]:
    ax2.plot(loss_hist[str_device], label=str_device)
ax2.legend(loc="best")
ax2.set_title("Loss curve")

plt.tight_layout()
plt.show()