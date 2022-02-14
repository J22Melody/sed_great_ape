import random
import math
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import itertools
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


DATA_PATH = './data_waveform_1'
MODEL_PATH = './torch_cnn.pt'

batch_size = 256
log_interval = 20
n_epoch = 50
lr = 0.001
step_size = 10
patience = 10
step_factor = 0.3

seed = 42
random.seed(seed)
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()

# # data loading

# number_all = 300

# paths = Path(DATA_PATH).rglob('*.csv')
# paths = list(itertools.islice(paths, number_all))
# print('Split {} files: {}'.format(len(paths), paths))

# data = [np.loadtxt(path, delimiter=',') for path in paths]
# data = np.concatenate(data, axis=0)

# # data split
# random.shuffle(data)

# total_size = len(data)
# train_size = math.floor(total_size*0.8)
# dev_size = math.floor(total_size*0.1)
# test_size = total_size - train_size - dev_size

# dev = torch.Tensor(data[:dev_size])
# test = torch.Tensor(data[dev_size:test_size + dev_size])
# train = torch.Tensor(data[test_size + dev_size:])

# print('Train:', train.size())
# print('Dev:', dev.size())
# print('Test:', test.size())

# number_1 = train[:, 0].sum().item()
# print('Number of 1 samples', number_1)
# ratio_1 = number_1 / len(train)
# print('Ratio of 1 samples', ratio_1)

# data loading

number_all = 300

paths = Path(DATA_PATH).rglob('*.csv')
paths = list(itertools.islice(paths, number_all))
print('Split {} files: {}'.format(len(paths), paths))

data = [np.loadtxt(path, delimiter=',') for path in paths]
random.shuffle(data)

# data split

total_size = len(data)
train_size = math.floor(total_size*0.8)
dev_size = math.floor(total_size*0.1)
test_size = total_size - train_size - dev_size

dev = torch.tensor(np.concatenate(data[:dev_size], axis=0)).float()
test = torch.tensor(np.concatenate(data[dev_size:test_size + dev_size], axis=0)).float()
train = torch.tensor(np.concatenate(data[test_size + dev_size:], axis=0)).float()

print('Train:', train.size())
print('Dev:', dev.size())
print('Test:', test.size())

number_1 = train[:, 0].sum().item()
print('Number of 1 samples', number_1)
ratio_1 = number_1 / len(train)
print('Ratio of 1 samples', ratio_1)

# dataset and dataloader

train_set = TensorDataset(train[:, 1:], train[:, 0])
dev_set = TensorDataset(dev[:, 1:], dev[:, 0])
test_set = TensorDataset(test[:, 1:], test[:, 0])

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

g = torch.Generator()
g.manual_seed(0)
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    generator=g,
)
dev_loader = DataLoader(
    dev_set,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=False,
    generator=g,
)
test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=False,
    generator=g,
)

print("--- %s seconds ---" % (time.time() - start_time))

# model

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

model = M5(n_input=1, n_output=2)
model.to(device)
print(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

n = count_parameters(model)
print("Number of parameters: %s" % n)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=step_factor)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=step_factor, mode='max')

# train, dev, test functions

def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        data = data.view(data.size()[0], 1, -1)
        target = target.to(device).long()

        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        # loss = F.nll_loss(output.squeeze(), target, weight=torch.tensor([1.0, 100.0]))
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def validate(model, epoch):
    model.eval()
    pred_list = []
    target_list = []
    for data, target in dev_loader:

        data = data.to(device)
        data = data.view(data.size()[0], 1, -1)
        target = target.to(device).long()

        output = model(data)

        pred = get_likely_index(output)

        pred_list.append(pred.squeeze())
        target_list.append(target.squeeze())

        # update progress bar
        pbar.update(pbar_update)

    pred = torch.cat(pred_list).to('cpu').numpy()
    target = torch.cat(target_list).to('cpu').numpy()

    accuracy = accuracy_score(pred, target)
    precision = precision_score(pred, target)
    recall = recall_score(pred, target)
    f1 = f1_score(pred, target)

    print(f"Validate Epoch: {epoch} accuracy: {accuracy:.2f} precision: {precision:.2f} recall: {recall:.2f} f1: {f1:.2f}\n")

    return f1

def test(model):
    model.eval()
    pred_list = []
    target_list = []
    for data, target in test_loader:

        data = data.to(device)
        data = data.view(data.size()[0], 1, -1)
        target = target.to(device).long()

        output = model(data)

        pred = get_likely_index(output)

        pred_list.append(pred.squeeze())
        target_list.append(target.squeeze())

    pred = torch.cat(pred_list).to('cpu').numpy()
    target = torch.cat(target_list).to('cpu').numpy()

    accuracy = accuracy_score(pred, target)
    precision = precision_score(pred, target)
    recall = recall_score(pred, target)
    f1 = f1_score(pred, target)

    print(f"Test: accuracy: {accuracy:.2f} precision: {precision:.2f} recall: {recall:.2f} f1: {f1:.2f}\n")

    return f1

# train and save

pbar_update = 1 / (len(train_loader) + len(dev_loader))
losses = []

with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval)
        dev_score = validate(model, epoch)

        scheduler.step()
        
        if dev_score > 0.8:
            torch.save(model.state_dict(), MODEL_PATH)

# Let's plot the training loss versus the number of iteration.
# plt.plot(losses)
# plt.title("training loss")
# plt.show()

# load and test

model = M5(n_input=1, n_output=2)
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH))
test(model)

print("--- %s seconds ---" % (time.time() - start_time))

