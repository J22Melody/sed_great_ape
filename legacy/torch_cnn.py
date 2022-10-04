import random
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, IterableDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, PrecisionRecallDisplay

# DATA_PATH = '../data/data_waveform_1_split'
DATA_PATH = './data_waveform_1_split'
MODEL_PATH = './torch_cnn.pt'

batch_size = 256
log_interval = 20
n_epoch = 100
lr = 0.001
step_size = 10
patience = 7
step_factor = 0.3

seed = 42
random.seed(seed)
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
start_time = time.time()

# read data

class MyIterableDataset(IterableDataset):
    def __init__(self, csv_path):
        super(MyIterableDataset).__init__()
        self.csv_path = csv_path

    def __iter__(self):
        reader = pd.read_csv(self.csv_path, sep=",", header=None, chunksize=5000)

        for chunk in reader:
            chunk_arr = chunk.to_numpy()
            for i in range(chunk_arr.shape[0]):
                yield torch.tensor(chunk_arr[i, 1:]).float(), torch.tensor(chunk_arr[i, 0]).float()

print('Reading data ...')

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

g = torch.Generator()
g.manual_seed(0)

# train = torch.Tensor(np.loadtxt('{}/train.csv'.format(DATA_PATH), delimiter=','))
# print('Train:', train.size())
# train_set = TensorDataset(train[:, 1:], train[:, 0])
train_set = MyIterableDataset('{}/train.csv'.format(DATA_PATH))
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    generator=g,
)

# dev = torch.Tensor(np.loadtxt('{}/dev.csv'.format(DATA_PATH), delimiter=','))
# print('Dev:', dev.size())
# dev_set = TensorDataset(dev[:, 1:], dev[:, 0])
dev_set = MyIterableDataset('{}/dev.csv'.format(DATA_PATH))
dev_loader = DataLoader(
    dev_set,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=False,
    generator=g,
)

# test = torch.Tensor(np.loadtxt('{}/test.csv'.format(DATA_PATH), delimiter=','))
# print('Test:', test.size())
# test_set = TensorDataset(test[:, 1:], test[:, 0])
test_set = MyIterableDataset('{}/test.csv'.format(DATA_PATH))
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
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=step_factor)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=step_factor, mode='max')

# train, dev, test functions

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def train(model, epoch, log_interval):
    model.train()
    pred_list = []
    target_list = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        data = data.view(data.size()[0], 1, -1)
        target = target.to(device).long()

        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target, weight=torch.tensor([1.0, 10.0]).to(device))
        # loss = F.nll_loss(output.squeeze(), target)
        pred = get_likely_index(output)

        pred_list.append(pred.squeeze())
        target_list.append(target.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} loss: {loss.item():.6f}")

        # record loss
        losses.append(loss.item())
    
    pred = torch.cat(pred_list).to('cpu').detach().numpy()
    target = torch.cat(target_list).to('cpu').detach().numpy()

    accuracy = accuracy_score(target, pred)
    precision = precision_score(target, pred)
    recall = recall_score(target, pred)
    f1 = f1_score(target, pred)

    print(f"\nTrain Epoch: {epoch} accuracy: {accuracy:.2f} precision: {precision:.2f} recall: {recall:.2f} f1: {f1:.2f}\n")

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

    pred = torch.cat(pred_list).to('cpu').numpy()
    target = torch.cat(target_list).to('cpu').numpy()

    accuracy = accuracy_score(target, pred)
    precision = precision_score(target, pred)
    recall = recall_score(target, pred)
    f1 = f1_score(target, pred)

    print(f"\nValidate Epoch: {epoch} accuracy: {accuracy:.2f} precision: {precision:.2f} recall: {recall:.2f} f1: {f1:.2f}\n")

    return f1

def test(model):
    model.eval()
    pred_list = []
    target_list = []
    output_list = []
    for data, target in test_loader:

        data = data.to(device)
        data = data.view(data.size()[0], 1, -1)
        target = target.to(device).long()

        output = model(data)

        pred = get_likely_index(output)

        pred_list.append(pred.squeeze())
        target_list.append(target.squeeze())
        output_list.append(output.squeeze())

    pred = torch.cat(pred_list).to('cpu').numpy()
    target = torch.cat(target_list).to('cpu').numpy()
    output = torch.cat(output_list).detach().to('cpu').numpy()

    accuracy = accuracy_score(target, pred)
    precision = precision_score(target, pred)
    recall = recall_score(target, pred)
    f1 = f1_score(target, pred)

    print(f"\nTest: accuracy: {accuracy:.2f} precision: {precision:.2f} recall: {recall:.2f} f1: {f1:.2f}\n")
    
    print("--- %s seconds ---" % (time.time() - start_time))

    probs = np.exp(output)
    display = PrecisionRecallDisplay.from_predictions(target, probs[:, 1], pos_label=1)
    _ = display.ax_.set_title("2-class Precision-Recall curve")
    plt.show()

# train and save

best_val_score = 0
losses = []
for epoch in range(1, n_epoch + 1):
    train(model, epoch, log_interval)

    val_score = validate(model, epoch)
    # scheduler.step()
    scheduler.step(val_score)

    if val_score > best_val_score:
        print('New best val score, save model ...')
        torch.save(model.state_dict(), MODEL_PATH)
        best_val_score = val_score

    print("--- %s seconds ---" % (time.time() - start_time))

# # Let's plot the training loss versus the number of iteration.
# plt.plot(losses)
# plt.title("training loss")
# plt.show()

# load and test

model = M5(n_input=1, n_output=2)
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH))
test(model)


