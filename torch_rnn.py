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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


DATA_PATH = './data_wav2vec2_1_split'
MODEL_PATH = './torch_rnn.pt'

n_embedding = 768
n_hidden = 16
batch_size = 16
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
        reader = pd.read_csv(self.csv_path, sep=",", header=None, chunksize=batch_size)

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

class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, dropout=0.2)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, input):
        input = input.view(input.size()[0], -1, n_embedding)
        lstm_out, _ = self.lstm(input)
        tag_space = self.hidden2tag(lstm_out)[:, -1]
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

model = LSTM(n_embedding, n_hidden, 2)
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
        # loss = F.nll_loss(output.squeeze(), target, weight=torch.tensor([1.0, 10.0]).to(device))
        loss = F.nll_loss(output.squeeze(), target)
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

    accuracy = accuracy_score(pred, target)
    precision = precision_score(pred, target)
    recall = recall_score(pred, target)
    f1 = f1_score(pred, target)

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

    accuracy = accuracy_score(pred, target)
    precision = precision_score(pred, target)
    recall = recall_score(pred, target)
    f1 = f1_score(pred, target)

    print(f"\nValidate Epoch: {epoch} accuracy: {accuracy:.2f} precision: {precision:.2f} recall: {recall:.2f} f1: {f1:.2f}\n")

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

    print(f"\nTest: accuracy: {accuracy:.2f} precision: {precision:.2f} recall: {recall:.2f} f1: {f1:.2f}\n")

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

model = LSTM(n_embedding, n_hidden, 2)
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH))
test(model)


print("--- %s seconds ---" % (time.time() - start_time))

