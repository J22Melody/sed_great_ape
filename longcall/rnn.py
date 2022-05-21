import random
import math
import time
from xml.etree.ElementInclude import include
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

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

n_embedding = 768
n_hidden = 256
batch_size = 1
log_interval = 1
n_epoch = 100
lr = 0.001
dropout = 0.2
patience = 10
step_factor = 0.7
n_class = 7
include_unknown = False

seed = 42
random.seed(seed)
torch.manual_seed(0)
np.random.seed(0)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)
start_time = time.time()

DATA_PATH = './data_wav2vec2'
# DATA_PATH = '../../data/data_wav2vec2'
MODEL_PATH = './rnn_unknown.pt' if include_unknown else './rnn.pt'

# read data

class MyIterableDataset(IterableDataset):
    def __init__(self, data, repeat=1):
        super(MyIterableDataset).__init__()
        self.data = data
        self.repeat = repeat

    def __iter__(self):
        for i in range(self.repeat):
            for item in data:
                yield torch.tensor(item[:, 1:]).float(), torch.tensor(item[:, 0]).float()

print('Reading data ...')

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

g = torch.Generator()
g.manual_seed(0)

paths = Path(DATA_PATH).rglob('*.csv')
paths = list(itertools.islice(paths, 1000))
print('Read {} files: {}'.format(len(paths), paths))

data_with_files = []
for path in paths:
    item = np.loadtxt(path, delimiter=',')
    if include_unknown:
        data_with_files.append((path, item))
    else:
        known_class_samples = item[item[:, 0] > 1]
        if known_class_samples.shape[0] > 0:
            data_with_files.append((path, item))

random.shuffle(data_with_files)

data = [d[1] for d in data_with_files]

print('Train on {} annotated files'.format(len(data)))

data_all = np.concatenate(data)
print(data_all.shape)
unique, counts = np.unique(data_all[:, 0], return_counts=True)
print('Labels in data: ', dict(zip(unique, counts)))

if include_unknown:
    train_index = int(len(data) * 0.8)
    dev_index = int(len(data) * 0.9)
else:
    train_index = 11
    dev_index = 12
train = data[:train_index]
dev = data[train_index:dev_index]
test = data[dev_index:]

print('train, dev, test number of files: {}, {}, {}.'.format(len(train), len(dev), len(test)))
print('dev files: ', [d[0] for d in data_with_files[train_index:dev_index]])
print('test files: ', [d[0] for d in data_with_files[dev_index:]])

train_set = MyIterableDataset(train, repeat=20)
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    generator=g,
)

dev_set = MyIterableDataset(dev)
dev_loader = DataLoader(
    dev_set,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=False,
    generator=g,
)

test_set = MyIterableDataset(test)
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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, dropout=dropout, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

model = LSTM(n_embedding, n_hidden, n_class)
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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(model, epoch, log_interval):
    model.train()
    pred_list = []
    target_list = []
    losses = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device).long()

        output = model(data)

        # negative log-likelihood for a tensor of size (batch x m x n_output)
        weight = torch.tensor([1.0, 0.1, 10.0, 100.0, 20.0, 20.0, 1000.0]).to(device)
        loss = F.nll_loss(output.squeeze(), target.squeeze(), weight=weight)
        # loss = F.nll_loss(output.squeeze(), target.squeeze())
        pred = get_likely_index(output)

        pred_list.append(pred.squeeze())
        target_list.append(target.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        # if batch_idx % log_interval == 0:
        #     print(f"Train Epoch: {epoch} loss: {loss.item():.6f}")

        # record loss
        losses.append(loss.item())

    pred = torch.cat(pred_list).to('cpu').detach().numpy()
    target = torch.cat(target_list).to('cpu').detach().numpy()

    accuracy = accuracy_score(target, pred)
    precision = precision_score(target, pred, average=None, zero_division=1)
    recall = recall_score(target, pred, average=None, zero_division=1)
    f1 = f1_score(target, pred, average=None, zero_division=1)
    f1_avg = f1_score(target, pred, average='weighted', zero_division=1)

    print(f"\nTrain Epoch: {epoch} accuracy: {accuracy:.2f} \n precision: {precision} \n recall: {recall} \n f1: {f1} \n f1_avg: {f1_avg}\n")

    # TensorBoard
    writer.add_scalar("loss/train", sum(losses), epoch)
    writer.add_scalar("accuracy/train", accuracy, epoch)
    writer.add_scalar("f1_avg/train", f1_avg, epoch)
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        writer.add_scalar("precision/class {}/train".format(i), p, epoch)
        writer.add_scalar("recall/class {}/train".format(i), r, epoch)
        writer.add_scalar("f1/class {}/train".format(i), f, epoch)
    writer.add_scalar("learning_rate/train", get_lr(optimizer), epoch)

def validate(model, epoch):
    model.eval()
    pred_list = []
    target_list = []

    for data, target in dev_loader:
        data = data.to(device)
        target = target.to(device).long()

        output = model(data)

        pred = get_likely_index(output)

        pred_list.append(pred.squeeze())
        target_list.append(target.squeeze())

    pred = torch.cat(pred_list).to('cpu').detach().numpy()
    target = torch.cat(target_list).to('cpu').detach().numpy()

    accuracy = accuracy_score(target, pred)
    precision = precision_score(target, pred, average=None, zero_division=1)
    recall = recall_score(target, pred, average=None, zero_division=1)
    f1 = f1_score(target, pred, average=None, zero_division=1)
    f1_avg = f1_score(target, pred, average='weighted', zero_division=1)

    print(f"\nValidation Epoch: {epoch} accuracy: {accuracy:.2f} \n precision: {precision} \n recall: {recall} \n f1: {f1} \n f1_avg: {f1_avg}\n")

    # TensorBoard
    writer.add_scalar("accuracy/dev", accuracy, epoch)
    writer.add_scalar("f1_avg/dev", f1_avg, epoch)
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        writer.add_scalar("precision/class {}/dev".format(i), p, epoch)
        writer.add_scalar("recall/class {}/dev".format(i), r, epoch)
        writer.add_scalar("f1/class {}/dev".format(i), f, epoch)

    return f1_avg

def test(model):
    model.eval()
    pred_list = []
    target_list = []
    output_list = []

    for data, target in test_loader:
        data = data.to(device)
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
    precision = precision_score(target, pred, average=None, zero_division=1)
    recall = recall_score(target, pred, average=None, zero_division=1)
    f1 = f1_score(target, pred, average=None, zero_division=1)
    f1_avg = f1_score(target, pred, average='weighted', zero_division=1)

    np.savetxt('./rnn.target.txt', target, delimiter=',')
    np.savetxt('./rnn.pred.txt', pred, delimiter=',')

    print(f"\nTest Epoch: accuracy: {accuracy:.2f} \n precision: {precision} \n recall: {recall} \n f1: {f1} \n f1_avg: {f1_avg}\n")
    
    print("--- %s seconds ---" % (time.time() - start_time))

# train and save
# best_val_score = 0
# for epoch in range(1, n_epoch + 1):
#     train(model, epoch, log_interval)

#     val_score = validate(model, epoch)
#     # scheduler.step()
#     scheduler.step(val_score)
#     writer.flush()

#     if val_score > best_val_score:
#         print('New best val score, save model ...')
#         torch.save(model.state_dict(), MODEL_PATH)
#         best_val_score = val_score

#     print("--- %s seconds ---" % (time.time() - start_time))

# writer.close()

# load and test
model = LSTM(n_embedding, n_hidden, n_class)
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
test(model)
