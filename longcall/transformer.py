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
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, IterableDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, PrecisionRecallDisplay

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

n_embedding = 768
d_hid = 2048  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 6  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8  # number of heads in nn.MultiheadAttention
batch_size = 1
log_interval = 1
n_epoch = 100
lr = 0.0001
dropout = 0.4
patience = 10
step_factor = 0.7
n_class = 7
include_unknown = False

seed = 42
random.seed(seed)
torch.manual_seed(0)
np.random.seed(0)

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
start_time = time.time()

DATA_PATH = './data_wav2vec2'
DATA_PATH = '../../data/data_wav2vec2'
MODEL_PATH = 'transformer.pt'

# read data

class MyIterableDataset(IterableDataset):
    def __init__(self, data, repeat=1):
        super(MyIterableDataset).__init__()
        self.data = data
        self.repeat = repeat

    def __iter__(self):
        for i in range(self.repeat):
            for item in self.data:
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

train = np.load('{}/train.npy'.format(DATA_PATH), allow_pickle=True)
dev = np.load('{}/dev.npy'.format(DATA_PATH), allow_pickle=True)
test = np.load('{}/test.npy'.format(DATA_PATH), allow_pickle=True)

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

# test = [np.loadtxt(DATA_PATH + '/4T10lcFugit.csv', delimiter=',')]
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

class Transformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, nclass: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, nclass)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        tag_scores = F.log_softmax(output, dim=1)
        return tag_scores

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

model = Transformer(n_embedding, nhead, d_hid, nlayers, n_class, dropout)
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
        weight = torch.tensor([1.0, 0.00001, 10.0, 100.0, 20.0, 20.0, 1000.0]).to(device)
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

def test(model, use_dev=False):
    model.eval()
    pred_list = []
    target_list = []
    output_list = []

    loader = dev_loader if use_dev else test_loader
    for data, target in loader:
        data = data.to(device)
        target = target.to(device).long().squeeze()

        output = model(data)

        pred = get_likely_index(output).squeeze()

        pred_list.append(pred)
        target_list.append(target)
        output_list.append(output.squeeze())

        filename = 'dev' if use_dev else 'test'
        np.savetxt('./transformer_results/{}.target.txt'.format(filename), target.to('cpu').numpy(), delimiter=',')
        np.savetxt('./transformer_results/{}.pred.txt'.format(filename), pred.to('cpu').numpy(), delimiter=',')

    pred = torch.cat(pred_list).to('cpu').numpy()
    target = torch.cat(target_list).to('cpu').numpy()
    output = torch.cat(output_list).detach().to('cpu').numpy()

    accuracy = accuracy_score(target, pred)
    precision = precision_score(target, pred, average=None, zero_division=1)
    recall = recall_score(target, pred, average=None, zero_division=1)
    f1 = f1_score(target, pred, average=None, zero_division=1)
    f1_avg = f1_score(target, pred, average='weighted', zero_division=1)

    print(f"\nTest Epoch: accuracy: {accuracy:.2f} \n precision: {precision} \n recall: {recall} \n f1: {f1} \n f1_avg: {f1_avg}\n")
    
    print("--- %s seconds ---" % (time.time() - start_time))

# train and save
best_val_score = 0
for epoch in range(1, n_epoch + 1):
    train(model, epoch, log_interval)

    val_score = validate(model, epoch)
    # scheduler.step()
    scheduler.step(val_score)
    writer.flush()

    if val_score > best_val_score:
        print('New best val score, save model ...')
        torch.save(model.state_dict(), MODEL_PATH)
        best_val_score = val_score

    print("--- %s seconds ---" % (time.time() - start_time))
writer.close()

# load and test
model = Transformer(n_embedding, nhead, d_hid, nlayers, n_class, dropout)
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
test(model)
test(model, use_dev=True)
