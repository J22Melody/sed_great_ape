import random
import math
import datetime
import time
import numpy as np
import pandas as pd
import itertools
from pathlib import Path
import json
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from torch.utils.tensorboard import SummaryWriter


start_time = time.time()

# seed
seed = 42
random.seed(seed)
torch.manual_seed(0)
np.random.seed(0)

# config
parser = ArgumentParser()
parser.add_argument("-c", "--config")
args = parser.parse_args()

CONFIG = json.load(open(args.config))

print(CONFIG)

DATA_PATH = CONFIG['data_path']
RESULT_PATH =  './models/{}/results'.format(CONFIG['name'])
MODEL_PATH = './models/{}/model.pt'.format(CONFIG['name'])

writer = SummaryWriter(log_dir='runs/{}/{}/'.format(CONFIG['name'], datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# The MPS backend is supported on MacOS 12.3+
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.has_mps else 'cpu'))
# device = torch.device('cpu')
print(device)

# read data

class MyIterableDataset(IterableDataset):
    def __init__(self, data, repeat=1, training=False):
        super(MyIterableDataset).__init__()

        # bucketing
        if training:
            data = list(sorted(data, key=lambda d: -d[1].shape[0]))

        self.data = data
        self.repeat = repeat
        self.training = training

    def __iter__(self):
        for _ in range(self.repeat):
            for filename, item in self.data:
                if self.training:
                    yield torch.tensor(item).float()
                else:
                    yield torch.tensor(item[:, 1:]).float(), torch.tensor(item[:, 0]).float(), str(filename)
                

print('Reading data ...')

if device == "cuda" or device == "mps":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

g = torch.Generator()
g.manual_seed(seed)

train = np.load('{}/train.npy'.format(DATA_PATH), allow_pickle=True)
dev = np.load('{}/dev.npy'.format(DATA_PATH), allow_pickle=True)
test = np.load('{}/test.npy'.format(DATA_PATH), allow_pickle=True)

print('train, dev, test number of files: {}, {}, {}.'.format(len(train), len(dev), len(test)))
print('dev files: ', [d[0] for d in dev])
print('test files: ', [d[0] for d in test])

train_data_all = np.concatenate([d[1] for d in train])
print('train_data_all:', train_data_all.shape)
unique, counts = np.unique(train_data_all[:, 0], return_counts=True)
num_classes = dict(zip(unique, counts))
print('Labels in train_data_all: ', num_classes)

def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask

def collate_fn(batch):
    batch_padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([t.shape[0] for t in batch]).to(device)
    return batch_padded, lengths, length_to_mask(lengths)

train_set = MyIterableDataset(train, training=True)
dev_set = MyIterableDataset(dev)
test_set = MyIterableDataset(test)

train_loader = DataLoader(
    train_set,
    batch_size=CONFIG['batch_size'],
    num_workers=num_workers,
    pin_memory=pin_memory,
    generator=g,
    collate_fn=collate_fn,
)
dev_loader = DataLoader(
    dev_set,
    batch_size=1,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=False,
    generator=g,
)
test_loader = DataLoader(
    test_set,
    batch_size=1,
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

class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, nlayers, tagset_size, dropout):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=nlayers, dropout=dropout, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def init_model(model_type):
    if model_type == 'transformer':
        model = Transformer(CONFIG['n_embedding'], CONFIG['nhead'], CONFIG['d_hid'], CONFIG['nlayers'], len(num_classes), CONFIG['dropout'])
    elif model_type == 'lstm':
        model = LSTM(CONFIG['n_embedding'], CONFIG['d_hid'], CONFIG['nlayers'], len(num_classes), CONFIG['dropout'])
    model.to(device)
    return model

model = init_model(CONFIG['model_type'])
print(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

n = count_parameters(model)
print("Number of parameters: %s" % n)

optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=0.0001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=step_factor)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=CONFIG['patience'], factor=CONFIG['step_factor'], mode='max')

# train, dev, test functions

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(model, epoch):
    model.train()
    pred_list = []
    target_list = []
    losses = []

    for data, lengths, mask in train_loader:
        input = data[:, :, 1:].to(device)
        target = data[:, :, 0].to(device).long()

        output = model(input)

        target = target[mask]
        output = output[mask]

        weight = None
        if CONFIG['balance_weights']:
            weight_num = [1 / v for v in num_classes.values()]
            weight = torch.tensor(weight_num).float().to(device) # inverse to num training samples

        # loss = F.nll_loss(output.transpose(1, 2), target, weight=weight)
        loss = F.nll_loss(output, target, weight=weight)
        pred = get_likely_index(output)

        pred_list.append(torch.flatten(pred))
        target_list.append(torch.flatten(target))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

    for data, target, _ in dev_loader:
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
    if CONFIG['test_only']:
        Path(RESULT_PATH).mkdir(parents=True, exist_ok=True)

    model.eval()
    pred_list = []
    target_list = []
    output_list = []

    loader = dev_loader if use_dev else test_loader
    for data, target, filename in loader:
        data = data.to(device)
        target = target.to(device).long().squeeze()

        output = model(data)

        pred = get_likely_index(output).squeeze()

        pred_list.append(pred)
        target_list.append(target)
        output_list.append(output.squeeze())

        filename = filename[0]

        if CONFIG['test_only']:
            np.savetxt('./{}/{}.target.txt'.format(RESULT_PATH, filename), target.to('cpu').numpy(), delimiter=',')
            np.savetxt('./{}/{}.pred.txt'.format(RESULT_PATH, filename), pred.to('cpu').numpy(), delimiter=',')

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

if not CONFIG['test_only']:
    # train and save
    best_val_score = 0
    for epoch in range(1, CONFIG['n_epoch'] + 1):
        train(model, epoch)

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
model = init_model(CONFIG['model_type'])
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

test(model)
test(model, use_dev=True)
