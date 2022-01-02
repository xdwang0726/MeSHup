import argparse
import logging
import os
import random

import dgl
import ijson
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtext.vocab import Vectors
from tqdm import tqdm

from model import multichannel_GCN
from pytorchtools import EarlyStopping
from util import *
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import ngrams_iterator
from torchtext.vocab import build_vocab_from_iterator

from util import _create_data_from_csv, _RawTextIterableDataset, _create_data_from_csv_vocab


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # cudnn


def _vocab_iterator(all_text, ngrams=1):

    tokenizer = get_tokenizer('basic_english')

    for i, text in enumerate(all_text):
        texts_pre_article = ' '.join(text.values())
        texts = tokenizer(texts_pre_article)
        texts = text_clean(texts)
        yield ngrams_iterator(texts, ngrams)


def weight_matrix(vocab, vectors, dim=200):
    weight_matrix = np.zeros([len(vocab.itos), dim])
    for i, token in enumerate(vocab.stoi):
        try:
            weight_matrix[i] = vectors.__getitem__(token)
        except KeyError:
            weight_matrix[i] = np.random.normal(scale=0.5, size=(dim,))
    return torch.from_numpy(weight_matrix)


def generate_batch(batch):
    """
    Output:
        text: the text entries in the data_batch are packed into a list and
            concatenated as a single tensor for the input of nn.EmbeddingBag.
        cls: a tensor saving the labels of individual text entries.
    """
    label = []
    for entry in batch:
        l = entry[0].replace('[', '')
        l = l.replace(']', '')
        l = l.replace("'", '')
        l = l.split(',')
        label.append(l)

    title_abstract = [torch.tensor(convert_text_tokens(entry[1])) for entry in batch]
    title_abstract = pad_sequence(title_abstract, ksz=3, batch_first=True)

    intro = [torch.tensor(convert_text_tokens(entry[2])) for entry in batch]
    intro = pad_sequence(intro, ksz=3, batch_first=True)

    method = [torch.tensor(convert_text_tokens(entry[3])) for entry in batch]
    method = pad_sequence(method, ksz=3, batch_first=True)

    result = [torch.tensor(convert_text_tokens(entry[4])) for entry in batch]
    result = pad_sequence(result, ksz=3, batch_first=True)

    discuss = [torch.tensor(convert_text_tokens(entry[5])) for entry in batch]
    discuss = pad_sequence(discuss, ksz=3, batch_first=True)

    return label, title_abstract, intro, method, result, discuss


def train(train_dataset, valid_dataset, model, mlb, G, batch_sz, num_epochs, criterion, device, num_workers, optimizer,
          lr_scheduler):

    train_data = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, collate_fn=generate_batch,
                            num_workers=num_workers, pin_memory=True)

    valid_data = DataLoader(valid_dataset, batch_size=batch_sz, shuffle=True, collate_fn=generate_batch,
                            num_workers=num_workers, pin_memory=True)

    print('train', len(train_data.dataset))
    # num_lines = num_epochs * len(train_data)

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    early_stopping = EarlyStopping(patience=3, verbose=True)

    print("Training....")
    for epoch in range(num_epochs):
        model.train()  # prep model for training
        for i, (label, abstract, intro, method, results, discuss) in enumerate(train_data):
            label = torch.from_numpy(mlb.fit_transform(label)).type(torch.float)
            label = label.to(device)

            abstract, intro, method, results, discuss = abstract.to(device), intro.to(device), method.to(device), results.to(device), discuss.to(device)
            G, G.ndata['feat'] = G.to(device), G.ndata['feat'].to(device)

            output = model(abstract, intro, method, results, discuss, G, G.ndata['feat'])
            loss = criterion(output, label)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None
            train_losses.append(loss.item())  # record training loss

        # Adjust the learning rate
        lr_scheduler.step()

        with torch.no_grad():
            model.eval()
            for i, (label, abstract, intro, method, results, discuss) in enumerate(valid_data):
                label = torch.from_numpy(mlb.fit_transform(label)).type(torch.float)
                label = label.to(device)

                abstract, intro, method, results, discuss = abstract.to(device), intro.to(device), method.to(device), results.to(device), discuss.to(device)
                G, G.ndata['feat'] = G.to(device), G.ndata['feat'].to(device)

                output = model(abstract, intro, method, results, discuss, G, G.ndata['feat'])

                loss = criterion(output, label)
                valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        # print('[{} / {}] Train Loss: {.5f}, Valid Loss: {.5f}'.format(epoch+1, num_epochs, train_loss, valid_loss))
        epoch_len = len(str(num_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    return model, avg_train_losses, avg_valid_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path')
    parser.add_argument('--dev_path')
    parser.add_argument('--test_path')
    parser.add_argument('--word2vec_path')
    parser.add_argument('--meSH_pair_path')
    parser.add_argument('--graph')
    parser.add_argument('--save-model-path')

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--ksz', default=3)

    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_sz', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--scheduler_step_sz', type=int, default=2)
    parser.add_argument('--lr_gamma', type=float, default=0.9)

    parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:3456')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')

    args = parser.parse_args()
    set_seed(0)
    torch.backends.cudnn.benchmark = True
    n_gpu = torch.cuda.device_count()  # check if it is multiple gpu
    print('{} gpu is avaliable'.format(n_gpu))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print('Device:{}'.format(device))

    # NUM_LINES = {
    #     'all': 957426,
    #     'train': 765920,
    #     'dev': 95737,
    #     'test': 95769
    # }
    NUM_LINES = {
        'all': 1000,
        'train': 70,
        'dev': 20,
        'test': 95769
    }
    print('load and prepare Mesh')
    # read full MeSH ID list
    mapping_id = {}
    with open(args.meSH_pair_path, 'r') as f:
        for line in f:
            (key, value) = line.split('=')
            mapping_id[key] = value.strip()

    meshIDs = list(mapping_id.values())
    print('Total number of labels %d' % len(meshIDs))
    index_dic = {k: v for v, k in enumerate(meshIDs)}
    mesh_index = list(index_dic.values())
    mlb = MultiLabelBinarizer(classes=mesh_index)
    mlb.fit(mesh_index)
    num_nodes = len(meshIDs)

    print('load pre-trained BioWord2Vec')
    vocab_iterator = _RawTextIterableDataset(NUM_LINES['all'], _create_data_from_csv_vocab(args.train_path))
    cache, name = os.path.split(args.word2vec_path)
    vectors = Vectors(name=name, cache=cache)
    vocab = build_vocab_from_iterator(yield_tokens(vocab_iterator))
    vocab_size = len(vocab)

    print('Load graph')
    G = dgl.load_graphs(args.graph)[0][0]
    print('graph', G.ndata['feat'].shape)

    train_iterator = _RawTextIterableDataset(NUM_LINES, _create_data_from_csv(args.train_path))
    dev_iterator = _RawTextIterableDataset(NUM_LINES, _create_data_from_csv(args.dev_path))
    train_dataset = to_map_style_dataset(train_iterator)
    dev_dataset = to_map_style_dataset(dev_iterator)
    model = multichannel_GCN(vocab_size, args.dropout, args.ksz, num_nodes)
    model.embedding_layer.weight.data.copy_(weight_matrix(vocab, vectors)).cuda()

    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_sz, gamma=args.lr_gamma)
    criterion = nn.BCEWithLogitsLoss().cuda()


    # pre-allocate GPU memory
    # preallocate_gpu_memory(G, model, args.batch_sz, device, num_nodes, criterion)

    # training
    print("Start training!")
    def convert_text_tokens(text): return [vocab[token] for token in text]
    model, train_loss, valid_loss = train(train_dataset, dev_dataset, model, mlb, G, args.batch_sz, args.num_epochs,
                                          criterion, device, args.num_workers, optimizer, lr_scheduler)
    print('Finish training!')

    print('save model for inference')
    torch.save(model.state_dict(), args.save_model_path)

