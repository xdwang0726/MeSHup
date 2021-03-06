import argparse
import os
import random

import dgl
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import ngrams_iterator
from torchtext.vocab import Vectors
from torchtext.vocab import build_vocab_from_iterator
import pickle

from model import multichannel_GCN_title_abstract
from pytorchtools import EarlyStopping
from util import *
from util import _RawTextIterableDataset, _create_data_from_csv_abstract, _create_data_from_csv_vocab_abstract


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
        l = [item.strip() for item in l]
        label.append(l)

    title_abstract = [torch.tensor(convert_text_tokens(entry[1])) for entry in batch]
    title_abstract = pad_sequence(title_abstract, ksz=3, batch_first=True)

    return label, title_abstract


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
        for i, (label, abstract) in enumerate(train_data):
            label = torch.from_numpy(mlb.fit_transform(label)).type(torch.float)
            label = label.to(device)

            abstract = abstract.to(device)
            G, G.ndata['feat'] = G.to(device), G.ndata['feat'].to(device)

            output = model(abstract, G, G.ndata['feat'])
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
            for i, (label, abstract) in enumerate(valid_data):
                label = torch.from_numpy(mlb.fit_transform(label)).type(torch.float)
                label = label.to(device)

                abstract = abstract.to(device)
                G, G.ndata['feat'] = G.to(device), G.ndata['feat'].to(device)

                output = model(abstract, G, G.ndata['feat'])

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


def test(test_dataset, model, mlb, G, batch_sz, device):
    test_data = DataLoader(test_dataset, batch_size=batch_sz, collate_fn=generate_batch, shuffle=False, pin_memory=True)
    pred = []
    true_label = []

    print('Testing....')
    with torch.no_grad():
        model.eval()
        for label, abstract in test_data:
            label = mlb.fit_transform(label)

            abstract = abstract.to(device)
            G, G.ndata['feat'] = G.to(device), G.ndata['feat'].to(device)

            output = model(abstract, G, G.ndata['feat'])

            results = output.data.cpu().numpy()
            pred.append(results)
            true_label.append(label)

    return pred, true_label


def preallocate_gpu_memory(G, model, batch_sz, device, num_label, criterion):
    sudo_abstract = torch.randint(10000, size=(batch_sz, 500), device=device)
    sudo_label = torch.randint(2, size=(batch_sz, num_label), device=device).type(torch.float)
    G, G.ndata['feat'] = G.to(device), G.ndata['feat'].to(device)

    output = model(sudo_abstract, G, G.ndata['feat'])
    loss = criterion(output, sudo_label)
    loss.backward()
    model.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_path')
    parser.add_argument('--train_path')
    parser.add_argument('--dev_path')
    parser.add_argument('--test_path')
    parser.add_argument('--word2vec_path')
    parser.add_argument('--meSH_pair_path')
    parser.add_argument('--graph')
    parser.add_argument('--save-model-path')
    parser.add_argument('--model')
    parser.add_argument('--true')
    parser.add_argument('--results')

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--ksz', default=3)

    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_sz', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--scheduler_step_sz', type=int, default=2)
    parser.add_argument('--lr_gamma', type=float, default=0.9)

    parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:3456')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')

    args = parser.parse_args()
    set_seed(42)
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
        'all': 765920,
        'train': 765920,
        'dev': 95737,
        'test': 60000
    }
    print('load and prepare Mesh')
    # read full MeSH ID list
    mapping_id = {}
    with open(args.meSH_pair_path, 'r') as f:
        for line in f:
            (key, value) = line.split('=')
            mapping_id[key] = value.strip()

    meshIDs = list(mapping_id.keys())
    print('Total number of labels %d' % len(meshIDs))
    mlb = MultiLabelBinarizer(classes=meshIDs)
    mlb.fit(meshIDs)
    num_nodes = len(meshIDs)

    print('load pre-trained BioWord2Vec')
    vocab_iterator = _RawTextIterableDataset(NUM_LINES['all'], None, _create_data_from_csv_vocab_abstract(args.train_path))
    cache, name = os.path.split(args.word2vec_path)
    vectors = Vectors(name=name, cache=cache)
    vocab = build_vocab_from_iterator(yield_tokens(vocab_iterator))
    vocab_size = len(vocab)

    print('Load graph')
    G = dgl.load_graphs(args.graph)[0][0]
    print('graph', G.ndata['feat'].shape)

    # train_iterator = _RawTextIterableDataset(NUM_LINES['train'], 500000, _create_data_from_csv_abstract(args.train_path))
    # dev_iterator = _RawTextIterableDataset(NUM_LINES['dev'], 60000, _create_data_from_csv_abstract(args.dev_path))
    # train_dataset = to_map_style_dataset(train_iterator)
    # dev_dataset = to_map_style_dataset(dev_iterator)
    test_iterator = _RawTextIterableDataset(NUM_LINES['test'], None, _create_data_from_csv_abstract(args.test_path))
    test_dataset = to_map_style_dataset(test_iterator)
    model = multichannel_GCN_title_abstract(vocab_size, args.dropout, args.ksz, num_nodes)
    model.embedding_layer.weight.data.copy_(weight_matrix(vocab, vectors)).cuda()

    # model.cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_sz, gamma=args.lr_gamma)
    # criterion = nn.BCEWithLogitsLoss().cuda()

    # pre-allocate GPU memory
    #preallocate_gpu_memory(G, model, args.batch_sz, device, num_nodes, criterion)
    #print('pre-allocated GPU done')

    # load model
    model.load_state_dict(torch.load(args.model))
    model.to(device)
    model.eval()

    # training
    # print("Start training!")
    def convert_text_tokens(text): return [vocab[token] for token in text]
    # model, train_loss, valid_loss = train(train_dataset, dev_dataset, model, mlb, G, args.batch_sz, args.num_epochs,
    #                                       criterion, device, args.num_workers, optimizer, lr_scheduler)
    # print('Finish training!')

    # testing
    pred, true_label = test(test_dataset, model, mlb, G, args.batch_sz, device)

    # save
    pickle.dump(pred, open(args.results, 'wb'))
    pickle.dump(true_label, open(args.true, 'wb'))

    # print('save model for inference')
    # torch.save(model.state_dict(), args.save_model_path)

