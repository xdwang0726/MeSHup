import argparse
import os
import random

import dgl
import numpy as np
import torch.distributed as dist
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtext.vocab import Vectors, build_vocab_from_iterator

from model import multichannel_GCN
from pytorchtools import EarlyStopping
from util import *
from util import _RawTextIterableDataset, _create_data_from_csv_vocab, _create_data_from_csv


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # cudnn


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

    return label, title_abstract


def train(train_dataset, valid_dataset, model, mlb, G, batch_sz, num_epochs, criterion, device, num_workers, optimizer,
          lr_scheduler, world_size, rank):

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank)
    train_data = DataLoader(train_dataset, batch_size=batch_sz, sampler=train_sampler, collate_fn=generate_batch, num_workers=num_workers, pin_memory=True)

    valid_data = DataLoader(valid_dataset, batch_size=batch_sz, sampler=valid_sampler, collate_fn=generate_batch, num_workers=num_workers, pin_memory=True)

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


def preallocate_gpu_memory(G, model, batch_sz, device, num_label, criterion):
    sudo_abstract = torch.randint(123900, size=(batch_sz, 400), device=device)
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

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.2)

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
    ngpus_per_node = torch.cuda.device_count()
    print('number of gpus per node: %d' % ngpus_per_node)
    world_size = int(os.environ['SLURM_NTASKS'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + int(local_rank)

    available_gpus = list(os.environ.get('CUDA_VISIBLE_DEVICES').replace(',',""))  # check if it is multiple gpu
    print('available gpus: ', available_gpus)
    current_device = int(available_gpus[local_rank])
    torch.cuda.set_device(current_device)

    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    # init the process group
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=world_size, rank=rank)
    print("process group ready!")

    print('From Rank: {}, ==> Making model..'.format(rank))
    # Get dataset and label graph & Load pre-trained embeddings

    NUM_LINES = {
        'all': 957426,
        'train': 765920,
        'dev': 95737,
        'test': 95769
    }
    # NUM_LINES = {
    #     'all': 1000,
    #     'train': 70,
    #     'dev': 20,
    #     'test': 95769
    # }
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
    vocab_iterator = _RawTextIterableDataset(NUM_LINES['all'], _create_data_from_csv_vocab(args.full_path))
    cache, name = os.path.split(args.word2vec_path)
    vectors = Vectors(name=name, cache=cache)
    vocab = build_vocab_from_iterator(yield_tokens(vocab_iterator))
    vocab_size = len(vocab)

    print('Load graph')
    G = dgl.load_graphs(args.graph)[0][0]
    print('graph', G.ndata['feat'].shape)

    def convert_text_tokens(text): return [vocab[token] for token in text]
    train_iterator = _RawTextIterableDataset(NUM_LINES['train'], _create_data_from_csv(args.train_path))
    dev_iterator = _RawTextIterableDataset(NUM_LINES['dev'], _create_data_from_csv(args.dev_path))
    train_dataset = to_map_style_dataset(train_iterator)
    dev_dataset = to_map_style_dataset(dev_iterator)

    model = multichannel_GCN(vocab_size, args.dropout, num_nodes)
    model.embedding_layer.weight.data.copy_(weight_matrix(vocab, vectors)).cuda()

    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[current_device], output_device=current_device)
    print('From Rank: {}, ==> Preparing data..'.format(rank))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_sz, gamma=args.lr_gamma)
    criterion = nn.BCEWithLogitsLoss().cuda()


    # pre-allocate GPU memory
    # preallocate_gpu_memory(G, model, args.batch_sz, current_device, num_nodes, criterion)

    # training
    print("Start training!")
    model, train_loss, valid_loss = train(train_dataset, dev_dataset, model, mlb, G, args.batch_sz, args.num_epochs,
                                          criterion, current_device, args.num_workers, optimizer, lr_scheduler,
                                          world_size, rank)
    print('Finish training!')

    print('save model for inference')
    torch.save(model.state_dict(), args.save_model_path)

