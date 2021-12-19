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
from util import MeSH_indexing, pad_sequence


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # cudnn


def prepare_dataset(train_data_path, dev_data_path, test_data_path, MeSH_id_pair_file, word2vec_path, graph_file):
    """ Load Dataset and Preprocessing """
    # load training data
    f = open(train_data_path, encoding="utf8")
    objects = ijson.items(f, 'articles.item')

    train_pmid = []
    train_text = []
    train_id = []

    print('Start loading training data')
    logging.info("Start loading training data")
    for i, obj in enumerate(tqdm(objects)):
        if i <= 100000:
            text = {}
            try:
                ids = obj["pmid"]
                title = obj['title'].strip()
                text['TITLE'] = title
                abstract = obj['abstractText'].strip()
                text['ABSTRACT'] = abstract
                intro = obj['INTRO']
                text['INTRO'] = intro
                method = obj['METHODS']
                text['METHODS'] = method
                results = obj['RESULTS']
                text['RESULTS'] = results
                discuss = obj['DISCUSS']
                text['DISCUSS'] = discuss
                mesh_id = list(obj['mesh'].keys())
                train_pmid.append(ids)
                train_text.append(text)
                train_id.append(mesh_id)
            except AttributeError:
                print(obj["pmid"].strip())
        else:
            break

    print("Finish loading training data")

    # load dev data
    f_dev = open(dev_data_path, encoding="utf8")
    dev_objects = ijson.items(f_dev, 'articles.item')

    dev_pmid = []
    dev_text = []
    dev_id = []

    print('Start loading dev data')
    logging.info("Start loading training data")
    for i, obj in enumerate(tqdm(dev_objects)):
        text = {}
        try:
            ids = obj["pmid"]
            title = obj['title'].strip()
            text['TITLE'] = title
            abstract = obj['abstractText'].strip()
            text['ABSTRACT'] = abstract
            intro = obj['INTRO']
            text['INTRO'] = intro
            method = obj['METHODS']
            text['METHODS'] = method
            results = obj['RESULTS']
            text['RESULTS'] = results
            discuss = obj['DISCUSS']
            text['DISCUSS'] = discuss
            mesh_id = list(obj['mesh'].keys())
            dev_pmid.append(ids)
            dev_text.append(text)
            dev_id.append(mesh_id)
        except AttributeError:
            print(obj["pmid"].strip())

    print("Finish loading dev data, number of development", len(dev_id))

    # load test data
    f_t = open(test_data_path, encoding="utf8")
    test_objects = ijson.items(f_t, 'documents.item')

    test_pmid = []
    test_text = []
    test_id = []

    print('Start loading test data')
    logging.info("Start loading test data")
    for i, obj in enumerate(tqdm(test_objects)):
        text = {}
        try:
            ids = obj["pmid"]
            title = obj['title'].strip()
            text['TITLE'] = title
            abstract = obj['abstractText'].strip()
            text['ABSTRACT'] = abstract
            intro = obj['INTRO']
            text['INTRO'] = intro
            method = obj['METHODS']
            text['METHODS'] = method
            results = obj['RESULTS']
            text['RESULTS'] = results
            discuss = obj['DISCUSS']
            text['DISCUSS'] = discuss
            mesh_id = list(obj['mesh'].keys())
            test_pmid.append(ids)
            test_text.append(text)
            test_id.append(mesh_id)
        except AttributeError:
            print(obj["pmid"].strip())

    print("Finish loading test data, number of test", len(test_pmid))

    print('load and prepare Mesh')
    # read full MeSH ID list
    mapping_id = {}
    with open(MeSH_id_pair_file, 'r') as f:
        for line in f:
            (key, value) = line.split('=')
            mapping_id[key] = value.strip()

    meshIDs = list(mapping_id.values())
    print('Total number of labels %d' % len(meshIDs))
    mlb = MultiLabelBinarizer(classes=meshIDs)
    mlb.fit(meshIDs)

    # create Vector object map tokens to vectors
    print('load pre-trained BioWord2Vec')
    cache, name = os.path.split(word2vec_path)
    vectors = Vectors(name=name, cache=cache)

    # Preparing training and test datasets
    print('prepare training and test sets')
    alltext = train_text + dev_text + test_text
    train_dataset = MeSH_indexing(alltext, train_text, train_id, is_test=False)
    dev_dataset = MeSH_indexing(alltext, train_texts=dev_text, train_labels=dev_id, is_test=False)
    test_dataset = MeSH_indexing(alltext, test_texts=test_text, test_labels=test_id, is_test=True)

    # build vocab
    print('building vocab')
    vocab = train_dataset.get_vocab()

    # Prepare label features
    print('Load graph')
    G = dgl.load_graphs(graph_file)[0][0]
    print('graph', G.ndata['feat'].shape)

    print('prepare dataset `````````and labels graph done!')
    return len(meshIDs), mlb, vocab, train_dataset, dev_dataset, test_dataset, vectors, G


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
    # check if the dataset is multi-channel or not
    if len(batch[0]) == 5:
        label = [entry[0] for entry in batch]
        # padding according to the maximum sequence length in batch
        abstract = [entry[1] for entry in batch]
        abstract_length = torch.Tensor([len(seq) for seq in abstract])
        abstract = pad_sequence(abstract, ksz=3, batch_first=True)

        intro = [entry[2] for entry in batch]
        intro_length = torch.Tensor([len(seq) for seq in intro])
        intro = pad_sequence(intro, ksz=3, batch_first=True)

        method = [entry[3] for entry in batch]
        method_length = torch.Tensor([len(seq) for seq in method])
        method = pad_sequence(method, ksz=3, batch_first=True)

        results = [entry[4] for entry in batch]
        results_length = torch.Tensor([len(seq) for seq in results])
        results = pad_sequence(results, ksz=3, batch_first=True)

        discuss = [entry[5] for entry in batch]
        discuss_length = torch.Tensor([len(seq) for seq in discuss])
        discuss = pad_sequence(discuss, ksz=3, batch_first=True)

        return label, abstract, intro, method, results, discuss, abstract_length, intro_length, method_length, results_length, discuss_length
    else:
        print('WARNING: BATCH ERROR!')


def train(train_dataset, valid_dataset, model, mlb, G, batch_sz, num_epochs, criterion, device, num_workers, optimizer,
          lr_scheduler):

    train_data = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, collate_fn=generate_batch, num_workers=num_workers, pin_memory=True)

    valid_data = DataLoader(valid_dataset, batch_size=batch_sz, shuffle=True, collate_fn=generate_batch, num_workers=num_workers, pin_memory=True)

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
        for i, (label, abstract, intro, method, results, discuss, abstract_length, intro_length, method_length, results_length, discuss_length) in enumerate(train_data):
            label = torch.from_numpy(mlb.fit_transform(label)).type(torch.float)
            label = label.to(device)

            abstract, abstract_length = abstract.to(device), abstract_length.to(device)
            intro, intro_length = intro.to(device), intro_length.to(device)
            method, method_length = method.to(device), method_length.to(device)
            results, results_length = results.to(device), results_length.to(device)
            discuss, discuss_length = discuss.to(device), discuss_length.to(device)

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
            for i, (label, abstract, intro, method, results, discuss, abstract_length, intro_length, method_length, results_length, discuss_length) in enumerate(valid_data):
                label = torch.from_numpy(mlb.fit_transform(label)).type(torch.float)
                label = label.to(device)

                abstract, abstract_length = abstract.to(device), abstract_length.to(device)
                intro, intro_length = intro.to(device), intro_length.to(device)
                method, method_length = method.to(device), method_length.to(device)
                results, results_length = results.to(device), results_length.to(device)
                discuss, discuss_length = discuss.to(device), discuss_length.to(device)

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


def main():
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

    num_nodes, mlb, vocab, train_dataset, dev_dataset, test_dataset, vectors, G = \
        prepare_dataset(args.train_path, args.dev_path, args.test_path, args.meSH_pair_path, args.word2vec_path, args.graph)

    vocab_size = len(vocab)

    model = multichannel_GCN(vocab_size, args.dropout, num_nodes)
    model.embedding_layer.weight.data.copy_(weight_matrix(vocab, vectors)).cuda()

    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_sz, gamma=args.lr_gamma)
    criterion = nn.BCEWithLogitsLoss().cuda()


    # pre-allocate GPU memory
    # preallocate_gpu_memory(G, model, args.batch_sz, device, num_nodes, criterion)

    # training
    print("Start training!")
    model, train_loss, valid_loss = train(train_dataset, dev_dataset, model, mlb, G, args.batch_sz, args.num_epochs,
                                          criterion, device, args.num_workers, optimizer, lr_scheduler)
    print('Finish training!')

    print('save model for inference')
    torch.save(model.state_dict(), args.save_model_path)


if __name__ == "__main__":
    main()
