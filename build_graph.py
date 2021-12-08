import argparse
import os
import timeit

import dgl
import ijson
import numpy as np
import pandas as pd
import torch
from dgl.data.utils import save_graphs
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vectors
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import BertModel


tokenizer = get_tokenizer('basic_english')


def get_edge_and_node_fatures(MeSH_id_pair_file, parent_children_file, vectors):
    """
    :param file:
    :return: edge:          a list of nodes pairs [(node1, node2), (node3, node4), ...] (39904 relations)
             node_count:    int, number of nodes in the graph
             node_features: a Tensor with size [num_of_nodes, embedding_dim]
    """
    print('load MeSH id and names')
    # get descriptor and MeSH mapped
    mapping_id = {}
    with open(MeSH_id_pair_file, 'r') as f:
        for line in f:
            (key, value) = line.split('=')
            mapping_id[key] = value.strip()

    # count number of nodes and get edges
    print('count number of nodes and get edges of the graph')
    node_count = len(mapping_id)
    print('number of nodes: ', node_count)
    values = list(mapping_id.values())
    edges = []
    with open(parent_children_file, 'r') as f:
        for line in f:
            item = tuple(line.strip().split(" "))
            index_item = (values.index(item[0]), values.index(item[1]))
            edges.append(index_item)
    print('number of edges: ', len(edges))

    print('get label embeddings')
    label_embedding = torch.zeros(0)
    for key, value in tqdm(mapping_id.items()):
        key = tokenizer(key)
        key = [k.lower() for k in key]
        embedding = []
        for k in key:
            embedding.append(vectors.__getitem__(k))

        key_embedding = torch.mean(torch.stack(embedding), dim=0, keepdim=True)
        label_embedding = torch.cat((label_embedding, key_embedding), dim=0)

    return edges, node_count, label_embedding


def build_MeSH_graph(edge_list, nodes, label_embedding):
    print('start building the graph')
    g = dgl.DGLGraph()
    # add nodes into the graph
    print('add nodes into the graph')
    g.add_nodes(nodes)
    # add edges, directional graph
    print('add edges into the graph')
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # add node features into the graph
    print('add node features into the graph')
    g.ndata['feat'] = label_embedding
    return g


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--word2vec_path')
    parser.add_argument('--meSH_pair_path')
    parser.add_argument('--mesh_parent_children_path')
    parser.add_argument('--biobert')
    parser.add_argument('--output')
    parser.add_argument('--graph_type', type=str)

    args = parser.parse_args()

    print('Load pre-trained vectors')
    cache, name = os.path.split(args.word2vec_path)
    vectors = Vectors(name=name, cache=cache)

    edges, node_count, label_embedding = get_edge_and_node_fatures(args.meSH_pair_path, args.mesh_parent_children_path,
                                                                   vectors)
    G = build_MeSH_graph(edges, node_count, label_embedding)

    save_graphs(args.output, G)


if __name__ == "__main__":
    start = timeit.default_timer()
    main()
    stop = timeit.default_timer()
    print('Time: ', stop - start)