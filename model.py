import dgl.function as fn
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        """
        inputs: g,       object of Graph
                feature, node features
        """
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)


class LabelNet(nn.Module):
    def __init__(self, hidden_gcn_size, num_classes, in_node_features):
        super(LabelNet, self).__init__()
        self.gcn1 = GCNLayer(in_node_features, hidden_gcn_size)
        self.gcn2 = GCNLayer(hidden_gcn_size, num_classes)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = F.relu(x)
        x = self.gcn2(g, x)
        return x


class CorNet(nn.Module):
    def __init__(self, output_size, cornet_dim=1000, n_cornet_blocks=2):
        super(CorNet, self).__init__()
        self.intlv_layers = nn.ModuleList(
            [CorNetBlock(cornet_dim, output_size) for _ in range(n_cornet_blocks)])
        for layer in self.intlv_layers:
            nn.init.xavier_uniform_(layer.dstbn2cntxt.weight)
            nn.init.xavier_uniform_(layer.cntxt2dstbn.weight)

    def forward(self, logits):
        for layer in self.intlv_layers:
            logits = layer(logits)
        return logits


class multichannel_GCN(nn.Module):
    def __init__(self, vocab_size, dropout, ksz, output_size, G, device, embedding_dim=200, rnn_num_layers=2, cornet_dim=1000,
                 n_cornet_blocks=2):
        super(multichannel_GCN, self).__init__()

        self.vocab_size = vocab_size
        self.dropout = dropout
        self.ksz = ksz
        self.embedding_dim = embedding_dim

        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embedding_dim)

        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=rnn_num_layers,
                           dropout=self.dropout, bidirectional=True, batch_first=True)

        self.gcn = LabelNet(embedding_dim, embedding_dim, embedding_dim)

        # corNet
        self.cornet = CorNet(output_size, cornet_dim, n_cornet_blocks)

    def forward(self, abstract, title, mask, ab_length, title_length, g, g_node_feature):
        # get label features
        label_feature = self.gcn(g, g_node_feature)
        label_feature = torch.cat((label_feature, g_node_feature), dim=1) # torch.Size([29368, 200*2])

        # get title content features
        atten_mask = label_feature.transpose(0, 1) * mask.unsqueeze(1)
        title = self.embedding_layer(title.long())
        title = pack_padded_sequence(title, title_length, batch_first=True, enforce_sorted=False) # packed input title

        output_title, (_,_) = self.rnn(title) # packed rnn output title
        output_title, _ = pad_packed_sequence(output_title, batch_first=True)  # unpacked rnn output title with size: (bs, seq_len, emb_dim*2)

        alpha_title = torch.softmax(torch.matmul(output_title, atten_mask), dim=1)
        title_features = torch.matmul(output_title.transpose(1, 2), alpha_title).transpose(1, 2)  # size: (bs, 29368, embed_dim*2)

        # get abstract content features
        abstract = self.embedding_layer(abstract)  # size: (bs, seq_len, embed_dim)
        abstract = pack_padded_sequence(abstract, ab_length, batch_first=True, enforce_sorted=False)
        output_abstract, (_,_) = self.rnn(abstract)
        output_abstract, _ = pad_packed_sequence(output_abstract, batch_first=True)  # (bs, seq_len, emb_dim*2)

        alpha_abstract = torch.softmax(torch.matmul(output_abstract, atten_mask), dim=1)  # size: (bs, seq_len-ksz+1, 29368)
        abstract_features = torch.matmul(output_abstract.transpose(1, 2), alpha_abstract).transpose(1, 2)  # size: (bs, 29368, embed_dim*2)

        # get document feature
        x_feature = title_features + abstract_features  # size: (bs, 29368, embed_dim*2)
        x_feature = torch.sum(x_feature * label_feature, dim=2)

        # add CorNet
        x_feature = self.cornet(x_feature)
        return x_feature