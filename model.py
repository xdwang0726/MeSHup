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


class CorNetBlock(nn.Module):
    def __init__(self, context_size, output_size):
        super(CorNetBlock, self).__init__()
        self.dstbn2cntxt = nn.Linear(output_size, context_size)
        self.cntxt2dstbn = nn.Linear(context_size, output_size)
        self.act_fn = torch.sigmoid

    def forward(self, output_dstrbtn):
        identity_logits = output_dstrbtn
        output_dstrbtn = self.act_fn(output_dstrbtn)
        context_vector = self.dstbn2cntxt(output_dstrbtn)
        context_vector = F.elu(context_vector)
        output_dstrbtn = self.cntxt2dstbn(context_vector)
        output_dstrbtn = output_dstrbtn + identity_logits
        return output_dstrbtn


class multichannel_GCN(nn.Module):
    def __init__(self, vocab_size, dropout, ksz, output_size, embedding_dim=200, cornet_dim=1000, n_cornet_blocks=2):
        super(multichannel_GCN, self).__init__()

        self.vocab_size = vocab_size
        self.dropout = dropout
        self.ksz = ksz
        self.embedding_dim = embedding_dim

        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embedding_dim)

        self.dconv = nn.Sequential(nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=self.ksz, padding=0, dilation=1),
                                   nn.SELU(), nn.AlphaDropout(p=0.05),
                                   nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=self.ksz, padding=0, dilation=2),
                                   nn.SELU(), nn.AlphaDropout(p=0.05),
                                   nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=self.ksz, padding=0, dilation=3),
                                   nn.SELU(), nn.AlphaDropout(p=0.05))

        self.gcn = LabelNet(embedding_dim, embedding_dim, embedding_dim)

        # corNet
        self.cornet = CorNet(output_size, cornet_dim, n_cornet_blocks)

    def forward(self, abstract, intro, method, results, discuss, g, g_node_feature):
        # get label features
        label_feature = self.gcn(g, g_node_feature)
        # label_feature = torch.cat((label_feature, g_node_feature), dim=1) # torch.Size([29368, 200*2])
        label_feature = label_feature + g_node_feature  # torch.Size([29368, 200])

        # get content features
        abstract = self.embedding_layer(abstract.long())  #size: (bs, seq_len, embed_dim)
        print('embedding', abstract.size())
        abstract_conv = self.dconv(abstract.permute(0, 2, 1))  # (bs, embed_dim, seq_len-ksz+1)
        abstract_atten = torch.softmax(torch.matmul(abstract_conv.transpose(1, 2), label_feature.transpose(0, 1)), dim=1)  # size: (bs, seq_len-ksz+1, 29368)
        abstract_feature = torch.matmul(abstract_conv, abstract_atten).transpose(1, 2)  # size: (bs, 29368, embed_dim)

        intro = self.embedding_layer(intro.long())
        intro_conv = self.dconv(intro)
        intro_atten = torch.softmax(torch.matmul(intro_conv.transpose(1, 2), label_feature.transpose(0, 1)), dim=1)
        intro_feature = torch.matmul(intro_conv, intro_atten).transpose(1, 2)

        method = self.embedding_layer(method.long())
        method_conv = self.dconv(method)
        method_atten = torch.softmax(torch.matmul(method_conv.transpose(1, 2), label_feature.transpose(0, 1)), dim=1)
        method_feature = torch.matmul(method_conv, method_atten).transpose(1, 2)

        results = self.embedding_layer(results.long())
        results_conv = self.dconv(results)
        results_atten = torch.softmax(torch.matmul(results_conv.transpose(1, 2), label_feature.transpose(0, 1)), dim=1)
        results_feature = torch.matmul(results_conv, results_atten).transpose(1, 2)

        discuss = self.embedding_layer(discuss.long())
        discuss_conv = self.dconv(discuss)
        discuss_atten = torch.softmax(torch.matmul(discuss_conv.transpose(1, 2), label_feature.transpose(0, 1)), dim=1)
        discuss_feature = torch.matmul(discuss_conv, discuss_atten).transpose(1, 2)

        context_feature = abstract_feature + intro_feature + method_feature + results_feature + discuss_feature

        # get document feature
        x_feature = torch.sum(context_feature * label_feature, dim=2)

        # add CorNet
        x_feature = self.cornet(x_feature)
        return x_feature