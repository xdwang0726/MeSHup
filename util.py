import logging
import string
from typing import TypeVar, List

import torch
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import ngrams_iterator
from torchtext.vocab import Vocab
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

stop_words = set(stopwords.words('english'))
table = str.maketrans('', '', string.punctuation)


def text_clean(tokens):

    stripped = [w.translate(table) for w in tokens]  # remove punctuation
    clean_tokens = [w for w in stripped if w.isalpha()]  # remove non alphabetic tokens
    text_nostop = [word for word in clean_tokens if word not in stop_words]  # remove stopwords
    filtered_text = [w for w in text_nostop if len(w) > 1]  # remove single character token

    return filtered_text


def _text_iterator(text, labels=None, ngrams=1):
    """ all_text: a list of dictionary, each dictionary: {section: texts, ....}"""
    tokenizer = get_tokenizer('basic_english')
    for i, text in enumerate(text):
        title = tokenizer(text['TITLE'])
        abstract = tokenizer(text['ABSTRACT'])
        title_abstract = title + abstract
        title_abstract = text_clean(title_abstract)

        intro = tokenizer(text['INTRO'])
        intro = text_clean(intro)

        method = tokenizer(text['METHODS'])
        method = text_clean(method)

        results = tokenizer(text['RESULTS'])
        results = text_clean(results)

        discuss = tokenizer(text['DISCUSS'])
        discuss = text_clean(discuss)

        label = labels[i]

        yield label, ngrams_iterator(title_abstract, ngrams), ngrams_iterator(intro, ngrams), ngrams_iterator(method, ngrams), ngrams_iterator(results, ngrams), ngrams_iterator(discuss, ngrams)


def _create_data_from_iterator(iterator):
    data = []
    labels = []
    with tqdm(unit_scale=0, unit='lines') as t:
        for label, abstract, intro, method, results, discuss in iterator:
            # if include_unk:
            # abstract_token = torch.tensor([vocab[token] for token in abstract])
            # print('abtoken', abstract_token)
            #     intro_token = torch.tensor([vocab[token] for token in intro])
            #     method_token = torch.tensor([vocab[token] for token in method])
            #     results_token = torch.tensor([vocab[token] for token in results])
            #     discuss_token = torch.tensor([vocab[token] for token in discuss])
            # else:
            #     abstract_token = torch.tensor(list(filter(lambda x: x is not Vocab.UNK, [vocab[token] for token in abstract])))
            #     intro_token = torch.tensor(list(filter(lambda x: x is not Vocab.UNK, [vocab[token] for token in intro])))
            #     method_token = torch.tensor(list(filter(lambda x: x is not Vocab.UNK, [vocab[token] for token in method])))
            #     results_token = torch.tensor(list(filter(lambda x: x is not Vocab.UNK, [vocab[token] for token in results])))
            #     discuss_token = torch.tensor(list(filter(lambda x: x is not Vocab.UNK, [vocab[token] for token in discuss])))
            data.append((label, abstract, intro, method, results, discuss))
            labels.extend(label)
            t.update(1)
        return data, list(set(labels))


class MultiLabelTextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, vocab, data, labels=None):
        """Initiate text-classification dataset.
         Arguments:
             vocab: Vocabulary object used for dataset.
             data: a list of label/tokens tuple. tokens are a tensor after numericalizing the string tokens.
                   label is a list of list.
                 [([label1], ab_tokens1, title_tokens1), ([label2], ab_tokens2, title_tokens2), ([label3], ab_tokens3, title_tokens3)]
             label: a set of the labels.
                 {label1, label2}
        """
        super(MultiLabelTextClassificationDataset, self).__init__()
        # self._vocab = vocab
        self._data = data
        self._labels = labels

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            yield x

    def get_labels(self):
        return self._labels

    # def get_vocab(self):
    #     return self._vocab


def _setup_datasets(alltext, train_texts=None, train_labels=None, test_texts=None, test_labels=None, ngrams=1, vocab=None,
                    include_unk=False, is_test=False):
    if vocab is None:
        logging.info('Building Vocab based on {}'.format(alltext))
        # vocab = build_vocab_from_iterator(_vocab_iterator(alltext, ngrams))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    print('Vocab has {} entries'.format(len(vocab)))
    if is_test:
        logging.info('Creating testing data')
        test_data, test_labels = _create_data_from_iterator(_text_iterator(test_texts, labels=test_labels, ngrams=ngrams))
        logging.info('Total number of labels in test set:'.format(len(test_labels)))
        return MultiLabelTextClassificationDataset(vocab, test_data, test_labels)
    else:
        logging.info('Creating training data')
        train_data, train_labels = _create_data_from_iterator(_text_iterator(train_texts, labels=train_labels, ngrams=ngrams))
        logging.info('Total number of labels in training set:'.format(len(train_labels)))
        return MultiLabelTextClassificationDataset(vocab, train_data, train_labels)


def MeSH_indexing(alltexts, train_texts=None, train_labels=None, test_texts=None, test_labels=None, is_test=False):
    """
    Defines MeSH_indexing datasets.
    The label set contains all mesh terms in 2019 version (https://meshb.nlm.nih.gov/treeView)
    """
    return _setup_datasets(alltexts, train_texts, train_labels, test_texts, test_labels, ngrams=1, vocab=None, include_unk=False,
                           is_test=is_test)


def pad_sequence(sequences, ksz, batch_first=False, padding_value=0.0):
    # type: (List[Tensor], bool, float) -> Tensor
    r"""Pad a list of variable length Tensors with ``padding_value``
    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.
    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.
    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])
    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.
    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.
    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if max_len < ksz:
        max_len = ksz
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor
