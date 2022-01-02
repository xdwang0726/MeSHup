import csv
import io
import string
import sys
from typing import TypeVar, List

import torch
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import ngrams_iterator

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

stop_words = set(stopwords.words('english'))
table = str.maketrans('', '', string.punctuation)

tokenizer = get_tokenizer('basic_english')


def _create_data_from_csv_vocab(data_path):
    with io.open(data_path, encoding="utf8") as f:
        next(f)
        reader = unicode_csv_reader(f)
        for row in reader:
            text = row[5] + row[1] + row[2] + row[3] + row[0]
            yield text


def yield_tokens(data_iter, ngrams=1):
    for text in data_iter:
        yield ngrams_iterator(tokenizer(text), ngrams)


def unicode_csv_reader(unicode_csv_data, **kwargs):
    r"""Since the standard csv library does not handle unicode in Python 2, we need a wrapper.
    Borrowed and slightly modified from the Python docs:
    https://docs.python.org/2/library/csv.html#csv-examples
    Args:
        unicode_csv_data: unicode csv data (see example below)
    Examples:
        >>> from torchtext.utils import unicode_csv_reader
        >>> import io
        >>> with io.open(data_path, encoding="utf8") as f:
        >>>     reader = unicode_csv_reader(f)
    """

    # Fix field larger than field limit error
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)
    csv.field_size_limit(maxInt)

    for line in csv.reader(unicode_csv_data, **kwargs):
        yield line


def _create_data_from_csv(data_path):
    with io.open(data_path, encoding="utf8") as f:
        next(f)
        reader = unicode_csv_reader(f)
        for row in reader:
            title_abstract = text_clean(tokenizer(row[5]))
            intro = text_clean(tokenizer(row[1]))
            method = text_clean(tokenizer(row[2]))
            results = text_clean(tokenizer(row[3]))
            discuss = text_clean(tokenizer(row[0]))
            print('discuss original', row[0], type(row[0]))
            print('tokenize', tokenizer(row[0]))
            print('clean', text_clean(tokenizer(row[0])))
            print('utils', discuss)
            yield row[6], title_abstract, intro, method, results, discuss


class _RawTextIterableDataset(torch.utils.data.IterableDataset):
    """Defines an abstraction for raw text iterable datasets.
    """

    def __init__(self, full_num_lines, iterator):
        """Initiate the dataset abstraction.
        """
        super(_RawTextIterableDataset, self).__init__()
        self.full_num_lines = full_num_lines
        self._iterator = iterator
        self.num_lines = full_num_lines
        self.current_pos = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_pos == self.num_lines - 1:
            raise StopIteration
        item = next(self._iterator)
        if self.current_pos is None:
            self.current_pos = 0
        else:
            self.current_pos += 1
        return item

    def __len__(self):
        return self.num_lines

    def pos(self):
        """
        Returns current position of the iterator. This returns None
        if the iterator hasn't been used yet.
        """
        return self.current_pos


def to_map_style_dataset(iter_data):
    r"""Convert iterable-style dataset to map-style dataset.
    args:
        iter_data: An iterator type object. Examples include Iterable datasets, string list, text io, generators etc.
    """

    # Inner class to convert iterable-style to map-style dataset
    class _MapStyleDataset(torch.utils.data.Dataset):

        def __init__(self, iter_data):
            # TODO Avoid list issue #1296
            self._data = list(iter_data)

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            return self._data[idx]

    return _MapStyleDataset(iter_data)


def text_clean(tokens):

    stripped = [w.translate(table) for w in tokens]  # remove punctuation
    clean_tokens = [w for w in stripped if w.isalpha()]  # remove non alphabetic tokens
    text_nostop = [word for word in clean_tokens if word not in stop_words]  # remove stopwords
    filtered_text = [w for w in text_nostop if len(w) > 1]  # remove single character token

    return filtered_text


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
