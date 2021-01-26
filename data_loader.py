import json

import numpy as np
import nltk
from nltk.corpus import stopwords
import string
from torchtext.data import Field
import torch
from torch.autograd import Variable
# nltk.download('stopwords')


class Loader:
    def __init__(self, file_path, max_len=75):

        self.file_path = file_path
        self.file = open(self.file_path, 'r')
        self.max_len = max_len
        self._vector_file = None
        self._word_vec_mat = None
        self._word_vec_tot = None
        self._word_vec_dim = None
        self.data = None
        self._word_id = None
        self._clean_text = None


    @property
    def vector_file(self):
        """Fast Text word vector file."""
        if self._vector_file is None:
            processed = []
            for words in self.clean_text:
                processed.append(words['text'])

            text_field = Field()
            print("Loading Vectors")
            text_field.build_vocab(
                                 processed, 
                                 vectors='fasttext.simple.300d'
                                )
            self._vector_file = text_field.vocab
        return self._vector_file

    @property
    def word_vec_tot(self):
        if self._word_vec_tot is None:

            self._word_vec_tot = len(self.vector_file)
        return self._word_vec_tot

    @property
    def word_vec_dim(self):
        if self._word_vec_dim is None:
            self._word_vec_dim = 300
        return self._word_vec_dim

    @property
    def word_vec_mat(self):
        if self._word_vec_mat is None:
            self._word_vec_mat = np.zeros((self.word_vec_tot, self.word_vec_dim), dtype=np.float32)
        return self._word_vec_mat
    
    def word_encoder(self, words):
        encoded = []
        for word in words:
            word_emb = self.vector_file[word]
            encoded.append(word_emb)
        return encoded

    @property
    def json_loader(self):
        """Load json for training."""
        if self.data is None:
            json_files = []
            for item in self.file:
                json_files.append(json.loads(item))
            self.data = json_files
        return self.data

    @staticmethod
    def normalizer(tokens):
        """Remove stop words and punctuations."""
        stop = set(stopwords.words('english') + list(string.punctuation))
        clean_text = []
        for word in tokens:
            if word.lower() not in stop and len(word) > 2:
                clean_text.append(word.lower())
        return clean_text

    @property
    def clean_text(self):
        """Clean text arrays."""
        if self._clean_text is None:
            output = []
            for item in self.json_loader:
                item['text'] = self.normalizer(item['text'])
                output.append(item)
            self._clean_text = output
        return self._clean_text

    def data_splitter(self, data_set):
        '''
            Split the dataset according to the specified train_classes, val_classes
            and test_classes
            @param all_data: list of examples (dictionaries)
            @param train_classes: list of int
            @param val_classes: list of int
            @param test_classes: list of int
            @return train_data: list of examples
            @return val_data: list of examples
            @return test_data: list of examples
        '''
        if 'amazon' in self.file_path:
            train_classes, val_classes, test_classes = self._get_amazon_classes

        train_data, val_data, test_data = [], [], []

        for example in data_set:
            if example['label'] in train_classes:
                train_data.append(example)
            if example['label'] in val_classes:
                val_data.append(example)
            if example['label'] in test_classes:
                test_data.append(example)

        return train_data, val_data, test_data

    @property
    def _get_amazon_classes(self):
        train_classes = [2, 3, 4, 7, 11, 12, 13, 18, 19, 20]
        val_classes = [1, 22, 23, 6, 9]
        test_classes = [0, 5, 14, 15, 8, 10, 16, 17, 21]

        return train_classes, val_classes, test_classes

    def array_maker(self, data):
        """Make data into list of arrays for batching."""
        out_ = {}
        label = []
        text = []
        for i in data:
            label.append(i['label'])
            text.append(i['text'])

        out_['label'] = label
        out_['text'] = text
        return out_

    def train_val_test(self):
        """Split data to train test split based on labels."""
        data_input = self.clean_text

        train_data, val_data, test_data = self.data_splitter(data_input)

        train_data = self.array_maker(train_data)
        val_data = self.array_maker(val_data)
        test_data = self.array_maker(test_data)

        train_data['encoded'] = self.pad_sequence([self.word_encoder(words) for words in train_data['text']], self.max_len)
        val_data['encoded'] = self.pad_sequence([self.word_encoder(words) for words in val_data['text']], self.max_len)
        test_data['encoded'] = self.pad_sequence([self.word_encoder(words) for words in test_data['text']], self.max_len)


        return train_data, val_data, test_data

    @staticmethod
    def pad_sequence(arrays, size):
        out = []
        for item in arrays:
            new_item = item + [0] * (size - len(item))
            new_item = new_item[:size]
            out.append(new_item)
        return out



def extract_sample(n_way, n_support, n_query, datax, datay):
    """
    Picks random sample of size n_support+n_querry, for n_way classes
    Args:
    n_way (int): number of classes in a classification task
    n_support (int): number of labeled examples per class in the support set
    n_query (int): number of labeled examples per class in the query set
    datax (np.array): dataset of images
    datay (np.array): dataset of labels
    Returns:
    (dict) of:
    (torch.Tensor): sample of images. Size (n_way, n_support+n_query, (dim))
    (int): n_way
    (int): n_support
    (int): n_query
    """


    support_set = {'pos': []}
    query_set = {'pos': []}
    query_label = []
    K = np.random.choice(np.unique(datay), n_way, replace=False)
    datax = np.array(datax)

    for i, ls in enumerate(K):
        datax_cls = datax[datay == ls]

        support, query, _ = np.split(datax_cls , [n_support, n_support + n_query])

        support_set['pos'].append(support)
        query_set['pos'].append(query)
        query_label += [i] * n_query

    support_set['pos'] = np.stack(support_set['pos'], 0)
    query_set['pos'] = np.concatenate(query_set['pos'], 0)
    query_label = np.array(query_label)

    perm = np.random.permutation(n_way * n_query)
    query_set['pos'] = query_set['pos'][perm]
    query_label =query_label[perm]

    return support_set, query_set, query_label  


def batch_maker(batches, n_way, n_support, n_query, max_length, x, y):
    """Create batch for training."""
    support = {'pos': []}
    query = {'pos': []}
    label = []

    for solo in range(batches):
        current_support, current_query, current_label = extract_sample(n_way, n_support, n_query, x, y)
        support['pos'].append(current_support['pos'])
        query['pos'].append(current_query['pos'])
        label.append(current_label)

    support['pos'] = Variable(torch.from_numpy(np.stack(support['pos'], 0)).long().view(-1, max_length))
    query['pos'] = Variable(torch.from_numpy(np.stack(query['pos'], 0)).long().view(-1, max_length))
    label = Variable(torch.from_numpy(np.stack(label, 0).astype(np.int64)).long())

    return support, query, label



r = Loader('/data/andrew/amazon.json')
train, val, test = r.train_val_test()



n_way = 3
n_support = 5
n_query = 5
max_length = 75


support, query, label = batch_maker(1, n_way, n_support, n_query, max_length, train['encoded'], train['label'])


from embedding import Embedder


model = Embedder(r.word_vec_tot, 10)

print('Support: ', support['pos'].shape)

output = model(support['pos'])
query1 = model(query['pos'])

print('LAbel', label)

print('Support Shape Embedding', output.shape)

output = output.view(3, 5, 750)


query1 = query1.view(15, 1, 750)

# print(output.shape)
print('Query Post Embedding', query1.shape)


proto = torch.mean(output, 1)

print('Proto: ', proto.shape)

# batch * classes * support_n

# proto = []
# for i in range(3):
#     proto.append(torch.mean(output[i * 5: (i + 1) * 5], dim=1))

# proto = torch.cat(proto, dim=0)

# # print(proto)
# print('PROTO', proto.shape)
# print('BEfore', query1.shape)

# # query1 = query1.reshape(15, 75, 10)
# print('after', query1.shape)

# print('unsqueeze', proto.unsqueeze(0))
# for i in query1:
diff = torch.pow(proto.unsqueeze(1) - query1.unsqueeze(2), 2).sum(3)
# print('Distance', diff)




# print('diff View', vi.shape)
_, pred = torch.max(diff, 1)

print(pred)

import torch.nn.functional as F
import torch.nn as nn

acc = torch.mean((pred == label).type(torch.FloatTensor))
# pred = pred.reshape(1,15)

# loss =  nn.CrossEntropyLoss(pred, label)

print(acc)