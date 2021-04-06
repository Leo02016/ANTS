import scipy.io as sio
from sklearn.model_selection import StratifiedKFold
import os
from bs4 import BeautifulSoup as Soup
import codecs
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
import random
import scipy.io as sio
import math

class load_synthetic():
    def __init__(self, stage, k_1, k_2):
        if stage == 'test' or stage == 'Test':
            data = sio.loadmat('./dataset/syn_dataset/syn_test.mat')
            self.view_1 = data['view_1']
            self.view_2 = data['view_2']
            self.num_class = 2
            self.label = self.label_encoding(data['label'][0])
            # print(data['index'])
            # print('noise level for 400-450: {}'.format(len(np.where(data['index'][0][400:450] >= 500)[0])/50.0))
            # print('noise level for 450-500: {}'.format(len(np.where(data['index'][0][450:500] >= 500)[0])/50.0))
            # print('noise level for 500-550: {}'.format(len(np.where(data['index'][0][500:] >= 500)[0])/50.0))
        elif stage == 'train' or stage == 'Train':
            data = sio.loadmat('./dataset/syn_dataset/syn_train.mat')
            self.view_1 = data['view_1']
            self.view_2 = data['view_2']
            self.num_class = 2
            self.label = self.label_encoding(data['label'][0])
        else:
            raise (NameError('The stage should be either train or test'))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        view_1 = self.view_1[idx]
        view_2 = self.view_2[idx]
        y = self.label[idx]
        sample = {'view_1': np.array(view_1), 'view_2': np.array(view_2), 'label': y}
        return sample

    def label_encoding(self, label):
        arr = np.zeros((len(label), self.num_class))
        for i in range(len(label)):
            arr[i, label[i]-1] = 1.0
        return arr

    def shuffle(self):
        idx = np.random.permutation(range(len(self.view_1)))
        self.view_1 = self.view_1[idx]
        self.view_2 = self.view_2[idx]
        self.label = self.label[idx]


class load_semisynthetic():
    def __init__(self, stage, num_class, num, k_1, k_2):
        if stage == 'test' or stage == 'Test':
            data = sio.loadmat('./dataset/synthetic_dataset_1/view_1.mat')
            self.num_class = num_class
            (size, dim) = data['test'].shape
            d_feature_1 = math.ceil(dim / float(k_1)) * k_1
            self.view_1 = np.zeros((size, d_feature_1))
            self.view_1[:, :dim] = data['test']
            data = sio.loadmat('./dataset/synthetic_dataset_1/view_2.mat')
            (size, dim) = data['test'].shape
            d_feature_2 = math.ceil(dim / float(k_2)) * k_2
            self.view_2 = np.zeros((size, d_feature_2))
            self.view_2[:, :dim] = data['test']
            label = sio.loadmat('./dataset/synthetic_dataset_1/label.mat')
            self.label = self.label_encoding(label['test'][0])
        elif stage == 'train' or stage == 'Train':
            data = sio.loadmat('./dataset/synthetic_dataset_1/view_1.mat')
            (size, dim) = data['train'].shape
            d_feature_1 = math.ceil(dim / float(k_1)) * k_1
            self.view_1 = np.zeros((size, d_feature_1))
            self.view_1[:, :dim] = data['train']
            self.num_class = num_class
            data = sio.loadmat('./dataset/synthetic_dataset_1/view_2.mat')
            (size, dim) = data['train'].shape
            d_feature_2 = math.ceil(dim / float(k_2)) * k_2
            self.view_2 = np.zeros((size, d_feature_2))
            self.view_2[:, :dim] = data['train']
            label = sio.loadmat('./dataset/synthetic_dataset_1/label.mat')
            self.label = self.label_encoding(label['train'][0][:num])
        else:
            raise (NameError('The stage should be either train or test'))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        view_1 = self.view_1[idx]
        view_2 = self.view_2[idx]
        y = self.label[idx]
        sample = {'view_1': np.array(view_1), 'view_2': np.array(view_2), 'label': y}
        return sample

    def label_encoding(self, label):
        arr = np.zeros((len(label), self.num_class))
        for i in range(len(label)):
            arr[i, label[i]-1] = 1.0
        return arr

    def shuffle(self):
        idx = np.random.permutation(range(len(self.view_1)))
        self.view_1 = self.view_1[idx]
        self.view_2 = self.view_2[idx]
        self.label = self.label[idx]


class load_web_kb():
    def __init__(self, stage, num_class, num):
        self.class_num = num_class
        if stage == 'test' or stage == 'Test':
            data = sio.loadmat('./web_content.mat')
            self.view_1 = data['test']
            self.num_class = num_class
            data = sio.loadmat('./web_link.mat')
            self.view_2 = data['test']
            label = sio.loadmat('./label.mat')
            self.label = label['test']
        elif stage == 'train' or stage == 'Train':
            data = sio.loadmat('./web_content.mat')
            self.view_1 = data['train']
            self.num_class = num_class
            data = sio.loadmat('./web_link.mat')
            self.view_2 = data['train']
            label = sio.loadmat('./label.mat')
            self.label = label['train']
        else:
            raise (NameError('The stage should be either train or test'))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        view_1 = self.data[0][idx]
        view_2 = self.data[1][idx]
        view_3 = self.data[2][idx]
        # y = self.oneHotEncode(idx)
        y = self.label[idx]
        sample = {'view_1': np.array(view_1), 'view_2': np.array(view_2), 'view_3': np.array(view_3), 'label': y}
        return sample

    def shuffle(self):
        idx = np.random.permutation(range(len(self.view_1)))
        self.view_1 = self.view_1[idx]
        self.view_2 = self.view_2[idx]
        self.label = self.label[idx]


def preprocess(data_path):
    k_fold = 5
    task = ['cornell', 'washington', 'texas', 'wisconsin', 'misc']
    # feature_type = ['course', 'department', 'faculty', 'project', 'student', 'staff']
    feature_type = ['course', 'department', 'faculty', 'project', 'student', 'staff', 'other']
    web_content = []
    web_links = []
    label = []
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tok_func = lambda x: [el.lower() for el in tokenizer.tokenize(x)]
    for file in os.listdir(data_path):
        if file in feature_type:
            os.chdir('%s/%s' % (data_path, file))
            print('%s/%s' % (data_path, file))
            for subfile in os.listdir(os.curdir):
                if subfile in task:
                    file_count = 0
                    os.chdir('%s/%s/%s' % (data_path, file, subfile))
                    print('%s/%s/%s' % (data_path, file, subfile))
                    for url in os.listdir(os.curdir):
                        try:
                            with codecs.open(url, "rU", encoding='utf-8', errors='ignore') as fdata:
                                soup = Soup(fdata, "html.parser")
                                file_count += 1
                                # ---- get title of the web page ---
                                title = soup.title
                                if title is None:
                                    title = []
                                else:
                                    title = [tok_func(x) for x in title.text.split() if tok_func(x) != []]
                                    new_title = []
                                    for sublist in title:
                                        if len(sublist) != 0:
                                            for item in sublist:
                                                new_title.append(item)
                                        else:
                                            new_title.append(sublist)
                                    title = new_title
                                # ---- get all words in the web page ---
                                text = soup.get_text()
                                text = text.split()
                                tokens = [tok_func(x) for x in text if tok_func(x) != []]
                                new_tokens = []
                                for x in tokens:
                                    new_tokens = new_tokens + x
                                # ---- get all links pointing to that page  ----
                                links = []
                                for link in soup.find_all(name="a"):
                                    if 'href' in link.attrs:
                                        text = link.text
                                        text = text.split()
                                        text = [tok_func(x) for x in text if len(x) != 0 and tok_func(x) != []]
                                        for x in text:
                                            links.append(x)
                                New_links = []
                                for sublist in links:
                                    for item in sublist:
                                        New_links.append(item)
                                links = New_links
                                # add features
                                web_content.append(new_tokens)
                                web_links.append(links)
                                web_links.append(title)
                                # add labels
                                if file == 'course':
                                    label.append(1)
                                elif file == 'department':
                                    label.append(2)
                                elif file == 'faculty':
                                    label.append(3)
                                elif file == 'project':
                                    label.append(4)
                                elif file == 'student':
                                    label.append(5)
                                elif file == 'staff':
                                    label.append(6)
                                else:
                                    label.append(7)
                        except OSError:
                            print("File open Error!")
    os.chdir('%s' % data_path)
    os.chdir('../')

    web_content = np.array(web_content)
    web_links = np.array(web_links)
    label = np.array(label)
    # splitting data into training data set and testing data set
    skf = StratifiedKFold(n_splits=k_fold, random_state=10, shuffle=True)
    for train_index, test_index in skf.split(np.zeros(len(label)), label):
        random.shuffle(train_index)
        random.shuffle(test_index)
        train_label = label[train_index]
        test_label = label[test_index]
        break
    print('Saving label...')
    scipy.io.savemat('./dataset/synthetic_dataset_1/label.mat', mdict={'train': train_label, 'test': test_label})
    tfidf(web_content, train_index, test_index)


def sigmoid(x):
    return 1 / (1 + np.exp(-x)) - 0.5


def tanh(x):
    a = np.exp(-x)
    b = np.exp(x)
    return (b - a)/(b + a) - 0.5


def tfidf(data, train_index, test_index):
    stopwords = set([e.lower() for e in set(nltk.corpus.stopwords.words('english'))])
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tok_func = lambda x: [el.lower() for el in tokenizer.tokenize(x) if el.lower() not in stopwords]
    tok_nop = lambda x: x
    tfidf = TfidfVectorizer(tokenizer=tok_nop, strip_accents=False,
                            stop_words=set(nltk.corpus.stopwords.words('english')), min_df=10, max_df=250,
                            lowercase=False)
    tokens = [tok_func(str(x)) for x in data]
    tfidf_mat = tfidf.fit_transform(tokens)
    tfidf_mat = np.array(tfidf_mat.todense())
    noise_1 = np.random.normal(0, 0.2, (data.shape[0], 100))
    noise_2 = np.random.normal(0, 0.2, (data.shape[0], 100))
    view_1 = np.concatenate((sigmoid(tfidf_mat * 20), noise_1), axis=1)
    view_2 = np.concatenate((tanh(tfidf_mat * 20), noise_2), axis=1)
    np.random.shuffle(view_1.T)
    np.random.shuffle(view_2.T)
    train_v1 = view_1[train_index]
    test_v1 = view_1[test_index]
    train_v2 = view_2[train_index]
    test_v2 = view_2[test_index]
    scipy.io.savemat('./dataset/synthetic_dataset_1/view_1.mat', mdict={'train': train_v1, 'test': test_v1})
    scipy.io.savemat('./dataset/synthetic_dataset_1/view_2.mat', mdict={'train': train_v2, 'test': test_v2})


if __name__ == '__main__':
    data_path = 'C:/Users/leo/Dropbox/paper/KDD_2019/webkb'
    file_list = ['train_web_content.npy', 'train_web_links.npy', 'train_web_title.npy',
                 'test_web_content.npy', 'test_web_links.npy', 'test_web_title.npy',
                 'train_label.npy', 'test_label.npy']
    aaa = list(map(os.path.exists, file_list))
    if sum(aaa) != len(aaa):
        print('Raw data has not been pre-processed! Start pre-processing the raw data.')
        preprocess(data_path)
    else:
        print('Loading the existing data set...')

