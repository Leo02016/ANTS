import itertools
import nltk
from itertools import chain
# from gensim.models import Word2Vec
from nltk.corpus import stopwords
import math
from sklearn.model_selection import StratifiedKFold
import os
import numpy as np
import scipy.io as sio
import random
import cv2
import imutils
import yaml
from sklearn.preprocessing import normalize
from copy import deepcopy


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


class load_celeba():
    def __init__(self, stage,  class_num, num, k_1, k_2):
        data_path = './dataset/'
        # data_path = './dataset/img_align_celeba/'
        file_list = [data_path + 'label.mat', data_path + 'view_1.mat', data_path + 'view_2.mat']
        aaa = list(map(os.path.exists, file_list))
        if sum(aaa) != len(aaa):
            self.preprocess(data_path, 5)
        self.num_class = class_num
        if stage == 'test' or stage == 'Test':
            data = sio.loadmat('{}/view_1.mat'.format(data_path))
            self.view_1 = np.array(data['test'])
            data = sio.loadmat('{}/view_2.mat'.format(data_path))
            self.view_2 = np.array(data['test'])
            label = sio.loadmat('{}/label.mat'.format(data_path))
            self.label = self.label_encoding(label['test'])
        elif stage == 'train' or stage == 'Train':
            data = sio.loadmat('{}/view_1.mat'.format(data_path))
            self.view_1 = np.array(data['train'][:num])
            data = sio.loadmat('{}/view_2.mat'.format(data_path))
            self.view_2 = np.array(data['train'][:num])
            label = sio.loadmat('{}/label.mat'.format(data_path))
            self.label = self.label_encoding(label['train'][:num])
        else:
            raise (NameError('The stage should be either train or test'))

    def label_encoding(self, label):
        arr = np.zeros((len(label), self.num_class))
        arr[np.where(label == 1.0)] = 1.0
        return arr

    def preprocess(self, data_path, k_fold):
        images = []
        labels = []
        # [bold, black hair, straingt hair, wavy hair, wearing hat]
        label_index = [5, 9, 33, 34, 36]
        with open('./dataset/list_attr_celeba.txt', 'r+') as f:
            next(f)
            next(f)
            for i in range(100000):
                data = np.array(f.readline().split())
                label = list(map(int, data[label_index]))
                if sum(label) != -3:
                    continue
                images.append(data_path + 'img_align_celeba/' + data[0])
                labels.append(label)
        images = np.array(images)
        labels = np.array(labels)
        train_index = range(45000)
        test_index = range(45000, len(images))
        train_image = images[train_index]
        test_image = images[test_index]
        train_label = labels[train_index]
        test_label = labels[test_index]
        train_v1 = []
        train_v2 = []
        for image in train_image:
            v1, v2 = self.load_image(image, 100)
            train_v1.append(v1)
            train_v2.append(v2)
        test_v1 = []
        test_v2 = []
        for image in test_image:
            v1, v2 = self.load_image(image, 100)
            test_v1.append(v1)
            test_v2.append(v2)
        sio.savemat('{}/view_1.mat'.format(data_path), mdict={'train': train_v1, 'test': test_v1})
        sio.savemat('{}/view_2.mat'.format(data_path), mdict={'train': train_v2, 'test': test_v2})
        sio.savemat('{}/label.mat'.format(data_path), mdict={'train': train_label, 'test': test_label})

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def load_image(self, image_name, scale):
        blob = np.zeros((scale, scale), dtype=np.float32)
        imgs = cv2.imread(image_name)
        assert imgs is not None, 'File {} is not loaded correctly'.format(image_name)
        if imgs.shape[0] > scale or imgs.shape[1] > scale:
            pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
            imgs = self.prep_im_for_blob(imgs, pixel_means, scale)
        blob[0:imgs.shape[0], 0:imgs.shape[1]] = self.rgb2gray(imgs)/255
        # channel_swap = (2, 0, 1)
        angle = random.randint(-45, 45)
        img = imutils.rotate(blob, angle).reshape((10000,))
        blur_img = cv2.medianBlur(img, 5).reshape((10000,))
        # view_1 = blob[:, l[1::2]].ravel().astype(np.float32)
        # view_2 = blob[:, l[::2]].ravel().astype(np.float32)
        return img, blur_img

    def prep_im_for_blob(self, im, pixel_means, scale):
        """ Mean subtract and scale an image """
        im = im.astype(np.float32, copy=False)
        im -= pixel_means
        im = cv2.resize(im, dsize=(scale, scale), interpolation=cv2.INTER_LINEAR)
        return im

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {'view_1': self.view_1[idx], 'view_2': self.view_2[idx], 'label': self.label[idx]}

    def shuffle(self):
        idx = np.random.permutation(range(len(self.view_1)))
        self.view_1 = self.view_1[idx]
        self.view_2 = self.view_2[idx]
        self.label = self.label[idx]


class load_mnist():
    def __init__(self, stage, num, k_1, k_2):
        self.num_class = 10
        if stage == 'train' or stage == 'Train':
            if not os.path.isdir('./dataset'):
                os.mkdir('./dataset')
            if os.path.exists('./dataset/noisy_mnist_train.mat'):
                data = sio.loadmat('./dataset/noisy_mnist_train.mat')
                (size, dim) = data['view_2'][:num].shape
                d_feature_2 = math.ceil(dim / float(k_2)) * k_2
                self.view_2 = np.zeros((size, d_feature_2))
                self.view_2[:, :dim] = data['view_2'][:num] / 255
                (size, dim) = data['view_1'][:num].shape
                d_feature_1 = math.ceil(dim / float(k_1)) * k_1
                self.view_1 = np.zeros((size, d_feature_1))
                self.view_1[:, :dim] = data['view_1'][:num] / 255
                self.label = self.label_encoding(data['label'][0, :num])
            else:
                import tensorflow as tf
                if not os.path.isdir('./noise'):
                    os.mkdir('./noise')
                if not os.path.isdir('./rotated'):
                    os.mkdir('./rotated')
                mnist = tf.keras.datasets.mnist
                (x_train, y_train), (_, _) = mnist.load_data()
                x_train = x_train
                x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
                view_1 = []
                view_2 = []
                for i in range(len(x_train)):
                    angle = random.randint(-45, 45)
                    img = imutils.rotate(x_train[i], angle).reshape(28 * 28, )
                    view_1.append(np.array(img))
                    blur_img = cv2.blur(x_train[i].reshape((28, 28, 1)), (4, 4)).reshape(28 * 28, )
                    view_2.append(np.array(blur_img))
                sio.savemat('./dataset/noisy_mnist_train.mat',
                            {'view_1': np.array(view_1), 'view_2': np.array(view_2), 'label': np.array(y_train)})
                (size, dim) = view_2.shape
                d_feature_2 = math.ceil(dim / float(k_2)) * k_2
                self.view_2 = np.zeros((size, d_feature_2))
                self.view_2[:, :dim] = view_2[:num] / 255
                (size, dim) = view_1.shape
                d_feature_1 = math.ceil(dim / float(k_1)) * k_1
                self.view_1 = np.zeros((size, d_feature_1))
                self.view_1[:, :dim] = view_1[:num] / 255
                self.label = self.label_encoding(y_train[:num])
        elif stage == 'test' or stage == 'Test':
            if os.path.exists('./dataset/noisy_mnist_test.mat'):
                data = sio.loadmat('./dataset/noisy_mnist_test.mat')
                (size, dim) = data['view_2'].shape
                d_feature_2 = math.ceil(dim / float(k_2)) * k_2
                self.view_2 = np.zeros((size, d_feature_2))
                self.view_2[:, :dim] = data['view_2'] / 255
                (size, dim) = data['view_1'].shape
                d_feature_1 = math.ceil(dim / float(k_1)) * k_1
                self.view_1 = np.zeros((size, d_feature_1))
                self.view_1[:, :dim] = data['view_1'] / 255
                self.label = self.label_encoding(data['label'][0])
            else:
                import tensorflow as tf
                mnist = tf.keras.datasets.mnist
                (_, _), (x_test, y_test) = mnist.load_data()
                x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
                view_1 = []
                view_2 = []
                for i in range(len(x_test)):
                    angle = random.randint(-45, 45)
                    img = imutils.rotate(x_test[i], angle).reshape((28 * 28, ))
                    view_1.append(np.array(img))
                    blur_img = cv2.blur(x_test[i].reshape((28, 28, 1)), (4, 4)).reshape(28 * 28, )
                    view_2.append(np.array(blur_img))
                self.view_2 = view_2
                self.view_1 = view_1
                self.label = self.label_encoding(y_test)
                sio.savemat('./dataset/noisy_mnist_test.mat', {'view_1': np.array(view_1), 'view_2': np.array(view_2),
                                                               'label': np.array(y_test)})
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

    def shuffle(self):
        idx = np.random.permutation(range(len(self.view_1)))
        self.view_1 = self.view_1[idx]
        self.view_2 = self.view_2[idx]
        self.label = self.label[idx]

    # Allign two views and extract n samples.
    def view_allignment(self, view_1, view_2, label_1, label_2, n):
        num = int(n / 10)
        indices_1 = np.zeros((10, num), dtype=np.int32)
        indices_2 = np.zeros((10, num), dtype=np.int32)
        label = np.array([self.one_hot_encoding(i, num) for i in range(n)])
        for digit in range(10):
            # select the first n/10 samples for the first views for each digit.
            count = 0
            for i in range(len(view_1)):
                if label_1[i] == digit:
                    indices_1[digit, count] = i
                    count += 1
                if count >= num:
                    break
            # select the first n/10 samples for the second views for each digit.
            count = 0
            for i in range(len(view_2)):
                if label_2[i, digit] == 1:
                    indices_2[digit, count] = i
                    count += 1
                if count >= num:
                    break
        indices_1 = indices_1.flatten()
        indices_2 = indices_2.flatten()
        view_1 = view_1[indices_1]
        view_2 = view_2[indices_2]
        view_2 = view_2.reshape((view_2.shape[0], 28, 28, 1))
        # permute channel to (channel, height, width)
        channel_swap = (0, 3, 1, 2)
        view_1 = view_1.transpose(channel_swap)
        view_2 = view_2.transpose(channel_swap)
        shuffle_indices = np.random.permutation(n)
        view_1 = view_1[shuffle_indices]
        view_2 = view_2[shuffle_indices]
        label = label[shuffle_indices]
        return view_1, view_2, label

    def one_hot_encoding(self, i, num):
        arr = np.zeros((10))
        count = int(i / num)
        arr[count] = 1.0
        return arr

    def label_encoding(self, label):
        arr = np.zeros((len(label), 10))
        for i in range(len(label)):
            arr[i, label[i]] = 1.0
        return arr


class load_xrmb():
    def __init__(self, stage, num_class, num, k_1, k_2):
        num = int(num/ num_class)
        if stage == 'train' or stage == 'Train':
            data = sio.loadmat('./dataset/XRMBf2KALDI_window7_single2.mat')
            index_list = list(map(lambda x: np.where(data['trainLabel'] == x)[0][:num], range(num_class)))
            indices = np.array(list(itertools.chain.from_iterable(index_list)))
            shuffle = np.random.permutation(len(indices))
            indices = indices[shuffle]
            d_feature_2 = math.ceil(data['X2'][indices].shape[1] / float(k_2)) * k_2
            self.view_2 = np.zeros((data['X2'][indices].shape[0], d_feature_2))
            self.view_2[:, :data['X2'][indices].shape[1]] = data['X2'][indices]
            self.num_class = num_class
            self.label = self.label_encoding(data['trainLabel'][indices])
            data2 = sio.loadmat('./dataset/XRMBf2KALDI_window7_single1.mat')
            d_feature_1 = math.ceil(data2['X1'][indices].shape[1] / float(k_1)) * k_1
            self.view_1 = np.zeros((data2['X1'][indices].shape[0], d_feature_1))
            self.view_1[:, :data2['X1'][indices].shape[1]] = data2['X1'][indices]
        elif stage == 'test' or stage == 'Test':
            data = sio.loadmat('./dataset/XRMBf2KALDI_window7_single2.mat')
            index_list = list(map(lambda x: np.where(data['testLabel'] == x)[0][:1000], range(num_class)))
            indices = np.array(list(itertools.chain.from_iterable(index_list)))
            shuffle = np.random.permutation(len(indices))
            indices = indices[shuffle]
            d_feature_2 = math.ceil(data['XTe2'][indices].shape[1] / float(k_2)) * k_2
            self.view_2 = np.zeros((data['XTe2'][indices].shape[0], d_feature_2))
            self.view_2[:, :data['XTe2'][indices].shape[1]] = data['XTe2'][indices]
            self.num_class = num_class
            self.label = self.label_encoding(data['testLabel'][indices])
            data2 = sio.loadmat('./dataset/XRMBf2KALDI_window7_single1.mat')
            d_feature_1 = math.ceil(data2['XTe1'][indices].shape[1] / float(k_1)) * k_1
            self.view_1 = np.zeros((data2['XTe1'][indices].shape[0], d_feature_1))
            self.view_1[:, :data2['XTe1'][indices].shape[1]] = data2['XTe1'][indices]
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
            arr[i, label[i]] = 1.0
        return arr

    def shuffle(self):
        idx = np.random.permutation(range(len(self.view_1)))
        self.view_1 = self.view_1[idx]
        self.view_2 = self.view_2[idx]
        self.label = self.label[idx]


class load_noise_birds():
    def __init__(self, stage, num_class):
        self.num_class = num_class
        if stage == 'train' or stage == 'Train':
            data = sio.loadmat('./dataset/birds/birds_ants_data.mat')
            self.label = data['train_label']
            self.view_1 = data['train_view_1']
            self.view_2 = data['train_view_2']
            self.name = data['all_image_name']
        elif stage == 'test' or stage == 'Test':
            data = sio.loadmat('./dataset/birds/birds_ants_data.mat')
            self.label = data['test_label']
            self.view_1 = data['test_view_1']
            self.view_2 = data['test_view_2']
            self.name = data['all_image_name']
        elif stage == 'all' or stage == 'All':
            data = sio.loadmat('./dataset/birds/birds_ants_data.mat')
            self.label = data['all_label']
            self.view_1 = data['all_view_1']
            self.view_2 = data['all_view_2']
            self.name = data['all_image_name']
        else:
            raise (NameError('The stage should be either train or test'))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.fmax(np.zeros(x.shape), x)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        view_1 = self.view_1[idx]
        view_2 = self.view_2[idx]
        y = self.label[idx]
        name = self.name[idx]
        sample = {'view_1': np.array(view_1), 'view_2': np.array(view_2), 'label': y, 'image_name':name}
        return sample

    def label_encoding(self, label):
        arr = np.zeros((len(label), self.num_class))
        for i in range(len(label)):
            arr[i, label[i]] = 1.0
        return arr

    def shuffle(self):
        idx = np.random.permutation(range(len(self.view_1)))
        self.view_1 = self.view_1[idx]
        self.view_2 = self.view_2[idx]
        self.label = self.label[idx]
        self.name = self.name[idx]


#
# class load_food():
#     def __init__(self, stage, num_class, num):
#         num = int(num / num_class)
#         self.num_class = num_class
#         if not os.path.exists('./dataset/tobacco/tabocco_train.mat'):
#             self.preprocess('E:/experiment_results/DeepMUSE/dataset/tobacco')
#         if stage == 'train' or stage == 'Train':
#             data = sio.loadmat("./dataset/tobacco/tabocco_train.mat")
#             self.view_1 = np.array(data['view_1'])
#             self.view_2 = np.array(data['view_2']) /255
#             self.label = np.array(self.label_encoding(data['label'][0]))
#         elif stage == 'test' or stage == 'Test':
#             data = sio.loadmat("./dataset/tobacco/tabocco_test.mat")
#             self.view_1 = np.array(data['view_1'])
#             self.view_2 = np.array(data['view_2']) /255
#             self.label = np.array(self.label_encoding(data['label'][0]))
#         else:
#             raise (NameError('The stage should be either train or test'))
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, idx):
#         view_1 = self.view_1[idx]
#         view_2 = self.view_2[idx]
#         y = self.label[idx]
#         sample = {'view_1': view_1, 'view_2': view_2, 'label': y}
#         return sample
#
#     def label_encoding(self, label):
#         arr = np.zeros((len(label), self.num_class))
#         for i in range(len(label)):
#             arr[i, label[i]] = 1.0
#         return arr
#
#     def shuffle(self):
#         idx = np.random.permutation(range(len(self.view_1)))
#         self.view_1 = self.view_1[idx]
#         self.view_2 = self.view_2[idx]
#         self.label = self.label[idx]
#
#     def preprocess(self, data_path):
#         k_fold = 5
#         text_data = []
#         image_data_dir = []
#         image_data = []
#         label = []
#         stopword = stopwords.words('english')
#         tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
#         tok_func = lambda x: [el.lower().encode('ascii', 'ignore').decode("utf-8") for el in tokenizer.tokenize(x) if el.lower() not in stopword and el.lower().isalpha()]
#         # tok_func = lambda x: [el.lower() for el in tokenizer.tokenize(x)]
#         class_count = 0
#         for file in os.listdir('{}/text'.format(data_path)):
#             print('{}/text/{}'.format(data_path, file))
#             for subfile in os.listdir('{}/text/{}'.format(data_path, file)):
#                 try:
#                     with open('{}/text/{}/{}'.format(data_path, file, subfile), 'r', encoding='utf-8') as f:
#                         text = []
#                         for line in f.readlines():
#                             text.append(line)
#                         tokens = [tok_func(x) for x in text if tok_func(x) != []]
#                         tokens = list(chain.from_iterable(tokens))
#                         if len(tokens) > 500 or len(tokens) < 3:
#                             print('Skip file: {}/text/{}/{}'.format(data_path, file, subfile))
#                             continue
#                         text_data.append(tokens)
#                         label.append(class_count)
#                         image_name ='%s/image/%s/%s.jpg' % (data_path, file, subfile[:-4])
#                         image_data_dir.append(image_name)
#                         # load image data
#                         imgs = cv2.imread(image_name, 0)
#                         output = cv2.resize(imgs, (128, 128)).reshape(1, 128, 128)
#                         image_data.append(output)
#                 except OSError:
#                     print("File open Error!")
#             class_count += 1
#         print(os.getcwd())
#         label = np.array(label)
#         image_data = np.array(image_data)
#         text_data = np.array(text_data)
#         model = Word2Vec(text_data, size=50, window=5, min_count=1, workers=4)
#         model.save("word2vec.model")
#         max_length = 0
#         for doc in text_data:
#             if len(doc) > max_length:
#                 max_length = len(doc)
#         documents = np.zeros((len(text_data), max_length, 50))
#         for i in range(len(text_data)):
#             text = []
#             for token in text_data[i]:
#                 text.append(model[token])
#             documents[i, :len(text), :] = np.array(text)
#         documents = np.array(documents, dtype=np.float32)
#         skf = StratifiedKFold(n_splits=k_fold, random_state=10, shuffle=True)
#         for train_index, test_index in skf.split(np.zeros(len(label)), label):
#             random.shuffle(train_index)
#             random.shuffle(test_index)
#             break
#         sio.savemat('{}/tabocco_train.mat'.format(data_path),
#                     {'view_1': documents[train_index], 'view_2': image_data[train_index], 'label': label[train_index]})
#         sio.savemat('{}/tabocco_test.mat'.format(data_path),
#                     {'view_1': documents[test_index], 'view_2': image_data[test_index], 'label': label[test_index]})
