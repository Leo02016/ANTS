import numpy as np
import scipy.io as sio
import random
import cv2
import imutils
import tensorflow as tf
import os
import itertools
from sklearn.preprocessing import normalize


class load_mnist():
    def __init__(self, stage, num):
        self.num_class = 10
        if stage == 'train' or stage == 'Train':
            if not os.path.isdir('./dataset'):
                os.mkdir('./dataset')
            if os.path.exists('./dataset/noisy_mnist_train.mat'):
                data = sio.loadmat('./dataset/noisy_mnist_train.mat')
                self.view_2 = data['view_2'][:num] / 255
                self.view_1 = data['view_1'][:num] / 255
                self.label = self.label_encoding(data['label'][0, :num])
            else:
                mnist = tf.keras.datasets.mnist
                (x_train, y_train), (_, _) = mnist.load_data()
                x_train = x_train
                x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
                view_1 = []
                view_2 = []
                for i in range(len(x_train)):
                    angle = random.randint(-45, 45)
                    img = imutils.rotate(x_train[i], angle).reshape(28 * 28, )
                    # img = imutils.rotate(x_train[i], angle).reshape((28 * 28, ))
                    view_1.append(np.array(img))
                    # blur_img = cv2.medianBlur(x_train[i].reshape((28, 28, 1)), 5)
                    blur_img = cv2.blur(x_train[i].reshape((28, 28, 1)), (4, 4)).reshape(28 * 28, )
                    # blur_img = cv2.medianBlur(x_train[i].reshape((28, 28, 1)), 3).reshape((28 * 28, ))
                    view_2.append(np.array(blur_img))
                    # cv2.imwrite('./rotated/rotated_{}.png'.format(i), img)
                    # cv2.imwrite('./noise/noise_{}.png'.format(i), blur_img)
                sio.savemat('./dataset/noisy_mnist_train.mat',
                            {'view_1': np.array(view_1), 'view_2': np.array(view_2), 'label': np.array(y_train)})
                self.view_2 = view_2[:num]
                self.view_1 = view_1[:num]
                self.label = self.label_encoding(y_train[:num])
        elif stage == 'test' or stage == 'Test':
            if os.path.exists('./dataset/noisy_mnist_test.mat'):
                data = sio.loadmat('./dataset/noisy_mnist_test.mat')
                self.view_2 = data['view_2'] / 255
                self.view_1 = data['view_1'] / 255
                self.label = self.label_encoding(data['label'][0])
            else:
                mnist = tf.keras.datasets.mnist
                (_, _), (x_test, y_test) = mnist.load_data()
                x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
                view_1 = []
                view_2 = []
                for i in range(len(x_test)):
                    angle = random.randint(-45, 45)
                    img = imutils.rotate(x_test[i], angle).reshape((28 * 28, ))
                    view_1.append(np.array(img))
                    # blur_img = cv2.medianBlur(x_test[i].reshape((28, 28, 1)), 5).reshape((28 * 28, ))
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
