import numpy as np
import scipy
import scipy.io as sio
import torch
import torch.nn as nn
from sklearn.datasets import make_classification


class non_linear_mapping(nn.Module):
    def __init__(self, data_size, num_class):
        super(non_linear_mapping, self).__init__()
        dim_in_1 = data_size
        dim_in_2 = data_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.non_linear_1 = nn.Sequential(
            nn.Linear(dim_in_1, 500), nn.BatchNorm1d(500), nn.ReLU(inplace=True),
            nn.Linear(500, dim_in_1), nn.BatchNorm1d(dim_in_1), nn.ReLU(inplace=True))
        self.non_linear_2 = nn.Sequential(
            nn.Linear(dim_in_2, 500), nn.BatchNorm1d(500), nn.ReLU(inplace=True),
            nn.Linear(500, dim_in_2), nn.BatchNorm1d(dim_in_2), nn.ReLU(inplace=True))
        self.classifier_1 = nn.Sequential(nn.Linear(dim_in_1, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                                            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                            nn.Linear(256, num_class))
        self.classifier_2 = nn.Sequential(nn.Linear(dim_in_2, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                                            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                            nn.Linear(256, num_class))

    def forward(self, data, stage=1):
        if stage == 1:
            output = self.non_linear_1(data)
            return self.classifier_1(output)
        elif stage == 2:
            output = self.non_linear_2(data)
            return self.classifier_2(output)
        elif stage == 3:
            return self.non_linear_1(data)
        else:
            return self.non_linear_2(data)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.fmax(np.zeros(x.shape), x)


def main(path):
    data, label = make_classification(n_samples=10000, n_features=500, n_informative=250, n_redundant=50, n_repeated=50, n_classes=2,
                        n_clusters_per_class=2, weights=None, flip_y=0.015, class_sep=1.0, hypercube=True, shift=0.0,
                        scale=1.0, shuffle=True, random_state=None)
    norm = np.amax(data, axis=0) # axis=1 to select the max of a row, while axis=0 to select the max of a column
    data = data / norm

    view_1 = sigmoid(data - 0.02)
    view_2 = np.tanh(data - 0.01)
    mul_1 = np.mean(view_1)
    mul_2 = np.mean(view_2)
    noise_1 = np.random.normal(mul_1 + 0.05, 1, (data.shape[0], 50))
    noise_2 = np.random.normal(mul_2 + 0.05, 1, (data.shape[0], 50))
    view_1 = np.concatenate((view_1, noise_1), axis=1)
    view_2 = np.concatenate((view_2, noise_2), axis=1)
    row_indices = range(10000)
    np.random.permutation(row_indices)
    view_1 = view_1[row_indices]
    view_2 = view_2[row_indices]
    label = label[row_indices]
    n = 8000
    train_v_1 = view_1[:n]
    test_v_1 = view_1[n:]
    train_v_2 = view_2[:n]
    test_v_2 = view_2[n:]
    y_train = label[:n]
    y_test = label[n:]
    scipy.io.savemat(path + 'unshuffled_syn_train.mat', mdict={'view_1': train_v_1, 'view_2': train_v_2, 'label': y_train})
    scipy.io.savemat(path + 'unshuffled_syn_test.mat', mdict={'view_1': test_v_1, 'view_2': test_v_2, 'label': y_test})
    col_indices = range(550)
    col_indices = np.concatenate([col_indices[:400], np.random.permutation(col_indices[400:550])])
    view_2 = view_2[:, col_indices]
    view_1 = view_1[:, col_indices]
    train_v_1 = view_1[:n]
    test_v_1 = view_1[n:]
    train_v_2 = view_2[:n]
    test_v_2 = view_2[n:]
    y_train = label[:n]
    y_test = label[n:]
    scipy.io.savemat(path + 'syn_train.mat', mdict={'view_1': train_v_1, 'view_2': train_v_2, 'label': y_train, 'index': col_indices})
    scipy.io.savemat(path + 'syn_test.mat', mdict={'view_1': test_v_1, 'view_2': test_v_2, 'label': y_test, 'index': col_indices})


if __name__ == '__main__':
    data_path = './dataset/syn_dataset/'
    main(data_path)


