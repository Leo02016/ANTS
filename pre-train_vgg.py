from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torchvision import  models
import time
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
import cv2


class pre_train_vgg_model(nn.Module):
    def __init__(self, class_num=40, batch_size=32):
        super(pre_train_vgg_model, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        self.batch_size = batch_size
        self.fcl = nn.Sequential(nn.Linear(25088, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True))
        self.fcl2 = nn.Sequential(nn.Dropout(0.5), nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
                                  nn.Dropout(0.5), nn.Linear(1024, class_num))

    def forward(self, image):
        output = self.features(image)
        output = output.view(output.shape[0], -1)
        output = self.fcl(output)
        return output, self.fcl2(output)


class load_birds():
    def __init__(self, stage, num_class):
        self.num_class = num_class
        if stage == 'train' or stage == 'Train':
            data = sio.loadmat('./dataset/birds/birds_2_views.mat')
            labels = []
            view_1 = []
            view_2 = []
            for i in range(len(data['view_1'])):
                view_1.append(data['view_1'][i])
                view_1.append('{}_0.jpg'.format(data['view_1'][i].strip()[:-4]))
                view_2.append(data['view_2'][i])
                view_2.append('{}_0.jpg'.format(data['view_2'][i].strip()[:-4]))
                labels.append(data['label'][0][i])
                labels.append(data['label'][0][i])
            self.view_1 = np.concatenate([view_1, view_2])
            self.label = self.label_encoding(np.concatenate([labels, labels]))
        elif stage == 'ants_v1':
            data = sio.loadmat('./dataset/birds/birds_2_views.mat')
            labels = []
            view_1 = []
            for i in range(len(data['view_1'])):
                view_1.append('{}_0.jpg'.format(data['view_1'][i].strip()[:-4]))
                view_1.append('{}_1.jpg'.format(data['view_1'][i].strip()[:-4]))
                view_1.append('{}_2.jpg'.format(data['view_1'][i].strip()[:-4]))
                labels.append(data['label'][0][i])
                labels.append(data['label'][0][i])
                labels.append(data['label'][0][i])
            self.view_1 = np.array(view_1)
            self.label = self.label_encoding(labels)
        elif stage == 'ants_v2':
            data = sio.loadmat('./dataset/birds/birds_2_views.mat')
            labels = []
            view_2 = []
            for i in range(len(data['view_1'])):
                view_2.append('{}_0.jpg'.format(data['view_2'][i].strip()[:-4]))
                view_2.append('{}_1.jpg'.format(data['view_2'][i].strip()[:-4]))
                view_2.append('{}_2.jpg'.format(data['view_2'][i].strip()[:-4]))
                labels.append(data['label'][0][i])
                labels.append(data['label'][0][i])
                labels.append(data['label'][0][i])
            self.view_1 = np.array(view_2)
            self.label = self.label_encoding(labels)
        else:
            raise (NameError('The stage should be either train or test'))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        images = []
        for image in self.view_1[idx]:
            images.append(self.load_image(image, 224))
        images = np.array(images)
        y = self.label[idx]
        sample = {'view_1': images, 'label': y, 'name': self.view_1[idx]}
        return sample

    def label_encoding(self, label):
        arr = np.zeros((len(label), self.num_class))
        for i in range(len(label)):
            arr[i, label[i]-1] = 1.0
        return arr

    def shuffle(self):
        idx = np.random.permutation(range(len(self.view_1)))
        self.view_1 = self.view_1[idx]
        self.label = self.label[idx]

    def load_image(self, image_name, scale):
        blob = np.zeros((3, scale, scale), dtype=np.float32)
        imgs = cv2.imread(image_name)
        assert imgs is not None, 'File {} is not loaded correctly'.format(image_name)
        if imgs.shape[0] > scale or imgs.shape[1] > scale:
            pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
            imgs = self.prep_im_for_blob(imgs, pixel_means, scale)
        blob[:, :imgs.shape[0], :imgs.shape[1]] = imgs.transpose(2, 0, 1)
        return blob

    def prep_im_for_blob(self, im, pixel_means, scale):
        """ Mean subtract and scale an image """
        im = im.astype(np.float32, copy=False)
        im -= pixel_means
        im = cv2.resize(im, dsize=(scale, scale), interpolation=cv2.INTER_LINEAR)
        return im


def batch(iterable, data=None, n=1,shuffle=True):
    l = len(iterable)
    if shuffle:
        iterable.shuffle()
    if data == None:
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
    else:
        data.shuffle()
        l2 = len(data)
        n2 = int(l2 * n / l)
        for ndx in range(0, int(l2/n2)):
            yield (iterable[ndx*n:min(ndx*n + n, l)], data[ndx*n2:min(ndx*n2+n2, l2)])


def train(model, train_loader):
    epoch = 0
    cur_iter = 0
    momentum = 0.95
    optimizer = torch.optim.SGD(model.parameters(), lr=model.learning_rate, momentum=momentum)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96)
    while epoch < model.epoch:
        start_time = time.time()
        epoch += 1
        truth = []
        predicted_label = []
        loss_mean = []
        for sample_batched in batch(train_loader, n=model.batch_size):
            x_1 = torch.tensor(sample_batched['view_1']).float().to(model.device)
            labels = torch.tensor(sample_batched['label']).float().to(model.device)
            cur_iter += 1
            optimizer.zero_grad()
            output, pred = model(x_1)
            loss = F.binary_cross_entropy_with_logits(pred, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(pred.data, 1)
            predicted_label = predicted_label + list(predicted.cpu().detach().numpy())
            truth = truth + list(torch.max(labels.data, 1)[1].cpu().detach().numpy())
            loss_mean.append(loss)
        my_lr_scheduler.step()
        acc = accuracy_score(truth, predicted_label)
        f1 = f1_score(np.array(truth), np.array(predicted_label), average='weighted')
        print('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}, F1:{:.4f},  time:{}'.format(
            epoch, model.epoch, torch.mean(torch.tensor(loss_mean)), acc, f1, time.time() - start_time))
    torch.save(model.state_dict(), 'vgg16_fine_tune_model.ckpt')


def evaluation(model, test_loader):
    model.eval()  # eval mode (batch norm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        label_correct = 0
        total = 0
        prediction = []
        truth = []
        for sample_batched in batch(test_loader, n=model.batch_size):
            x_1 = torch.tensor(sample_batched['view_1']).float().to(model.device)
            labels = torch.tensor(sample_batched['label']).float()
            hid, outputs = model(x_1)
            _, predicted_label = torch.max(outputs.data, 1)
            predicted = np.array([[1 if j == a else 0 for j in range(labels.shape[1])] for a in predicted_label.data])
            true_label = torch.tensor(np.array([list(labels[i]).index(1) for i in range(labels.shape[0])])).to(model.device)
            total += labels.size(0)
            label_correct += (predicted_label == true_label.long()).sum().item()
            prediction = prediction + list(predicted)
            truth = truth + list(labels.numpy())
        print('The Accuracy of the predictions on the {} test documents: {:.4f} %'.format(
            test_loader.label.shape[0], 100 * label_correct / total))
        print('F1 scores of the model on the {} test documents: {:.4f} %'.format(
            test_loader.label.shape[0], 100 * f1_score(np.array(truth), np.array(prediction), average='weighted')))


def generate_representations(model, test_loader_v1, test_loader_v2, name):
    data_path = './dataset/birds/'
    model.eval()  # eval mode (batch norm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        data = []
        labels = []
        y = []
        index = np.mod(range(len(test_loader_v1.view_1)), 3) == 0
        image_name = test_loader_v1.view_1[index]
        for sample_batched in batch(test_loader_v1, n=3, shuffle=False):
            x_1 = torch.tensor(sample_batched['view_1']).float().to(model.device)
            label = torch.tensor(sample_batched['label']).float()
            hid, _ = model(x_1)
            data.append(hid.cpu().detach().numpy())
            labels.append(label[0].cpu().detach().numpy())
            y.append(torch.max(label[0]).cpu().detach().numpy())
        data = np.array(data)
        labels = np.array(labels)
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(X=data, y=np.array(y)):
            break
        train_view_1 = data[train_index]
        test_view_1 = data[test_index]
        train_labels = labels[train_index]
        test_labels = labels[test_index]
        view_2 = []
        for sample_batched in batch(test_loader_v2, n=3, shuffle=False):
            x_1 = torch.tensor(sample_batched['view_1']).float().to(model.device)
            hid, _ = model(x_1)
            view_2.append(hid.cpu().detach().numpy())
        view_2 = np.array(view_2)
        train_view_2 = view_2[train_index]
        test_view_2 = view_2[test_index]
        sio.savemat('{}/birds_{}_data.mat'.format(data_path, name),
                    {'train_view_1': train_view_1, 'train_view_2': train_view_2, 'train_label': train_labels,
                     'test_view_1': test_view_1, 'test_view_2': test_view_2, 'test_label': test_labels,
                     'all_view_1': data, 'all_view_2': view_2, 'all_label': labels,
                     'all_image_name': image_name})


def main():
    gpu = 0
    batch_size = 50
    num_epochs = 25
    model = pre_train_vgg_model(40, batch_size)
    model.epoch = num_epochs
    model.learning_rate = 0.03
    model.device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    model = model.to(model.device)
    train_dataset = load_birds('train', 40)
    ants_dataset_v1 = load_birds('ants_v1', 40)
    ants_dataset_v2 = load_birds('ants_v2', 40)
    # new_model = torch.load('vgg16_fine_tune_model.ckpt')
    # model.load_state_dict(new_model)
    train(model, train_dataset)
    # evaluation(model, ants_dataset_v1)
    # evaluation(model, ants_dataset_v2)
    generate_representations(model, ants_dataset_v1, ants_dataset_v2, 'ants')


if __name__ == '__main__':
    main()