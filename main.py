from data_loader import *
import time
from module_GAN import mv_gan
import torch
from sklearn.metrics import f1_score, accuracy_score
import argparse


def main(args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    if args.data == 'birds':
        train_dataset = load_noise_birds('train', 40)
        test_dataset = load_noise_birds('test', 40)
        batch_size = 3
        size_1 = 1024 * args.k_1
        size_2 = 1024 * args.k_2
    else:
        if args.data == 'mnist':
            train_dataset = load_mnist('train', args.num, args.k_1, args.k_2)
            test_dataset = load_mnist('test', args.num, args.k_1, args.k_2)
            batch_size = 100
        elif args.data == 'semi_syn':
            train_dataset = load_semisynthetic('train', 7, args.num, args.k_1, args.k_2)
            test_dataset = load_semisynthetic('test', 7, args.num, args.k_1, args.k_2)
            batch_size = 100
        elif args.data == 'syn':
            train_dataset = load_synthetic('train', args.k_1, args.k_2)
            test_dataset = load_synthetic('test', args.k_1, args.k_2)
            batch_size = 50
        elif args.data == 'celeba':
            train_dataset = load_celeba('train', 5, args.num, args.k_1, args.k_2)
            test_dataset = load_celeba('test', 5, args.num, args.k_1, args.k_2)
            batch_size = int(len(train_dataset) / 500)
        else:
            train_dataset = load_xrmb('train', 15, args.num, args.k_1, args.k_2)
            test_dataset = load_xrmb('test', 15, args.num, args.k_1, args.k_2)
            batch_size = 100
        size_1 = getattr(train_dataset, 'view_1').shape[1]
        size_2 = getattr(train_dataset, 'view_2').shape[1]
    k_1, k_2 = args.k_1, args.k_2
    model = mv_gan(train_dataset.num_class, dim_in_1=size_1, dim_in_2=size_2, gpu=args.gpu, k_1=k_1, k_2=k_2,
                   dim_hidden=args.hidden, t_1=2, t_2=2, t_3=3, alpha=args.alpha, beta=args.beta).to(device)
    print('The size of training samples: {}\nBatch size: {}'.format(len(train_dataset), batch_size))
    print('Initial learning rate: {}\nThe size of hidden layer: {}'.format(args.lr, args.hidden))
    model = model.double()
    model.learning_rate = args.lr
    model.loss = args.reg
    model.epoch = args.epoch
    model.batch_size = batch_size
    # new_model = torch.load('model_150_cycle_loss_3_{}_new.ckpt'.format(args.hidden))
    # model.load_state_dict(new_model)
    train_model(model, train_dataset, args)
    print('Finish training process!')
    evaluation(model, test_dataset)


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


def train_model(model, train_loader, args):
    epoch = 0
    cur_iter = 0
    momentum = 0.95
    if args.optim != 'sgd':
        optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=model.learning_rate, momentum=momentum)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96)
    view_weight_update_flag = False
    model.view_weight_raw.requires_grad = False
    while epoch < model.epoch:
        start_time = time.time()
        epoch += 1
        truth = []
        predicted_label = []
        diff_mean = []
        loss_mean = []
        for sample_batched in batch(train_loader, n=model.batch_size, shuffle=False):
            x_1 = torch.tensor(sample_batched['view_1']).double().to(model.device)
            x_2 = torch.tensor(sample_batched['view_2']).double().to(model.device)
            labels = torch.tensor(sample_batched['label']).double().to(model.device)
            cur_iter += 1

            # Step 1: we only update the discriminator, thus we freeze the rest neural network
            for child in model.children():
                if child.name != 'discriminator':
                    for param in child.parameters():
                        try:
                            param.requires_grad = False
                        except:
                            continue
                else:
                    for param in child.parameters():
                        try:
                            param.requires_grad = True
                        except:
                            continue
            for j in range(model.t_1):
                optimizer.zero_grad()
                loss, _, att = model(x_1, x_2, labels, stage=1)
                loss.backward()
                optimizer.step()

            # Step 2: we only update the generators, thus we freeze the rest neural network
            for child in model.children():
                # if child.name == 'generator_g' or child.name == 'generator_f' or child.name == 'classifier':
                if child.name == 'generator':
                    for param in child.parameters():
                        try:
                            param.requires_grad = True
                        except:
                            continue
                else:
                    for param in child.parameters():
                        try:
                            param.requires_grad = False
                        except:
                            continue
            for j in range(model.t_2):
                optimizer.zero_grad()
                loss, pred, diff = model(x_1, x_2, labels, stage=2)
                loss.backward()
                optimizer.step()

            # # Step 3: we only update the parameters of the classifier
            for child in model.children():
                if child.name != 'classifier':
                    for param in child.parameters():
                        try:
                            param.requires_grad = False
                        except:
                            continue
                else:
                    for param in child.parameters():
                        try:
                            param.requires_grad = True
                        except:
                            continue
            for j in range(model.t_3):
                optimizer.zero_grad()
                loss, pred, diff = model(x_1, x_2, labels, stage=2)
                loss.backward()
                optimizer.step()
            _, predicted = torch.max(pred.data, 1)
            predicted_label = predicted_label + list(predicted.cpu().detach().numpy())
            truth = truth + list(torch.max(labels.data, 1)[1].cpu().detach().numpy())
            diff_mean.append(diff)
            loss_mean.append(loss)
        my_lr_scheduler.step()
        acc = accuracy_score(truth, predicted_label)
        f1 = f1_score(np.array(truth), np.array(predicted_label), average='weighted')
        # print('view weight ={}, loss={} at iter={}'.format(model.view_weight, loss, cur_iter))
        print('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}, F1:{:.4f}, diff:{:.4f}, time:{}'.format(
            epoch, model.epoch, torch.mean(torch.tensor(loss_mean)), acc, f1,
            torch.mean(torch.tensor(diff_mean)), time.time() - start_time))
        if epoch == 100:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=momentum)
        if epoch >= 0.7 * model.epoch and view_weight_update_flag:
            print('start to update view weights at iter={}'.format(cur_iter))
            model.view_weight_raw.requires_grad = True
            view_weight_update_flag = False
            torch.save(model.state_dict(), 'model_{}_cycle_loss_{}_{}.ckpt'.format(epoch, model.loss, args.hidden))
    # Save the model checkpoint
    torch.save(model.state_dict(), 'model_{}_cycle_loss_{}_{}_new.ckpt'.format(epoch, model.loss, args.hidden))


def evaluation(model, test_loader):
    model.eval()  # eval mode (batch norm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        label_correct = 0
        total = 0
        prediction = []
        truth = []
        atten_list = []
        sv1_list = []
        sv2_list = []
        for sample_batched in batch(test_loader, n=model.batch_size, shuffle=True):
            x_1 = torch.tensor(sample_batched['view_1']).double().to(model.device)
            x_2 = torch.tensor(sample_batched['view_2']).double().to(model.device)
            labels = torch.tensor(sample_batched['label']).double()
            name = sample_batched['image_name']
            atten, outputs, sv = model(x_1, x_2, stage=3, name=name)
            atten_list.append(atten)
            sv1_list.append(sv[0])
            sv2_list.append(sv[1])
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ANTS algorithm')
    parser.add_argument('-d', dest='data', type=str, default='mnist', help='which dataset is used for demo')
    parser.add_argument('-g', dest='gpu', type=int, default=0, help='the index of the gpu to use')
    parser.add_argument('-l', dest='reg', type=int, default=3, help='which regularizer to use')
    parser.add_argument('-lr', dest='lr', type=float, default=0.08, help='The learning rate')
    parser.add_argument('-e', dest='epoch', type=int, default=20, help='the total epoch for training')
    parser.add_argument('-hid', dest='hidden', type=int, default=100, help='the size of the hidden representation')
    parser.add_argument('-n', dest='num', type=int, default=75000, help='the size of training data set')
    parser.add_argument('-clamp', dest='clamp', type=float, default=0.02, help='the value of clap')
    parser.add_argument('-alpha', dest='alpha', type=float, default=2, help='the value of alpha')
    parser.add_argument('-beta', dest='beta', type=float, default=1, help='the value of classification coef')
    parser.add_argument('-k1', dest='k_1', type=int, default=3, help='the number of segment in the first view')
    parser.add_argument('-k2', dest='k_2', type=int, default=3, help='the number of segment in the first view')
    parser.add_argument('-opt', dest='optim', type=str, default='sgd', help='the optimizer to use, sgd or adams')
    args = parser.parse_args()
    main(args)
