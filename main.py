from data_loader import *
from torch.utils.data import DataLoader
import time
from module_GAN import mv_gan
import torch
from sklearn.metrics import f1_score
import argparse

def main(args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    train_dataset = load_mnist('train', args.num)
    test_dataset = load_mnist('test', args.num)
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    size_1 = 784
    size_2 = 784
    iterations = args.epoch * len(train_loader)
    model = mv_gan(train_dataset.num_class, dim_in_1=size_1, dim_in_2=size_2, gpu=args.gpu,
                   dim_hidden=args.hidden, k_1=2, k_2=2, k_3=4, alpha=args.alpha, beta=args.beta).to(device)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True, num_workers=8)
    print('The size of training samples: {}\nBatch size: {}'.format(len(train_dataset), batch_size))
    print('Initial learning rate: {}\nThe size of hidden layer: {}'.format(args.lr, args.hidden))
    model = model.double()
    model.learning_rate = args.lr
    model.loss = args.reg
    model.epoch = args.epoch
    train_model(model, train_loader, iterations, args)
    print('Finish training process!')
    evaluation(model, test_loader)


def train_model(model, train_loader, iterations, args, print_frequency=100):
    epoch = 0
    cur_iter = 0
    momentum = 0.95
    while cur_iter < iterations:
        model.learning_rate = model.learning_rate / (1 + 0.005 * epoch)
        start_time = time.time()
        epoch += 1
        for i, sample_batched in enumerate(train_loader):
            x_1 = sample_batched['view_1'].double().to(model.device)
            x_2 = sample_batched['view_2'].double().to(model.device)
            labels = sample_batched['label'].double().to(model.device)

            # Step 1: we only update the discriminator, thus we freeze the rest neural network
            optimizer = torch.optim.SGD(model.parameters(), lr=model.learning_rate, momentum=momentum)
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
                            param.data.clamp_(-args.clamp, args.clamp)
                        except:
                            continue
            for j in range(model.k_1):
                optimizer.zero_grad()
                loss, _ = model(x_1, x_2, labels, stage=1)
                loss.backward()
                optimizer.step()

            # Step 2: we only update the generators, thus we freeze the rest neural network
            for child in model.children():
                # if child.name == 'generator_g' or child.name == 'generator_f' or child.name == 'classifier':
                if child.name == 'generator_g' or child.name == 'generator_f':
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
            for j in range(model.k_2):
                optimizer.zero_grad()
                loss, _ = model(x_1, x_2, labels, stage=2)
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
            for j in range(model.k_3):
                optimizer.zero_grad()
                loss, pred = model(x_1, x_2, labels, stage=2)
                loss.backward()
                optimizer.step()
            if cur_iter % print_frequency == 0:
                _, predicted_label = torch.max(pred.data, 1)
                true_label = torch.tensor(np.array([list(labels[i]).index(1) for i in range(labels.shape[0])]))
                f1 = f1_score(true_label, predicted_label.cpu().numpy(), average='weighted')
                acc = np.double((predicted_label == true_label.to(model.device).long()).sum()) / labels.size(0)
                print('Iter [{}/{}], Loss: {:.4f}, Acc: {:.4f}, F1:{:.4f}'.format(cur_iter, iterations, loss, acc, f1))
            cur_iter += 1
        print("--- %s seconds for epoch %d---" % (time.time() - start_time, epoch))
    # Save the model checkpoint
    torch.save(model.state_dict(), 'model_{}_cycle_loss_{}_{}.ckpt'.format(epoch, model.loss, args.hidden))


def evaluation(model, test_loader):
    model.eval()  # eval mode (batch norm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        label_correct = 0
        total = 0
        prediction = []
        truth = []
        for i, sample_batched in enumerate(test_loader):
            x_1 = sample_batched['view_1'].double().to(model.device)
            x_2 = sample_batched['view_2'].double().to(model.device)
            labels = sample_batched['label'].double()
            _, outputs = model(x_1, x_2, stage=4)
            _, predicted_label = torch.max(outputs.data, 1)
            predicted = np.array([[1 if j == a else 0 for j in range(labels.shape[1])] for a in predicted_label.data])
            true_label = torch.tensor(np.array([list(labels[i]).index(1) for i in range(labels.shape[0])])).to(model.device)
            total += labels.size(0)
            label_correct += (predicted_label == true_label.long()).sum().item()
            prediction = prediction + list(predicted)
            truth = truth + list(labels.numpy())
        print('The Accuracy of the predictions on the {} test documents: {:.4f} %'.format(
            test_loader.dataset.label.shape[0], 100 * label_correct / total))
        print('F1 scores of the model on the {} test documents: {:.4f} %'.format(
            test_loader.dataset.label.shape[0], 100 * f1_score(np.array(truth), np.array(prediction), average='weighted')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepMUSE algorithm')
    parser.add_argument('-d', dest='data', type=str, default='mnist', help='which dataset is used for demo')
    parser.add_argument('-g', dest='gpu', type=int, default=0, help='the index of the gpu to use')
    parser.add_argument('-l', dest='reg', type=int, default=3, help='which regularizer to use')
    parser.add_argument('-lr', dest='lr', type=float, default=0.02, help='The learning rate')
    parser.add_argument('-e', dest='epoch', type=int, default=20, help='the total epoch for training')
    parser.add_argument('-hid', dest='hidden', type=int, default=100, help='the size of the hidden representation')
    parser.add_argument('-n', dest='num', type=int, default=32000, help='the size of training data set')
    parser.add_argument('-clamp', dest='clamp', type=float, default=0.02, help='the value of clap')
    parser.add_argument('-alpha', dest='alpha', type=float, default=1, help='the value of alpha')
    parser.add_argument('-beta', dest='beta', type=float, default=2, help='the value of beta')
    args = parser.parse_args()
    main(args)
