import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


class mv_gan(nn.Module):
    def __init__(self, num_class, dim_in_1=784, dim_in_2=784, dim_hidden=100,
                 k_1=3, k_2=2, k_3=2,  alpha=1, beta=1, gpu=0):
        super(mv_gan, self).__init__()
        self.device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
        self.k_1 = k_1
        self.k_2 = k_2
        self.k_3 = k_3
        self.discriminator_loss_coef = 1
        self.cycle_loss_coef = alpha
        self.beta = beta
        hidden_layer = 512
        # the first view
        self.generator_g_1 = nn.Sequential(nn.Linear(dim_in_1, hidden_layer), nn.ReLU(inplace=True),
                                           nn.Linear(hidden_layer, hidden_layer), nn.ReLU(inplace=True),
                                           nn.Linear(hidden_layer, dim_hidden))
        self.generator_f_1 = nn.Sequential(nn.Linear(dim_hidden, hidden_layer), nn.ReLU(inplace=True),
                                           nn.Linear(hidden_layer, hidden_layer), nn.ReLU(inplace=True),
                                           nn.Linear(hidden_layer, dim_in_1))
        self.generator_g_1.name = 'generator_g'
        self.generator_f_1.name = 'generator_f'
        # the second view
        self.generator_g_2 = nn.Sequential(nn.Linear(dim_in_2, hidden_layer), nn.ReLU(inplace=True),
                                           nn.Linear(hidden_layer, hidden_layer), nn.ReLU(inplace=True),
                                           nn.Linear(hidden_layer, dim_hidden))
        self.generator_f_2 = nn.Sequential(nn.Linear(dim_hidden, hidden_layer), nn.ReLU(inplace=True),
                                           nn.Linear(hidden_layer, hidden_layer), nn.ReLU(inplace=True),
                                           nn.Linear(hidden_layer, dim_in_2))
        self.generator_g_2.name = 'generator_g'
        self.generator_f_2.name = 'generator_f'
        self.hid_layer_1 = nn.Sequential(nn.Linear(dim_in_1, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                          nn.Linear(256, dim_hidden), nn.BatchNorm1d(dim_hidden), nn.ReLU(inplace=True))
        self.hid_layer_2 = nn.Sequential(nn.Linear(dim_in_2, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                          nn.Linear(256, dim_hidden), nn.BatchNorm1d(dim_hidden), nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(nn.Linear(dim_hidden * 3, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                          nn.Linear(256, num_class))
        self.discriminator = nn.Sequential(nn.Linear(dim_hidden, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                                           nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                           nn.Linear(256, 1), nn.Sigmoid())
        self.hid_layer_1.name = 'classifier'
        self.hid_layer_2.name = 'classifier'
        self.classifier.name = 'classifier'
        self.discriminator.name = 'discriminator'
        self.dropout = nn.Dropout(0.5)
        self.dropout.name = 'dropout'

    def mae_criterion(self, in_, target):
        return torch.mean((in_ - target)**2)


    def forward(self, x_1, x_2, labels=None, stage=1):
        z_1 = self.generator_g_1(x_1)
        z_2 = self.generator_g_2(x_2)
        if stage == 1:
            real_z = self.discriminator(z_1)
            real_z_loss = self.mae_criterion(real_z, torch.ones(real_z.shape, dtype=torch.float64).to(self.device))
            fake_z = self.discriminator(z_2)
            fake_z_loss = self.mae_criterion(fake_z, torch.zeros(real_z.shape, dtype=torch.float64).to(self.device))
            return fake_z_loss + real_z_loss, None
        elif stage == 2:
            real_z = self.discriminator(z_1)
            real_z_loss = self.mae_criterion(real_z, torch.zeros(real_z.shape, dtype=torch.float64).to(self.device))
            fake_z = self.discriminator(z_2)
            fake_z_loss = self.mae_criterion(fake_z, torch.ones(fake_z.shape, dtype=torch.float64).to(self.device))
            discriminator_loss = (fake_z_loss + real_z_loss) / 2
            loss, pred = self.clf_loss(z_1, z_2, x_1, x_2, labels, stage)
            loss = loss + self.discriminator_loss_coef * discriminator_loss
            return loss, pred
        else:
            return self.clf_loss(z_1, z_2, x_1, x_2, labels, stage)

    def clf_loss(self, z_1, z_2, x_1, x_2, labels=None, stage=2):
        fake_x_1_z1 = self.generator_f_1(z_1)
        fake_x_1_z2 = self.generator_f_1(z_2)
        fake_x_2_z1 = self.generator_f_2(z_1)
        fake_x_2_z2 = self.generator_f_2(z_2)
        z_s = (z_1 + z_2) / 2
        fake_x_1 = (fake_x_1_z1 + fake_x_1_z2) / 2
        fake_x_2 = (fake_x_2_z1 + fake_x_2_z2) / 2
        r_3 = self.hid_layer_1(self.dropout(x_1 - fake_x_1))
        r_4 = self.hid_layer_2(self.dropout(x_2 - fake_x_2))
        representation = torch.cat([z_s, r_3, r_4], 1)
        pred = F.softmax(self.classifier(representation), dim=1)
        if stage == 2:
            loss = F.binary_cross_entropy_with_logits(pred, labels) * self.beta
            constr_loss = self.cycle_loss_3(x_1, x_2, fake_x_1_z1, fake_x_2_z2, fake_x_1_z2, fake_x_2_z1, z_1, z_2)
            return constr_loss + loss, pred
        else:
            return None, pred

    def cycle_loss_1(self, x_1, x_2, fake_x_1, fake_x_2):
        loss_x_1 = torch.mean(torch.abs(fake_x_1 - x_1))
        loss_x_2 = torch.mean(torch.abs(fake_x_2 - x_2))
        cycle_loss = (loss_x_1 + loss_x_2) / 2
        loss = self.cycle_loss_coef * cycle_loss
        return loss

    # the variant of cycle-consistency loss
    def cycle_loss_2(self, x_1, x_2, fake_x_1, fake_x_2):
        loss_x_1 = torch.mean(torch.abs(fake_x_1 - x_1))
        loss_x_2 = torch.mean(torch.abs((fake_x_2 - x_2)))
        cycle_loss = (loss_x_1 + loss_x_2) / 2
        loss = self.cycle_loss_coef * cycle_loss
        return loss

    # the cross reconstruction loss
    def cycle_loss_3(self, x_1, x_2, fake_x_1, fake_x_2, fake_x_1_z2, fake_x_2_z1, z_1, z_2):
        loss_x_1 = torch.mean(torch.abs(fake_x_1 - x_1))
        loss_x_2 = torch.mean(torch.abs(fake_x_2 - x_2))
        loss_x_3 = torch.mean(torch.abs(fake_x_1_z2 - x_1))
        loss_x_4 = torch.mean(torch.abs(fake_x_2_z1 - x_2))
        cycle_loss = (loss_x_1 + loss_x_2 + loss_x_3 + loss_x_4 ) / 4
        loss = self.cycle_loss_coef * cycle_loss
        return loss

    # L1 norm of the difference between two representations
    def cycle_loss_4(self, z_1, z_2):
        loss = torch.mean(torch.abs(z_1 - z_2))
        return self.cycle_loss_coef * loss
