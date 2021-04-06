import torch
import torch.nn as nn
import torch.nn.functional as F


class mv_gan(nn.Module):
    def __init__(self, num_class, dim_in_1=784, dim_in_2=784, dim_hidden=100, k_1=3, k_2=4,
                 t_1=3, t_2=2, t_3=2,  alpha=1, beta=1, gpu=0):
        super(mv_gan, self).__init__()
        hidden_layer = 300
        self.hidden_layer = hidden_layer
        self.device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
        self.t_1 = t_1
        self.t_2 = t_2
        self.t_3 = t_3
        self.discriminator_loss_coef = 1
        self.cycle_loss_coef = alpha
        self.beta = beta
        self.k_1 = k_1
        self.k_2 = k_2
        self.dim_in_1 = dim_in_1
        self.hid_d_1 = int(self.dim_in_1 / self.k_1)
        self.dim_in_2 = dim_in_2
        self.hid_d_2 = int(self.dim_in_2 / self.k_2)
        # the first view
        self.generator_g_1 = nn.Sequential(nn.Linear(self.hid_d_1, self.hid_d_1))
        self.soft_att_1_1 = nn.Sequential(nn.Linear(self.hid_d_1, dim_hidden))
        self.soft_att_1_2 = nn.Sequential(nn.Linear(dim_hidden, 1))
        self.generator_g_1_2 = nn.Sequential(nn.Linear(self.hid_d_1, dim_hidden))
        self.generator_f_1 = nn.Sequential(nn.Linear(dim_hidden, hidden_layer), nn.ReLU(inplace=True),
                                           nn.BatchNorm1d(hidden_layer), nn.Linear(hidden_layer, self.dim_in_1))
        self.generator_g_1.name = 'generator'
        self.soft_att_1_1.name = 'generator'
        self.soft_att_1_2.name = 'generator'
        self.generator_g_1_2.name = 'generator'
        self.generator_f_1.name = 'generator'
        # the second view
        self.generator_g_2 = nn.Sequential(nn.Linear(self.hid_d_2, self.hid_d_2))
        self.soft_att_2_1 = nn.Sequential(nn.Linear(self.hid_d_2, dim_hidden))
        self.soft_att_2_2 = nn.Sequential(nn.Linear(dim_hidden, 1))
        self.generator_g_2_2 = nn.Sequential(nn.Linear(self.hid_d_2, dim_hidden))
        self.generator_f_2 = nn.Sequential(nn.Linear(dim_hidden, hidden_layer), nn.ReLU(inplace=True),
                                           nn.BatchNorm1d(hidden_layer), nn.Linear(hidden_layer, self.dim_in_2))
        self.generator_g_2.name = 'generator'
        self.soft_att_2_1.name = 'generator'
        self.soft_att_2_2.name = 'generator'
        self.generator_g_2_2.name = 'generator'
        self.generator_f_2.name = 'generator'
        self.attn_map = nn.Sequential(nn.Linear(self.hid_d_1, self.hid_d_2))
        self.attn_map.name = 'generator'
        compl_dim = max(int(dim_hidden / 2), 10)
        self.hid_layer_1 = nn.Sequential(nn.Linear(dim_in_1, 256), nn.ReLU(inplace=True), nn.Linear(256, compl_dim))
        self.hid_layer_2 = nn.Sequential(nn.Linear(dim_in_2, 256), nn.ReLU(inplace=True), nn.Linear(256, compl_dim))
        self.classifier = nn.Sequential(nn.Linear(dim_hidden, 256), nn.BatchNorm1d(256),
                                        nn.ReLU(inplace=True), nn.Linear(256, num_class))
        self.classifier_1 = nn.Sequential(nn.Linear(dim_hidden + compl_dim, 256), nn.BatchNorm1d(256),
                                        nn.ReLU(inplace=True), nn.Linear(256, num_class))
        self.classifier_2 = nn.Sequential(nn.Linear(dim_hidden + compl_dim, 256), nn.BatchNorm1d(256),
                                          nn.ReLU(inplace=True), nn.Linear(256, num_class))
        self.discriminator = nn.Sequential(nn.Linear(dim_hidden, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                                           nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                           nn.Linear(256, 1), nn.Sigmoid())
        self.hid_layer_1.name = 'generator'
        self.hid_layer_2.name = 'generator'
        self.classifier.name = 'classifier'
        self.classifier_1.name = 'classifier'
        self.classifier_2.name = 'classifier'
        self.discriminator.name = 'discriminator'
        self.dropout = nn.Dropout(0.3)
        self.dropout.name = 'dropout'
        self.view_weight_raw = nn.Parameter(torch.ones(2, requires_grad=False, device=self.device))
        self.eps = 0.3

    def mae_criterion(self, in_, target):
        return torch.mean((in_ - target)**2)

    def cosine_distance_torch(self, x1, x2, eps=1e-15):
        w1 = x1.norm(p=2, dim=0, keepdim=True)
        w2 = x2.norm(p=2, dim=0, keepdim=True)
        return torch.mm(x1.t(), x2) / (w1.t() * w2).clamp(min=eps)

    def cosine_distance_torch2(self, x1, x2, eps=1e-15):
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        return torch.bmm(x1.transpose(1, 2), x2) / torch.bmm(w1.transpose(1, 2), w2).clamp(min=eps)

    def forward(self, x_1, x_2, labels=None, stage=1, name=None):
        if len(x_1.shape) != 3:
            x_1 = x_1.reshape(-1, self.k_1, self.hid_d_1)
            x_2 = x_2.reshape(-1, self.k_2, self.hid_d_2)
            z_1 = self.generator_g_1(x_1).transpose(1, 2)
            z_2 = self.generator_g_2(x_2).transpose(1, 2)
        else:
            # z_1 = self.generator_g_1(x_1).transpose(1, 2)
            # z_2 = self.generator_g_2(x_2).transpose(1, 2)
            z_1 = x_1.transpose(1, 2)
            z_2 = x_2.transpose(1, 2)
        # co-attention map
        atten = torch.tanh(torch.matmul(self.attn_map(z_1.transpose(1, 2)), z_2))
        a = self.soft_att_1_1(z_1.transpose(1, 2)).transpose(1, 2)
        b = self.soft_att_2_1(z_2.transpose(1, 2)).transpose(1, 2)
        hv_1 = torch.tanh(a + torch.bmm(b, atten.transpose(1, 2)))
        hv_2 = torch.tanh(torch.bmm(a, atten) + b)
        sv_1 = F.softmax(self.soft_att_1_2(hv_1.transpose(1, 2)), dim=1)
        sv_2 = F.softmax(self.soft_att_2_2(hv_2.transpose(1, 2)), dim=1)
        z_1 = torch.bmm(z_1, sv_1).reshape(-1, self.hid_d_1)
        z_2 = torch.bmm(z_2, sv_2).reshape(-1, self.hid_d_2)
        torch.set_printoptions(precision=4, sci_mode=False)
        sv_1 = sv_1.reshape(-1, self.k_1)
        sv_2 = sv_2.reshape(-1, self.k_2)
        z_1 = self.generator_g_1_2(z_1)
        z_2 = self.generator_g_2_2(z_2)
        self.view_weight = (self.view_weight_raw / self.view_weight_raw.sum(0, keepdim=True)).clamp(min=self.eps, max=0.7)
        if stage == 1:
            real_z = self.discriminator(z_1)
            real_z_loss = self.mae_criterion(real_z, torch.ones(real_z.shape, dtype=torch.float64).to(self.device))
            fake_z = self.discriminator(z_2)
            fake_z_loss = self.mae_criterion(fake_z, torch.zeros(real_z.shape, dtype=torch.float64).to(self.device))
            return fake_z_loss + real_z_loss, None, [sv_1, sv_2]
        elif stage == 2:
            real_z = self.discriminator(z_1)
            real_z_loss = self.mae_criterion(real_z, torch.zeros(real_z.shape, dtype=torch.float64).to(self.device))
            fake_z = self.discriminator(z_2)
            fake_z_loss = self.mae_criterion(fake_z, torch.ones(fake_z.shape, dtype=torch.float64).to(self.device))
            discriminator_loss = (fake_z_loss + real_z_loss)
            loss, pred, _ = self.clf_loss(z_1, z_2, x_1, x_2, sv_1, sv_2, labels, stage)
            loss = loss + self.discriminator_loss_coef * discriminator_loss
            diff = torch.mean(torch.abs(z_1 - z_2))
            return loss, pred, diff
        else:
            _, pred, diff = self.clf_loss(z_1, z_2, x_1, x_2, sv_1, sv_2,  labels, stage)
            # print(name)
            # print('sv_1 ={},\n sv_2={}'.format(sv_1, sv_2))
            # print('prediction={}'.format(torch.max(pred, dim=1)[1] + torch.tensor([1, 1, 1]).to(self.device)))
            return atten, pred, [sv_1.mean(dim=0), sv_2.mean(dim=0)]

    def clf_loss(self, z_1, z_2, x_1, x_2, sv_1, sv_2, labels=None, stage=2):
        fake_x_1_z1 = self.generator_f_1(z_1).reshape(-1, self.k_1, self.hid_d_1)
        fake_x_1_z2 = self.generator_f_1(z_2).reshape(-1, self.k_1, self.hid_d_1)
        fake_x_2_z1 = self.generator_f_2(z_1).reshape(-1, self.k_2, self.hid_d_2)
        fake_x_2_z2 = self.generator_f_2(z_2).reshape(-1, self.k_2, self.hid_d_2)
        z_s = (z_1 + z_2) / 2
        diff = torch.mean(torch.abs(z_1 - z_2))
        fake_x_1 = (fake_x_1_z1 + fake_x_1_z2) / 2
        fake_x_2 = (fake_x_2_z1 + fake_x_2_z2) / 2
        r_3 = self.hid_layer_1(self.dropout((x_1 - fake_x_1).reshape(-1, self.dim_in_1)))
        r_4 = self.hid_layer_2(self.dropout((x_2 - fake_x_2).reshape(-1, self.dim_in_2)))
        representation_1 = torch.cat([z_1, r_3], 1)
        representation_2 = torch.cat([z_2, r_4], 1)
        cm_pred = F.softmax(self.classifier(z_s), dim=1)
        pred_1 = F.softmax(self.classifier_1(representation_1), dim=1)
        pred_2 = F.softmax(self.classifier_2(representation_2), dim=1)
        pred = self.view_weight[0] * pred_1 + self.view_weight[1] * pred_2
        if stage == 2:
            loss = (F.binary_cross_entropy_with_logits(pred, labels) +
                    F.binary_cross_entropy_with_logits(cm_pred, labels)) * self.beta
            constr_loss = self.cycle_loss_3(x_1, x_2, fake_x_1_z1, fake_x_2_z2, fake_x_1_z2, fake_x_2_z1,
                                            z_1, z_2, sv_1, sv_2)
            return constr_loss + loss, pred, diff
        else:
            return None, pred, diff

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

    # the cross cycle-consistency loss
    def cycle_loss_3(self, x_1, x_2, fake_x_1, fake_x_2, fake_x_1_z2, fake_x_2_z1, z_1, z_2, sv_1, sv_2):
        loss_x_1 = torch.mean(sv_1 * torch.mean(torch.abs(fake_x_1 - x_1), dim=2))
        loss_x_2 = torch.mean(sv_2 * torch.mean(torch.abs(fake_x_2 - x_2), dim=2))
        loss_x_3 = torch.mean(sv_1 * torch.mean(torch.abs(fake_x_1_z2 - x_1), dim=2))
        loss_x_4 = torch.mean(sv_2 * torch.mean(torch.abs(fake_x_2_z1 - x_2), dim=2))
        loss = torch.mean(torch.abs(z_1 - z_2))
        cycle_loss = (loss_x_1 + loss_x_2 + loss_x_3 + loss_x_4 + loss) / 5
        loss = self.cycle_loss_coef * cycle_loss
        return loss

    # L1 norm of the difference between two representations
    def cycle_loss_4(self, z_1, z_2):
        loss = torch.mean(torch.abs(z_1 - z_2))
        return self.cycle_loss_coef * loss
