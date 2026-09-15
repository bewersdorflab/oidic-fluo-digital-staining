import os
from math import log10

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as utils
from tqdm import tqdm

from models.utils import GANLoss, SparseLoss
from networks.networks import Dis, gaussian_weights_init
from networks.unet import AttnNet
from models.utils import AverageMeter, get_scheduler
from utils.utils import display_transform


class AttnModel(nn.Module):
    def __init__(self, cfg):
        super(AttnModel, self).__init__()
        self.netG = AttnNet()

        # add norm layer here
        self.netD = Dis(input_dim=1, n_layer=3)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=cfg['lr'])
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=cfg['lr'])

        self.criterion_GAN = GANLoss().cuda()
        self.criterion_L1 = nn.L1Loss().cuda()
        self.criterion_sparse = SparseLoss().cuda()

        self.cfg = cfg

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))
        self.netG.to(gpu_ids[0])
        self.netD.to(gpu_ids[0])
        self.netG = nn.DataParallel(self.netG, device_ids=gpu_ids)
        self.netD = nn.DataParallel(self.netD, device_ids=gpu_ids)

    def initialize(self):
        self.netG.apply(gaussian_weights_init)
        self.netD.apply(gaussian_weights_init)

    def set_scheduler(self, cfg):
        self.scheduler_G = get_scheduler(self.optimizer_G, cfg)
        self.scheduler_D = get_scheduler(self.optimizer_D, cfg)

    def set_input(self, data):
        self.cbct = data['cbct'].to(self.device)
        self.ct = data['ct'].to(self.device)

    def forward(self, input):
        return self.netG(input)

    def optimize(self):

        self.optimizer_D.zero_grad()
        # fake
        ct_fake, _ = self.netG(self.cbct)
        pred_fake = self.netD(ct_fake.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, target_is_real=False)
        # real
        pred_real = self.netD(self.ct)
        loss_D_real = self.criterion_GAN(pred_real, target_is_real=True)
        self.loss_D = (loss_D_fake + loss_D_real) * 0.5
        self.loss_D.backward()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        ct_fake, mask = self.netG(self.cbct)
        pred_fake = self.netD(ct_fake)
        self.loss_G_GAN = self.criterion_GAN(pred_fake, target_is_real=True)
        self.loss_G_recon = self.criterion_L1(ct_fake, self.ct)
        self.loss_G_sparse = self.criterion_sparse(mask)

        self.loss_G = self.loss_G_GAN + self.loss_G_recon * self.cfg['recon_w'] + self.loss_G_sparse * self.cfg['sparse_w']
        self.loss_G.backward()
        self.optimizer_G.step()

    @property
    def loss_summary(self):
        return 'loss_D: {:4f}, loss_G(GAN): {:4f}, loss_G(recon): {:4f}, loss_sparse: {:4f}'.format(self.loss_D.item(),
                                                                                                    self.loss_G_GAN.item(),
                                                                                                    self.loss_G_recon.item(),
                                                                                                    self.loss_G_sparse.item())

    def update_learning_rate(self):
        pass

    def save(self, checkpoint_dir, epoch):
        checkpoint_name = os.path.join(checkpoint_dir, 'model_{}.pt'.format(epoch))
        torch.save({'netG': self.netG.module.state_dict(),
                    'optimizer_G': self.optimizer_G.state_dict(),
                    'netD': self.netD.module.state_dict(),
                    'optimizer_D': self.optimizer_D.state_dict(),
                    'epoch': epoch},
                   checkpoint_name)

    def resume(self):
        pass

    def evaluate(self, loader, epoch, image_directory):
        val_bar = tqdm(loader)
        val_images = []
        mask_images = []
        avg_psnr = AverageMeter()

        for data in val_bar:
            self.set_input(data)
            recon, mask = self.netG(self.cbct)

            mask = mask.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
            mask -= np.min(mask)
            mask /= np.max(mask)
            mask *= 255.0
            mask = cv2.applyColorMap(np.uint8(mask), cv2.COLORMAP_JET)
            mask_images.append(mask)

            loss = (recon - self.ct).pow(2).mean().item()
            avg_psnr.update(-10 * log10(loss))

            val_bar.set_description(desc='PSNR:{:4f}'.format(avg_psnr.val))

            val_images.extend(
                [display_transform(self.cbct), display_transform(recon), display_transform(self.ct)])

        val_images = torch.stack(val_images)
        val_images = val_images.split(15)
        val_save_bar = tqdm(val_images, desc='[saving training results]')
        index = 1
        for image in val_save_bar:
            image = utils.make_grid(image, nrow=3, padding=5)
            utils.save_image(image, os.path.join(image_directory, 'ep{}_ind{}.png'.format(epoch, index)), padding=5)
            index += 1

        mask_images = [mask_images[i:i + 5] for i in range(0, len(mask_images), 5)]
        for i, m in enumerate(mask_images):
            image = np.concatenate(m, axis=1)
            cv2.imwrite(os.path.join(image_directory, 'mask_ep{}_ind{}.png'.format(epoch, i + 1)), image)

