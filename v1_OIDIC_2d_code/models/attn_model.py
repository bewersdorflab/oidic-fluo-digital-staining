import os
from math import log10

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as utils
from tqdm import tqdm

from networks.networks import gaussian_weights_init
from networks.unet import AttnNet
from utils.utils import display_transform
from models.utils import AverageMeter, get_scheduler


class AttnModel(nn.Module):
    def __init__(self, cfg):
        super(AttnModel, self).__init__()
        self.netG = AttnNet()

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=cfg['lr'])
        self.criterion_L1 = nn.L1Loss().cuda()

        if cfg['recon'] == 'L2':
            self.criterion_recon = nn.MSELoss().cuda()
        elif cfg['recon'] == 'L1':
            self.criterion_recon = nn.L1Loss().cuda()

        self.cfg = cfg

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))
        self.netG.to(gpu_ids[0])
        self.netG = nn.DataParallel(self.netG, device_ids=gpu_ids)

    def initialize(self):
        self.netG.apply(gaussian_weights_init)

    def set_scheduler(self, cfg):
        self.scheduler_G = get_scheduler(self.optimizer_G, cfg)

    def set_input(self, data):
        self.cbct = data['cbct'].to(self.device)
        self.ct = data['ct'].to(self.device)

    def forward(self, input):
        return self.netG(input)

    def optimize(self):
        self.optimizer_G.zero_grad()
        ct_fake, mask = self.netG(self.cbct)
        self.loss_G = self.criterion_recon(ct_fake, self.ct)
        self.loss_G.backward()
        self.optimizer_G.step()

    @property
    def loss_summary(self):
        return 'loss_G(L2): {:4f}'.format(self.loss_G.item())

    def update_learning_rate(self):
        pass

    def save(self, checkpoint_dir, epoch):
        checkpoint_name = os.path.join(checkpoint_dir, 'model_{}.pt'.format(epoch))
        torch.save({'netG': self.netG.module.state_dict(),
                    'optimizer_G': self.optimizer_G.state_dict(),
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

