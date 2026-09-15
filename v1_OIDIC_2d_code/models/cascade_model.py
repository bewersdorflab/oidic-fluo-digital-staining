import os
from math import log10
from collections import OrderedDict
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from networks import get_generator, get_discriminator, get_FBP, get_FP
from networks.networks import gaussian_weights_init
from models.utils import AverageMeter, get_scheduler, get_gan_loss, psnr, get_nonlinearity


class CascadeModel(nn.Module):
    def __init__(self, opts):
        super(CascadeModel, self).__init__()

        self.loss_names = []
        self.networks = []
        self.optimizers = []

        # set default loss flags
        loss_flags = ("w_img_L1")
        for flag in loss_flags:
            if not hasattr(opts, flag): setattr(opts, flag, 0)

        self.is_train = True if hasattr(opts, 'lr') else False

        self.net_G = get_generator(opts.net_G, opts)
        self.networks.append(self.net_G)

        if self.is_train:
            self.loss_names += ['loss_G_L1']
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(),
                                                lr=opts.lr,
                                                betas=(opts.beta1, opts.beta2),
                                                weight_decay=opts.weight_decay)
            self.optimizers.append(self.optimizer_G)

        self.criterion = nn.L1Loss()

        self.opts = opts

        # Data Consistency Layers
        self.nc = opts.nc

        self.FP = get_FP(opts)
        self.FBP = get_FBP(opts)

        dcs = []
        for i in range(self.nc):
            dcs.append(DataConsistencyInSinogram(noise_lvl=None, FP=self.FP, FBP=self.FBP))
        self.dcs = dcs

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    def set_scheduler(self, opts, epoch=-1):
        self.schedulers = [get_scheduler(optimizer, opts, last_epoch=epoch) for optimizer in self.optimizers]

    def set_input(self, data):
        self.spacing = data['spacing'].to(self.device)
        self.gt_CT = data['gt_CT'].to(self.device)

        self.lv_CT = data['lv_CT'].to(self.device)
        self.lv_sinogram = data['lv_sinogram'].to(self.device)

        self.dv_CT = data['dv_CT'].to(self.device)
        self.dv_sinogram = data['dv_sinogram'].to(self.device)

        self.mask_sinogram = data['mask_sinogram'].to(self.device)

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def forward(self):
        x = self.lv_CT
        x.requires_grad_(True)

        net = {}
        for i in range(1, self.nc+1):
            x = x.contiguous()

            net['r%d_img_pred' % i] = self.net_G(x)  # output CNN image
            net['r%d_img_dc_pred' % i] = self.dcs[i-1](net['r%d_img_pred' % i],
                                                       self.dv_sinogram,
                                                       self.mask_sinogram,
                                                       self.spacing)   # output data consistency images

            x = net['r%d_img_dc_pred' % i]

        self.refine_CT = net['r%d_img_dc_pred' % i]

    def update_G(self):
        loss_G_L1 = 0

        self.optimizer_G.zero_grad()

        loss_G_L1 = self.criterion(self.refine_CT, self.dv_CT)
        self.loss_G_L1 = loss_G_L1.item()

        total_loss = loss_G_L1
        total_loss.backward()
        self.optimizer_G.step()

    def optimize(self):
        self.loss_G_L1 = 0

        self.forward()
        self.update_G()

    @property
    def loss_summary(self):
        message = ''
        if self.opts.wr_L1 > 0:
            message += 'G_L1: {:.4e} '.format(self.loss_G_L1)

        return message

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = {:7f}'.format(lr))

    def save(self, filename, epoch, total_iter):

        state = {}
        if self.opts.wr_L1 > 0:
            state['net_G'] = self.net_G.module.state_dict()
            state['opt_G'] = self.optimizer_G.state_dict()

        state['epoch'] = epoch
        state['total_iter'] = total_iter

        torch.save(state, filename)
        print('Saved {}'.format(filename))

    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file)

        if self.opts.wr_L1 > 0:
            self.net_G.module.load_state_dict(checkpoint['net_G'])
            if train:
                self.optimizer_G.load_state_dict(checkpoint['opt_G'])

        print('Loaded {}'.format(checkpoint_file))

        return checkpoint['epoch'], checkpoint['total_iter']

    def evaluate(self, loader):
        val_bar = tqdm(loader)
        avg_psnr_sinogram = AverageMeter()
        avg_psnr_sinogram_FBP = AverageMeter()
        avg_psnr_CT = AverageMeter()

        sinogram_images = []
        sinogram_FBP_images = []
        CT_images = []

        for data in val_bar:
            self.set_input(data)
            self.forward()

            # loss_sinogram = (self.output_sinogram - self.dv_sinogram).pow(2).mean().item()
            # max_sino = self.dv_sinogram.max().pow(2).item()
            # psnr_sinogram = -10 * log10(loss_sinogram / max_sino)
            #
            # psnr_sinogram_FBP = psnr(self.output_sinogram_FBP, self.dv_CT)
            #
            # avg_psnr_sinogram.update(psnr_sinogram)
            # avg_psnr_sinogram_FBP.update(psnr_sinogram_FBP)
            #
            # sinogram_images.append(self.output_sinogram[0].cpu())
            # sinogram_FBP_images.append(self.output_sinogram_FBP[0].cpu())

            if self.opts.wr_L1 > 0:
                psnr_CT = psnr(self.refine_CT, self.dv_CT)
                avg_psnr_CT.update(psnr_CT)
                CT_images.append(self.refine_CT[0].cpu())

            message = 'PSNR CT: {:4f} '.format(avg_psnr_CT.avg)
            # message += 'PSNR sinogram: {:4f} '.format(avg_psnr_sinogram.avg)
            # message += 'PSNR sinogram FBP: {:4f} '.format(avg_psnr_sinogram_FBP.avg)
            val_bar.set_description(desc=message)

        # self.psnr_sinogram = avg_psnr_sinogram.avg
        # self.psnr_sinogram_FBP = avg_psnr_sinogram_FBP.avg
        self.psnr_CT = avg_psnr_CT.avg

        self.results = {}
        # self.results['sinogram'] = torch.stack(sinogram_images).squeeze().numpy()
        # self.results['sinogram_FBP'] = torch.stack(sinogram_FBP_images).squeeze().numpy()

        if self.opts.wr_L1 > 0:
            self.results['CT'] = torch.stack(CT_images).squeeze().numpy()


def data_consistency(s, s0, mask, noise_lvl=None):
    """
    s    - input in sinogram
    s0   - initially sampled elements in sinogram
    mask - corresponding nonzero location
    """
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * s + mask * (s + v * s0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * s + mask * s0
    return out


class DataConsistencyInSinogram(nn.Module):
    """ Create data consistency operator"""

    def __init__(self, noise_lvl=None, FP=None, FBP=None):
        super(DataConsistencyInSinogram, self).__init__()
        self.noise_lvl = noise_lvl

        self.FBP = FBP
        self.FP = FP

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, s0, mask, spacing):
        """
        x    - input in image domain, of shape (n, 1, nx, ny)
        s0   - initially sampled elements in sinogram
        mask - corresponding nonzero location
        """

        s = (self.FP(x.squeeze(1)).unsqueeze(1) * spacing.unsqueeze(3)).transpose(2, 3).contiguous()
        s_dcs = data_consistency(s, s0, mask, self.noise_lvl)
        x_out = self.FBP(s_dcs.squeeze(1)).unsqueeze(1) / spacing.unsqueeze(3)

        return x_out