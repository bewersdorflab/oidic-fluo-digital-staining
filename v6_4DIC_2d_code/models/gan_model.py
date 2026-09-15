from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm

from models.utils import LSGANLoss    # from models.utils import GANLoss
from networks.networks import Dis, gaussian_weights_init
from models.utils import AverageMeter, get_scheduler, get_gan_loss, psnr, mse, get_nonlinearity
from skimage.metrics import structural_similarity as ssim
from networks import get_generator, get_discriminator

class GANModel(nn.Module):
    def __init__(self, opts):
        super(GANModel, self).__init__()

        self.loss_names = []
        self.networks = []
        self.optimizers = []

        self.net_G = get_generator(opts.net_G, opts)  # [batch, ch, 32, 32, 32] -- [batch, 1, 32 ,32 ,32]
        self.net_D = get_discriminator(opts)   # [batch, 32, 32, 32] -- [batch, 1, 5, 5]

        self.networks.append(self.net_G)
        self.networks.append(self.net_D)

        # Optimizer
        self.loss_names += ['loss_G_GAN']
        self.loss_names += ['loss_G_recon']
        self.loss_names += ['loss_G']  # the name of the generator loss
        self.optimizer_G = torch.optim.Adam(self.net_G.parameters(),
                                            lr=opts.lr,
                                            betas=(opts.beta1, opts.beta2),
                                            weight_decay=opts.weight_decay)

        self.loss_names += ['loss_D']
        self.optimizer_D = torch.optim.Adam(self.net_D.parameters(),
                                            lr=opts.lr,
                                            betas=(opts.beta1, opts.beta2),
                                            weight_decay=opts.weight_decay)

        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        # Loss function.py
        self.criterion_GAN = LSGANLoss().to(opts.gpu_ids[0])  # common loss
        self.criterion_recon = nn.MSELoss().to(opts.gpu_ids[0])  # pixel-wise loss for G

        self.opts = opts


    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))


    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]  # for both generator and discriminator


    def set_scheduler(self, opts, epoch=-1):
        self.schedulers = [get_scheduler(optimizer, opts, last_epoch=epoch) for optimizer in self.optimizers]


    def set_input(self, data):
        self.vol_NC  = data['vol_NC'].to(self.device).float()
        self.vol_AC  = data['vol_AC'].to(self.device).float()
        self.vol_SC  = data['vol_SC'].to(self.device).float()
        self.vol_SC2 = data['vol_SC2'].to(self.device).float()
        self.vol_SC3 = data['vol_SC3'].to(self.device).float()
        self.vol_GD  = data['vol_GD'].to(self.device).float()
        self.vol_BMI = data['vol_BMI'].to(self.device).float()


    def get_current_losses(self):
        errors_ret = OrderedDict()  # Dictionary
        for name in self.loss_names:
            if isinstance(name, str):  # if name is a str
                errors_ret[name] = float(getattr(self, name))

        return errors_ret


    def set_epoch(self, epoch):
        self.curr_epoch = epoch


    # Generator forward
    def forward(self):
        inp = self.vol_NC

        if self.opts.use_scatter:
            inp = torch.cat([inp, self.vol_SC], 1)  # Be careful of the cat dimension

        if self.opts.use_scatter2:
            inp = torch.cat([inp, self.vol_SC2], 1)  # Be careful of the cat dimension

        if self.opts.use_scatter3:
            inp = torch.cat([inp, self.vol_SC3], 1)  # Be careful of the cat dimension

        if self.opts.use_bmi:
            inp = torch.cat([inp, self.vol_BMI], 1)

        if self.opts.use_gender:
            inp = torch.cat([inp, self.vol_GD], 1)

        inp.requires_grad_(True)  # Input data, [2,ch,32,32,32]
        self.vol_AC_pred = self.net_G(inp)  # Output data, [2,1,32,32,32]


    # Optimize = forward + update
    def optimize(self):
        self.forward()
        ## Traininig Discriminator
        self.optimizer_D.zero_grad()   # reset gradient must before backward to calculate gradients
        
        # fake
        pred_fake = self.net_D(self.vol_AC_pred.squeeze().detach())  # detach() the Generator update
        loss_D_fake = self.criterion_GAN(pred_fake, target_is_real=False)
        # real
        pred_real = self.net_D(self.vol_AC.squeeze())
        loss_D_real = self.criterion_GAN(pred_real, target_is_real=True)
        self.loss_D = (loss_D_fake + loss_D_real) * 0.5
        self.loss_D.backward()
        self.optimizer_D.step()

        ## Traininig Generator
        self.optimizer_G.zero_grad()
        pred_fake = self.net_D(self.vol_AC_pred.squeeze())  # no detach here

        self.loss_G_GAN = self.criterion_GAN(pred_fake, target_is_real=True)  # least-square Loss
        self.loss_G_recon = self.criterion_recon(self.vol_AC_pred, self.vol_AC)  # pixel-wise loss
        self.loss_G = self.opts.GAN_loss_weight*self.loss_G_GAN + self.loss_G_recon  # change the ratio here
        self.loss_G.backward()
        self.optimizer_G.step()


    # Property Display
    @property
    def loss_summary(self):
        return 'loss_G(GAN): {:4f}, loss_G(recon): {:4f}, loss_G: {:4f}, loss_D: {:4f}'.format(self.loss_G_GAN.item(),
                                                                                               self.loss_G_recon.item(),
                                                                                               self.loss_G.item(),
                                                                                               self.loss_D.item())

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()  # learning rate update


    def save(self, filename, epoch, total_iter):    # Save the net/optimizer state data
        torch.save({'net_G': self.net_G.module.state_dict(),
                    'optimizer_G': self.optimizer_G.state_dict(),
                    'net_D': self.net_D.module.state_dict(),
                    'optimizer_D': self.optimizer_D.state_dict(),
                    'epoch': epoch,
                    'total_iter': total_iter},
                   filename)

        print('Saved {}'.format(filename))


    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file)
        if self.opts.wr_L1 > 0:
            self.net_G.module.load_state_dict(checkpoint['net_G'])
            self.net_D.module.load_state_dict(checkpoint['net_D'])
            if train:
                self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
                self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])

        print('Loaded {}'.format(checkpoint_file))
        return checkpoint['epoch'], checkpoint['total_iter']


    def evaluate(self, loader):
        val_bar = tqdm(loader)

        avg_psnr_AC = AverageMeter()
        avg_ssim_AC = AverageMeter()
        avg_mse_AC = AverageMeter()

        pred_AC_images = []
        gt_AC_images = []
        gt_NC_images = []
        gt_SC_images = []
        gt_GD_images = []
        gt_BMI_images = []

        for data in val_bar:
            self.set_input(data)
            self.forward()

            if self.opts.wr_L1 > 0:
                psnr_AC = psnr(self.vol_AC_pred, self.vol_AC)
                mse_AC = mse(self.vol_AC_pred, self.vol_AC)
                ssim_AC = ssim(self.vol_AC_pred[0,0,...].cpu().numpy(), self.vol_AC[0,0,...].cpu().numpy())
                avg_psnr_AC.update(psnr_AC)
                avg_mse_AC.update(mse_AC)
                avg_ssim_AC.update(ssim_AC)

                pred_AC_images.append(self.vol_AC_pred[0].cpu())
                gt_AC_images.append(self.vol_AC[0].cpu())
                gt_NC_images.append(self.vol_NC[0].cpu())
                gt_SC_images.append(self.vol_SC[0].cpu())
                gt_GD_images.append(self.vol_GD[0].cpu())
                gt_BMI_images.append(self.vol_BMI[0].cpu())

            message = 'PSNR AC: {:4f} '.format(avg_psnr_AC.avg)
            message += 'SSIM AC: {:4f} '.format(avg_ssim_AC.avg)
            message += 'MSE AC: {:4f} '.format(avg_mse_AC.avg)
            val_bar.set_description(desc=message)

        self.psnr_AC = avg_psnr_AC.avg
        self.ssim_AC = avg_ssim_AC.avg
        self.mse_AC = avg_mse_AC.avg

        self.results = {}
        if self.opts.wr_L1 > 0:
            self.results['pred_AC'] = torch.stack(pred_AC_images).squeeze().numpy()
            self.results['gt_AC'] = torch.stack(gt_AC_images).squeeze().numpy()
            self.results['gt_NC'] = torch.stack(gt_NC_images).squeeze().numpy()
            self.results['gt_SC'] = torch.stack(gt_SC_images).squeeze().numpy()
            self.results['gt_GD'] = torch.stack(gt_GD_images).squeeze().numpy()
            self.results['gt_BMI'] = torch.stack(gt_BMI_images).squeeze().numpy()
