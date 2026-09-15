import os
from abc import ABC

import csv
import numpy as np
from collections import OrderedDict
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from scipy.special import entr
import pdb
from skimage.metrics import structural_similarity as ssim
from networks import get_generator, get_discriminator
from networks.networks import gaussian_weights_init
from models.utils import AverageMeter, get_scheduler, psnr, mse, nmse, nmae
from utils.data_patch_util import *
import tifffile
from torchmetrics import StructuralSimilarityIndexMeasure
from utils import prepare_sub_folder
# install jupyter, tqdm, torchmetrics, h5py, opencv, scipy, scikit-image, nibabel

class SSIMLoss(nn.Module):
       def __init__(self, data_range=1.0):
           super(SSIMLoss, self).__init__()
           self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range)

       def forward(self, preds, target):
           return 1 - self.ssim(preds, target)

class CombinedLoss(nn.Module):
    def __init__(self, data_range=1.0, weight=0.5):
        """
        Combined loss function that combines SSIM loss and masked MSE loss.

        Args:
            data_range (float): Dynamic range of the data for SSIM computation.
            weight (float): Weight to control the contribution of the masked MSE loss.
            threshold (float): Threshold value to create a mask for preds and target.
        """
        super(CombinedLoss, self).__init__()
        self.ssim_loss = SSIMLoss(data_range=data_range)
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.weight = weight
        #self.threshold = threshold

    def forward(self, preds, target):
        # Apply threshold mask
        threshold = torch.quantile(target, 0.00)
        mask_target = target >= threshold

        # Apply the mask to predictions and target
        mito_target = target * mask_target

        # Calculate the masked MSE loss
        mse_loss = self.mse_loss(preds, mito_target)

        # Calculate the SSIM loss
        ssim_loss = self.ssim_loss(preds, mito_target)

        # Combine the SSIM loss and the masked MSE loss
        combined_loss = (1.0-self.weight) * ssim_loss + self.weight * mse_loss

        return combined_loss


class CNNModel(nn.Module):
    def __init__(self, opts):
        super(CNNModel, self).__init__()

        self.loss_names = []  # list
        self.networks = []  # list
        self.optimizers = []  # list
        self.lr = opts.lr
        # self.lr_decay = opts.lr_decay

        # set default loss flags ??
        loss_flags = ["w_img_L1"]
        for flag in loss_flags:
            if not hasattr(opts, flag): setattr(opts, flag, 0)

        self.is_train = True if hasattr(opts, 'lr') else False

        self.net_G = get_generator(opts.net_G, opts)
        self.networks.append(self.net_G)

        if self.is_train:
            self.loss_names += ['loss_G_L1']
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(),
                                                lr=self.lr,  # initilize the learning rate for the optimizer
                                                betas=(opts.beta1, opts.beta2),
                                                weight_decay=opts.weight_decay)

            self.optimizers.append(self.optimizer_G)

        self.criterion = CombinedLoss(data_range=1.0, weight=0.00).cuda()
        self.opts = opts

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))  # Choose GPU for CUDA computing; For input setting

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    # LR decay can be realized here
    def set_scheduler(self, opts, epoch=-1):
        self.schedulers = [get_scheduler(optimizer, opts, last_epoch=epoch) for optimizer in self.optimizers]


    def set_input(self, data):
        self.vol_dic1 = data['vol_dic1'].to(self.device).float()
        self.vol_dic3 = data['vol_dic3'].to(self.device).float()
        self.vol_dic4 = data['vol_dic4'].to(self.device).float()
        self.vol_dic6 = data['vol_dic6'].to(self.device).float()
        self.vol_fluo = data['vol_fluo'].to(self.device).float()
        self.opts_drop = data['opts_drop'][0].numpy()  # Training: True; Testing: False

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name))  # get self.loss_G_L1
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def forward(self):
        inp_ = torch.cat((self.vol_dic1, self.vol_dic3, self.vol_dic4, self.vol_dic6,), 1)
        inp_.requires_grad_(True)
        self.vol_fluo_pred = self.net_G(inp_, self.opts_drop)

    def update_G(self):
        self.optimizer_G.zero_grad()  # Zero gradient
        
        loss_G_L1 = self.criterion(self.vol_fluo_pred, self.vol_fluo)
        self.loss_G_L1 = loss_G_L1.item()  # <class 'dict_items'>, for discription

        total_loss = loss_G_L1
        total_loss.backward()
        self.optimizer_G.step()


    def optimize(self):  # Use the last 2 functions
        # self.loss_G_L1 = 0

        self.forward()
        self.update_G()

    @property  # Only for this function.py
    def loss_summary(self):
        message = ''
        if self.opts.wr_L1 > 0:
            message += 'G_L1: {:.4e} '.format(self.loss_G_L1)

        return message

    # learning rate decay
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()  # learning rate update
        self.lr = self.optimizers[0].param_groups[0]['lr']
        # print('learning rate = {:7f}'.format(lr))

        # self.lr = self.lr_decay * self.lr
        # for param_group in self.optimizer_G.param_groups:
        #     param_group['lr'] = self.lr  # Update the lr for the optimizer

    def save(self, filename, epoch, total_iter):  # Save the net/optimizer state data
        state = {}  # dict
        if self.opts.wr_L1 > 0:
            state['net_G'] = self.net_G.module.state_dict()
            state['opt_G'] = self.optimizer_G.state_dict()

        state['epoch'] = epoch
        state['total_iter'] = total_iter

        torch.save(state, filename)

        print('Saved {}'.format(filename))


    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file, map_location=self.device)

        if self.opts.wr_L1 > 0:
            self.net_G.module.load_state_dict(checkpoint['net_G'])
            if train:
                self.optimizer_G.load_state_dict(checkpoint['opt_G'])

        print('Loaded {}'.format(checkpoint_file))

        return checkpoint['epoch'], checkpoint['total_iter']


    # -------------- Evaluation, Calculate PSNR ---------------
    def soft_dice_score(self, preds, target, epsilon=1e-6):
        numerator = 2.0 * np.sum(preds*target)
        denominator = np.sum(preds**2) + np.sum(target**2)

        return (numerator + epsilon) / (denominator + epsilon)
    
    def evaluate(self, loader):
        val_bar = tqdm(loader)

        # For calculating metrics
        avg_psnr_ = AverageMeter()
        avg_ssim_ = AverageMeter()
        avg_pcc_ = AverageMeter()   # pearson correlation
        avg_sds_ = AverageMeter()
        avg_mse_ = AverageMeter()
        avg_nmse_ = AverageMeter()
        avg_nmae_ = AverageMeter()


        for data in val_bar:
            self.set_input(data)  # [batch_szie=1, 1, 64, 64, 64]
            self.forward()

            #tar = self.vol_fluo[0,0,...].cpu().numpy()
            #pre = self.vol_fluo_pred[0,0,...].cpu().numpy()
            #tar_mask = tar >= 0.25
            #pre_mask = pre >= 0.25
            #tar = tar * tar_mask
            #pre = pre * pre_mask

            threshold = torch.quantile(self.vol_fluo, 0.00)
            #mask_preds = self.vol_fluo_pred >= threshold
            mask_target = self.vol_fluo >= threshold
            #preds = self.vol_fluo_pred * mask_preds
            target = self.vol_fluo * mask_target

            # Calculate the metrics; z_range can be used to calculate the mean
            psnr_ = psnr(self.vol_fluo_pred, target)
            mse_  =  mse(self.vol_fluo_pred, target)
            ssim_ = ssim(self.vol_fluo_pred[0,0,...].cpu().numpy(), target[0,0,...].cpu().numpy(), data_range=1.0)
            pcc_ = np.corrcoef(self.vol_fluo_pred[0,0,...].cpu().numpy().flatten(), target[0,0,...].cpu().numpy().flatten())[0,1]
            sds_ = self.soft_dice_score(self.vol_fluo_pred[0,0,...].cpu().numpy(), target[0,0,...].cpu().numpy())
            nmse_ = nmse(self.vol_fluo_pred, target)
            nmae_ = nmae(self.vol_fluo_pred, target)

            if self.opts.resume is not None:
                self.save_metric_pcc = pcc_
                self.save_mrtric_sds = sds_
                _, directory = prepare_sub_folder(os.path.join(self.opts.output_path, 'outputs', self.opts.experiment_name))
                with open(os.path.join(directory, 'metrics.csv'), 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.resume(self.opts.resume)[0], self.save_metric_pcc, self.save_mrtric_sds])

            avg_psnr_.update(psnr_)
            avg_mse_.update(mse_)
            avg_ssim_.update(ssim_)
            avg_pcc_.update(pcc_)
            avg_sds_.update(sds_)
            avg_nmse_.update(nmse_)
            avg_nmae_.update(nmae_)

            # Descrip show NMSE, NMAE, SSIM here
            message = 'NMSE: {:4f} '.format(avg_nmse_.avg)
            message += 'SSIM: {:4f} '.format(avg_ssim_.avg)
            message += 'PCC: {:4f} '.format(avg_pcc_.avg)
            message += 'SDS: {:4f} '.format(avg_sds_.avg)
            message += 'PSNR: {:4f} '.format(avg_psnr_.avg)
            message += 'NMAE: {:4f} '.format(avg_nmae_.avg)
            val_bar.set_description(desc=message)

        # Calculate the average metrics
        self.nmse_ = avg_nmse_.avg
        self.nmae_ = avg_nmae_.avg
        self.ssim_ = avg_ssim_.avg  # Saved as .csv files
        self.pcc_ = avg_pcc_.avg
        self.sds_ = avg_sds_.avg
        self.psnr_ = avg_psnr_.avg
        self.mse_  = avg_mse_.avg


    # --------------- Save the images ------------------------------
    def save_images(self, loader, folder):
        val_bar = tqdm(loader)
        val_bar.set_description(desc='Saving images ...')

        # Load data for each batch
        index = 0
        for data in val_bar:
            index += 1
            self.set_input(data)  # [batch_size=1, 1, 412, 412]
            self.forward()

            # --------------- Mkdir folder -------------------
            if not os.path.exists(os.path.join(folder, 'vol_dic1')):
                os.mkdir(os.path.join(folder, 'vol_dic1'))
            # if not os.path.exists(os.path.join(folder, 'vol_dic3')):
            #     os.mkdir(os.path.join(folder, 'vol_dic3'))
            # if not os.path.exists(os.path.join(folder, 'vol_dic4')):
            #     os.mkdir(os.path.join(folder, 'vol_dic4'))
            # if not os.path.exists(os.path.join(folder, 'vol_dic6')):
            #     os.mkdir(os.path.join(folder, 'vol_dic6'))

            if not os.path.exists(os.path.join(folder, 'vol_fluo')):
                os.mkdir(os.path.join(folder, 'vol_fluo'))

            if not os.path.exists(os.path.join(folder, 'vol_fluo_pred')):
                os.mkdir(os.path.join(folder, 'vol_fluo_pred'))

            # # save nifti image
            # save_nii(np.rot90(np.rot90(self.vol_oidic.squeeze().cpu().numpy())),      os.path.join(folder, 'vol_oidic', 'vol_oidic_' + str(index) + '.nii'))
            # save_nii(np.rot90(np.rot90(self.vol_fluo.squeeze().cpu().numpy())),       os.path.join(folder, 'vol_fluo', 'vol_fluo_' + str(index) + '.nii'))
            # save_nii(np.rot90(np.rot90(self.vol_fluo_pred.squeeze().cpu().numpy())),  os.path.join(folder, 'vol_fluo_pred', 'vol_fluo_pred_' + str(index) + '.nii'))

            # Save tif images
            tifffile.imwrite(os.path.join(folder, 'vol_dic1', 'vol_dic1_' + str(index) + '.tif'),            self.vol_dic1.squeeze().cpu().numpy().transpose(1, 0))
            #tifffile.imwrite(os.path.join(folder, 'vol_dic3', 'vol_dic3_' + str(index) + '.tif'),            self.vol_dic3.squeeze().cpu().numpy().transpose(1, 0))
            #tifffile.imwrite(os.path.join(folder, 'vol_dic4', 'vol_dic4_' + str(index) + '.tif'),            self.vol_dic4.squeeze().cpu().numpy().transpose(1, 0))
            #tifffile.imwrite(os.path.join(folder, 'vol_dic6', 'vol_dic6_' + str(index) + '.tif'),            self.vol_dic6.squeeze().cpu().numpy().transpose(1, 0))
            tifffile.imwrite(os.path.join(folder, 'vol_fluo', 'vol_fluo_' + str(index) + '.tif'),            self.vol_fluo.squeeze().cpu().numpy().transpose(1, 0))
            tifffile.imwrite(os.path.join(folder, 'vol_fluo_pred', 'vol_fluo_pred_' + str(index) + '.tif'),  self.vol_fluo_pred.squeeze().cpu().numpy().transpose(1, 0))












