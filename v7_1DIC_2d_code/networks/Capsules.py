from __future__ import print_function
import torch.nn.parallel
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import pdb

USE_CUDA = torch.cuda.is_available()


class convolutionalCapsule(nn.Module):
    def __init__(self, in_capsules, out_capsules, in_channels, out_channels, stride=1, padding=2,
                 kernel=5, num_routes=3, nonlinearity='sqaush', dynamic_routing='local'):
        super(convolutionalCapsule, self).__init__()
        self.num_routes = num_routes
        self.in_channels = in_channels
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.out_channels = out_channels
        self.nonlinearity = nonlinearity
        self.bn = nn.BatchNorm3d(in_capsules*out_capsules*out_channels)
        self.conv3d = nn.Conv3d(kernel_size=(kernel, kernel, kernel), stride=stride, padding=padding,
                                in_channels=in_channels, out_channels=out_channels*out_capsules)
        self.dynamic_routing = dynamic_routing

    def forward(self, x):
        batch_size = x.size(0)
        in_width, in_height, in_depth = x.size(3), x.size(4), x.size(5)
        x = x.view(batch_size*self.in_capsules, self.in_channels, in_width, in_height, in_depth)
        u_hat = self.conv3d(x)

        out_width, out_height, out_depth = u_hat.size(2), u_hat.size(3), u_hat.size(4)

        # batch norm layer
        u_hat = u_hat.permute(0, 2, 3, 4, 1).contiguous()
        u_hat = u_hat.view(batch_size, self.in_capsules, out_width, out_height, out_depth, self.out_capsules * self.out_channels)
        u_hat = u_hat.view(batch_size, self.in_capsules, out_width, out_height, out_depth, self.out_capsules, self.out_channels)

        b_ij = Variable(torch.zeros(1, self.in_capsules, out_width, out_height, out_depth, self.out_capsules))
        b_ij = b_ij.cuda()

        for iteration in range(self.num_routes):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(6)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            if (self.nonlinearity == 'relu') and (iteration == self.num_routes - 1):
                v_j = F.relu(s_j)
            elif (self.nonlinearity == 'leakyRelu') and (iteration == self.num_routes - 1):
                v_j = F.leaky_relu(s_j)
            else:
                v_j = self.squash(s_j)

            v_j = v_j.squeeze(1)

            if iteration < self.num_routes - 1:
                temp = u_hat.permute(0, 2, 3, 4, 5, 1, 6)
                temp2 = v_j.unsqueeze(6)
                a_ij = torch.matmul(temp, temp2).squeeze(6)    # dot product here
                a_ij = a_ij.permute(0, 5, 1, 2, 3, 4)
                b_ij = b_ij + a_ij.mean(dim=0)

        v_j = v_j.permute(0, 4, 5, 1, 2, 3).contiguous()

        return v_j

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class deconvolutionalCapsule(nn.Module):
    def __init__(self, in_capsules, out_capsules, in_channels, out_channels, stride=2, padding=2, kernel=4,
                 num_routes=3, nonlinearity='sqaush', dynamic_routing='local'):
        super(deconvolutionalCapsule, self).__init__()
        self.nonlinearity = nonlinearity
        self.num_routes = num_routes
        self.in_channels = in_channels
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.out_channels = out_channels
        self.bn = nn.BatchNorm3d(in_capsules*out_capsules*out_channels)
        self.deconv3d = nn.ConvTranspose3d(kernel_size=(kernel, kernel, kernel), stride=stride, padding=padding,
                                           in_channels=in_channels, out_channels=out_channels * out_capsules)
        self.dynamic_routing = dynamic_routing

    def forward(self, x):
        batch_size = x.size(0)
        in_width, in_height, in_depth = x.size(3), x.size(4), x.size(5)
        x = x.view(batch_size*self.in_capsules, self.in_channels, in_width, in_height, in_depth)
        u_hat = self.deconv3d(x)
        out_width, out_height, out_depth = u_hat.size(2), u_hat.size(3), u_hat.size(4)

        # batch norm layer
        u_hat = u_hat.permute(0, 2, 3, 4, 1).contiguous()
        u_hat = u_hat.view(batch_size, self.in_capsules, out_width, out_height, out_depth, self.out_capsules*self.out_channels)
        u_hat = u_hat.view(batch_size, self.in_capsules, out_width, out_height, out_depth, self.out_capsules, self.out_channels)

        b_ij = Variable(torch.zeros(1, self.in_capsules, out_width, out_height, out_depth, self.out_capsules))
        b_ij = b_ij.cuda()

        for iteration in range(self.num_routes):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(6)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            if (self.nonlinearity == 'relu') and (iteration == self.num_routes - 1):
                v_j = F.relu(s_j)
            elif (self.nonlinearity == 'leakyRelu') and (iteration == self.num_routes - 1):
                v_j = F.leaky_relu(s_j)
            else:
                v_j = self.squash(s_j)

            v_j = v_j.squeeze(1)

            if iteration < self.num_routes - 1:
                temp = u_hat.permute(0, 2, 3, 4, 5, 1, 6)
                temp2 = v_j.unsqueeze(6)
                a_ij = torch.matmul(temp, temp2).squeeze(6) # dot product here
                a_ij = a_ij.permute(0, 5, 1, 2, 3, 4)
                b_ij = b_ij + a_ij.mean(dim=0)

        v_j = v_j.permute(0, 4, 5, 1, 2, 3).contiguous()

        return v_j

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

