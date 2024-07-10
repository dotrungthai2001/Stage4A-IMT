import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from torch.autograd import Function


class DnCNN(nn.Module):
    def __init__(self, input_features, middle_features, num_conv_layers):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        if input_features == 5:
            output_features = 4  # Grayscale image
        elif input_features == 15:
            output_features = 12  # RGB image
        else:
            raise Exception('Invalid number of input features')

        layers.append(nn.Conv2d(in_channels=input_features, out_channels=middle_features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_conv_layers-2):
            layers.append(nn.Conv2d(in_channels=middle_features, out_channels=middle_features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(middle_features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=middle_features, out_channels=output_features, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self,x):
        out = self.dncnn(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.orthogonal_(m.weight)
                # nn.init.dirac_(m.weight)
                nn.init.kaiming_normal(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # nn.init.constant_(m.weight, 1)
                nn.init.normal_(mean=0,std=math.sqrt(2./9./64.)).clamp(-0.025,0.025)
                nn.init.constant_(m.bias, 0.0)

class FFDNet(nn.Module):
    def __init__(self,num_input_channels):
        super(FFDNet,self).__init__()
        self.num_input_channels = num_input_channels
        if self.num_input_channels == 1:
            # Grayscale image
            self.num_feature_maps = 64
            self.num_conv_layers = 15
            self.downsampled_channels = 5
            self.output_features = 4
        elif self.num_input_channels == 3:
            # RGB image
            self.num_feature_maps = 96
            self.num_conv_layers = 12
            self.downsampled_channels = 15
            self.output_features = 12
        else:
            raise Exception('Invalid number of input channels')

        self.intermediate_dncnn = DnCNN(input_features=self.downsampled_channels,
                                        middle_features=self.num_feature_maps,
                                        num_conv_layers=self.num_conv_layers)
    def forward(self,x,noise_sigma):
        concat_noise_x = concatenate_input_noise_map(x,noise_sigma)
        h_dncnn = self.intermediate_dncnn(concat_noise_x)
        pred_noise = upsamplefeatures(h_dncnn)
        return x - pred_noise

def concatenate_input_noise_map(input, noise_sigma):
    r"""Implements the first layer of FFDNet. This function returns a
    torch.autograd.Variable composed of the concatenation of the downsampled
    input image and the noise map. Each image of the batch of size CxHxW gets
    converted to an array of size 4*CxH/2xW/2. Each of the pixels of the
    non-overlapped 2x2 patches of the input image are placed in the new array
    along the first dimension.

    Args:
    input: batch containing CxHxW images
    noise_sigma: the value of the pixels of the CxH/2xW/2 noise map
    """
    # noise_sigma is a list of length batch_size
    N, C, H, W = input.shape
    sca = 2
    sca2 = sca*sca
    Cout = sca2*C
    Hout = H//sca
    Wout = W//sca
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

    downsampledfeatures = torch.zeros((N,Cout,Hout,Wout),dtype=input.dtype,device=input.device)

    # Build the CxH/2xW/2 noise map
    #noise_map = noise_sigma.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, C, Hout, Wout)
    noise_map = noise_sigma.view(N, 1, 1, 1).repeat(1, C, Hout, Wout)

    # Populate output
    for idx in range(sca2):
        downsampledfeatures[:, idx:Cout:sca2, :, :] = input[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca]

    # concatenate de-interleaved mosaic with noise map
    return torch.cat((noise_map, downsampledfeatures), 1)

class UpSampleFeaturesFunction(Function):
    r"""Extends PyTorch's modules by implementing a torch.autograd.Function.
    This class implements the forward and backward methods of the last layer
    of FFDNet. It basically performs the inverse of
    concatenate_input_noise_map(): it converts each of the images of a
    batch of size CxH/2xW/2 to images of size C/4xHxW
    """
    @staticmethod
    def forward(ctx, input):
        N, Cin, Hin, Win = input.shape
        sca = 2
        sca2 = sca*sca
        Cout = Cin//sca2
        Hout = Hin*sca
        Wout = Win*sca
        idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

        assert (Cin%sca2 == 0), 'Invalid input dimensions: number of channels should be divisible by 4'

        result = torch.zeros((N, Cout, Hout, Wout),dtype=input.dtype,device=input.device)
        for idx in range(sca2):
            result[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca] = input[:, idx:Cin:sca2, :, :]

        return result

    @staticmethod
    def backward(ctx, grad_output):
        N, Cg_out, Hg_out, Wg_out = grad_output.size()
        dtype = grad_output.data.type()
        sca = 2
        sca2 = sca*sca
        Cg_in = sca2*Cg_out
        Hg_in = Hg_out//sca
        Wg_in = Wg_out//sca
        idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

        # Build output
        grad_input = torch.zeros((N, Cg_in, Hg_in, Wg_in),dtype=grad_output.data.dtype,device=grad_output.data.device)
        # Populate output
        for idx in range(sca2):
            grad_input[:, idx:Cg_in:sca2, :, :] = grad_output.data[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca]

        return grad_input

# Alias functions
upsamplefeatures = UpSampleFeaturesFunction.apply
