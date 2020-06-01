# coding: utf-8
import torch.nn as nn
import torch
from .mynet3 import F_mynet3
from .BAM import BAM
from .PAM2 import PAM as PAM



def define_F(in_c, f_c, type='unet'):
    if type == 'mynet3':
        print("using mynet3 backbone")
        return F_mynet3(backbone='resnet18', in_c=in_c,f_c=f_c, output_stride=32)
    else:
        NotImplementedError('no such F type!')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



class CDSA(nn.Module):
    """self attention module for change detection

    """
    def __init__(self, in_c, ds=1, mode='BAM'):
        super(CDSA, self).__init__()
        self.in_C = in_c
        self.ds = ds
        print('ds: ',self.ds)
        self.mode = mode
        if self.mode == 'BAM':
            self.Self_Att = BAM(self.in_C, ds=self.ds)
        elif self.mode == 'PAM':
            self.Self_Att = PAM(in_channels=self.in_C, out_channels=self.in_C, sizes=[1,2,4,8],ds=self.ds)
        self.apply(weights_init)

    def forward(self, x1, x2):
        height = x1.shape[3]
        x = torch.cat((x1, x2), 3)
        x = self.Self_Att(x)
        return x[:,:,:,0:height], x[:,:,:,height:]




