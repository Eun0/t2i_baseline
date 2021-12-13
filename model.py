import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

from miscc.config import cfg 


class NetG(nn.Module):
    def __init__(self, ngf=64, nz=100, nef=256):
        super(NetG, self).__init__()
        self.ngf = ngf

        # layer1输入的是一个100x1x1的随机噪声, 输出尺寸(ngf*8)x4x4
        self.fc = nn.Linear(nz, ngf*8*4*4)
        self.block0 = G_Block(ngf * 8, ngf * 8)#4x4
        self.block1 = G_Block(ngf * 8, ngf * 8)#4x4
        self.block2 = G_Block(ngf * 8, ngf * 8)#8x8
        self.block3 = G_Block(ngf * 8, ngf * 8)#16x16
        self.block4 = G_Block(ngf * 8, ngf * 4)#32x32
        self.block5 = G_Block(ngf * 4, ngf * 2)#64x64
        self.block6 = G_Block(ngf * 2, ngf * 1)#128x128

        self.conv_img = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ngf, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x, c):

        out = self.fc(x)
        out = out.view(x.size(0), 8*self.ngf, 4, 4)
        out = self.block0(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block1(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block2(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block3(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block4(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block5(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block6(out,c)

        out = self.conv_img(out)

        return out



# class NetG(nn.Module):
#     def __init__(self, ngf=64, nz=100,nef=256):
#         super(NetG, self).__init__()
        
#         self.ngf = ngf
#         self.nz = nz 
#         self.nef = nef 

#         self.fc = nn.Linear(nz,ngf*8*4*4)
#         blocks= []
#         assert cfg.TREE.BASE_SIZE in [64,128,256]
#         if cfg.TREE.BASE_SIZE == 256:
#             in_dims = [ngf*8, ngf*8, ngf*8, ngf*8, ngf*8, ngf*4, ngf*2]
#             out_dims = [ngf*8, ngf*8, ngf*8, ngf*8, ngf*4, ngf*2, ngf*1]
#         elif cfg.TREE.BASE_SIZE == 128:
#             in_dims = [ngf*8, ngf*8, ngf*8, ngf*8, ngf*4, ngf*2]
#             out_dims = [ngf*8, ngf*8, ngf*8, ngf*4, ngf*2, ngf*1]
#         else:
#             in_dims = [ngf*8, ngf*8, ngf*8, ngf*4, ngf*2]
#             out_dims = [ngf*8, ngf*8, ngf*4, ngf*2, ngf*1]

#         for in_dim,out_dim in zip(in_dims,out_dims):
#             blocks.append(G_Block(in_dim,out_dim,nef))
#         self.blocks = nn.ModuleList(blocks)

#         self.conv_img = nn.Sequential(
#             nn.LeakyReLU(0.2,inplace=True),
#             nn.Conv2d(ngf, 3, 3, 1, 1),
#             nn.Tanh(),
#         )

#     def forward(self, x, c):

#         out = self.fc(x)
#         out = out.view(x.size(0), 8*self.ngf, 4, 4)

#         for i in range(len(self.blocks)):
#             out = self.blocks[i](out,c)
#             if i != len(self.blocks)-1:
#                 out = F.interpolate(out,scale_factor=2)

#         out = self.conv_img(out)

#         return out


class NetD(nn.Module):
    def __init__(self, ndf, nef=256):
        super(NetD, self).__init__()

        self.conv_img = nn.Conv2d(3, ndf, 3, 1, 1)#128
        blocks= []

        assert cfg.TREE.BASE_SIZE in [64,128,256]
        if cfg.TREE.BASE_SIZE == 256:
            in_dims = [ndf*1, ndf*2, ndf*4, ndf*8, ndf*16, ndf*16]
            out_dims = [ndf*2, ndf*4, ndf*8, ndf*16, ndf*16, ndf*16]
        elif cfg.TREE.BASE_SIZE == 128:
            in_dims = [ndf*1, ndf*2, ndf*4, ndf*8, ndf*16]
            out_dims = [ndf*2, ndf*4, ndf*8, ndf*16, ndf*16]
        else:
            in_dims = [ndf*1, ndf*2, ndf*4, ndf*8]
            out_dims = [ndf*2, ndf*4, ndf*8, ndf*16]

        for in_dim,out_dim in zip(in_dims,out_dims):
            blocks.append(resD(in_dim,out_dim))

        self.blocks = nn.ModuleList(blocks)
        self.REC = D_REC(ndf)
        self.COND_DNET = D_GET_LOGITS(ndf,nef)

    def forward(self,x,sent_embs,mode='cond'):

        out = self.conv_img(x)

        for i in range(len(self.blocks)):
            out = self.blocks[i](out)
            if i==3:
                out8 = out
        if mode=='mis':
            out = self.COND_DNET(out[:(x.size(0)-1)],sent_embs[1:x.size(0)])
        else:
            out = self.COND_DNET(out,sent_embs)
        rec = self.REC(out8)
        return out,rec,out8

        '''if mode=='rec':
            return out,rec
        elif mode=='fm':
            return out,out8
        else:
            return out
        '''
class D_REC(nn.Module):
    def __init__(self,ndf):
        super(D_REC,self).__init__()
        self.df_dim = ndf
        self.block0 = upBlock(ndf*16,ndf*8)
        self.block1 = upBlock(ndf*8,ndf*4)
        self.block2 = upBlock(ndf*4,ndf*2)
        self.block3 = upBlock(ndf*2,ndf*1)
        self.conv = nn.Conv2d(ndf,3,3,1,1)

    def forward(self,out8):
        out8 = self.block0(out8)
        out8 = self.block1(out8)
        out8 = self.block2(out8)
        out8 = self.block3(out8)
        out8 = self.conv(out8)
        return out8 



class G_Block(nn.Module):

    def __init__(self, in_ch, out_ch,nef=256):
        super(G_Block, self).__init__()

        self.learnable_sc = in_ch != out_ch 
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.affine0 = affine(in_ch,nef)
        self.affine1 = affine(in_ch,nef)
        self.affine2 = affine(out_ch,nef)
        self.affine3 = affine(out_ch,nef)
        self.gamma = nn.Parameter(torch.zeros(1))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)

    def forward(self, x, y=None):
        return self.shortcut(x) + self.gamma * self.residual(x, y)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, y=None):
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.c1(h)
        
        h = self.affine2(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.affine3(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        return self.c2(h)



class affine(nn.Module):

    def __init__(self, num_features,nef):
        super(affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(nef, nef)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(nef, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(nef, nef)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(nef, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):

        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)        

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf,nef):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef

        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf * 16+self.ef_dim, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )
    def forward(self, out, y):
        y = y.view(-1, self.ef_dim, 1, 1)
        y = y.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((out, y), 1)
        out = self.joint_conv(h_c_code)
        return out


def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_planes, out_planes * 2,3,1,1),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block




class resD(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_s = nn.Conv2d(fin,fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, c=None):
        return self.shortcut(x)+self.gamma*self.residual(x)

    def shortcut(self, x):
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        return self.conv_r(x)