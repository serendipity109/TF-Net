from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .torch_dct.__init__ import *
from .templates import get_templates
import math

import os
import sys


class DCT(nn.Module):
    def __init__(self, map_size=10):
        super().__init__()
        self.map_size = map_size
        self.init_fbase()
        self.init_fw()
        self.sigmoid = nn.Sigmoid()

    def init_fbase(self):
        self.lfb = torch.zeros([ self.map_size, self.map_size], dtype = float)
        self.mfb = torch.zeros([ self.map_size, self.map_size], dtype = float)
        self.hfb = torch.zeros([ self.map_size, self.map_size], dtype = float)
        self.lfb,  self.mfb,  self.hfb = self.fill(self.lfb,  self.mfb,  self.hfb)

    def fill(self, l, m, h):
        for i in range(l.shape[0]):
            for j in range(l.shape[1]):
                if i + j < math.ceil(l.shape[0]/16):
                    l[i, j] += 1
                elif math.ceil(l.shape[0]/16) <= i + j < math.ceil(l.shape[0]/8):
                    m[i, j] += 1
                else:
                    h[i, j] += 1
        return l.cuda(), m.cuda(), h.cuda()

    def init_fw(self):
        self.lfw = torch.Tensor(self.map_size, self.map_size).cuda() # 左上角
        self.mfw = torch.Tensor(self.map_size, self.map_size).cuda()
        self.hfw = torch.Tensor(self.map_size, self.map_size).cuda()
        nn.init.xavier_normal_(self.lfw)
        nn.init.xavier_normal_(self.mfw)
        nn.init.xavier_normal_(self.hfw)

    def sigma(self, x):
        return (1-torch.exp(-x))/(1+torch.exp(-x))

    def forward(self, x):
        x1 = dct_2d(x)
        y1 = idct_2d(x1 * (self.lfb + self.sigma(self.lfw)))
        y2 = idct_2d(x1 * (self.mfb + self.sigma(self.mfw)))
        y3 = idct_2d(x1 * (self.hfb + self.sigma(self.hfw)))
        return x *self.sigmoid(y1 + y2 + y3).float()

class SeparableConv2d(nn.Module):
    def __init__(self, c_in, c_out, ks, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.c = nn.Conv2d(c_in, c_in, ks, stride, padding,
                           dilation, groups=c_in, bias=bias)
        self.pointwise = nn.Conv2d(c_in, c_out, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.c(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, c_in, c_out, reps, stride=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        self.skip = None
        self.skip_bn = None
        if c_out != c_in or stride != 1:
            self.skip = nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False)
            self.skip_bn = nn.BatchNorm2d(c_out)

        self.relu = nn.ReLU(inplace=True)

        rep = []
        c = c_in
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(c_in, c_out, 3,
                       stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(c_out))
            c = c_out

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(
                c, c, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(c))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(c_in, c_out, 3,
                       stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(c_out))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if stride != 1:
            rep.append(nn.MaxPool2d(3, stride, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            y = self.skip(inp)
            y = self.skip_bn(y)
        else:
            y = inp

        x += y
        return x


class TemplateMap(nn.Module):
    def __init__(self, c_in, templates):
        super(TemplateMap, self).__init__()
        self.c = Block(c_in, 364, 2, 2, start_with_relu=True,
                       grow_first=False)
        self.l = nn.Linear(364, 10)
        self.relu = nn.ReLU(inplace=True)

        self.templates = templates

    def forward(self, x):
        v = self.c(x)
        v = self.relu(v)
        v = F.adaptive_avg_pool2d(v, (1, 1))
        v = v.view(v.size(0), -1)
        v = self.l(v)
        mask = torch.mm(v, self.templates.reshape(10, 361))
        mask = mask.reshape(x.shape[0], 1, 19, 19)

        return mask, v


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self,  num_classes=2, template=False, dct=False):
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block( 728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block( 728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.last_linear = nn.Linear(2048, num_classes)

        self.template = template
        if template:
            self.map = TemplateMap(728, get_templates())

        self.dct = dct
        if dct:
            self.DCT = DCT()

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        if self.template:
            mask, vec = self.map(x)
            x = x * mask
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)  # 64, 1024, 10, 10
        if self.dct:
            x = self.DCT(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)

        if self.template:
            return x, mask, vec
        else:
            return x

    def logits(self, features):
        x = self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        if self.template:
            x, mask, vec = self.features(input)
            x = self.logits(x)
            return x, mask, vec
        else:
            x = self.features(input)
            x = self.logits(x)
            return x

def load_partial_pth(weight_filepath, model):
    checkpoint = torch.load(weight_filepath, map_location='cpu')
    pretrained_au = {}; unused = []
    for k, v in checkpoint['net'].items():
        # if k.replace('xcp.', '') in model.state_dict():
            # pretrained_au[k.replace('xcp.', '')] = v
        if k in model.state_dict():
            pretrained_au[k] = v
        else:
            unused.append(k)
    # print('unused: ', unused)
    md = deepcopy(model.state_dict())
    for key, value in pretrained_au.items():
        md.update(md.fromkeys([key], pretrained_au[key]))
    model.load_state_dict(md)
    return model

############### GRU ###############
class GRU(nn.Module):
    def __init__(self, batch_size):  
        super().__init__()
        self.batch_size = batch_size
        self.gru1 = nn.GRU(2048, 512, 3, batch_first = True, bidirectional = True)
        self.gru2 = nn.GRU(1024, 512, 1, batch_first = True, bidirectional = False)
        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_hidden()
    def init_hidden(self):
        self.h1=torch.Tensor(3*2, self.batch_size, 512).cuda()
        self.h2=torch.Tensor(1, self.batch_size, 512).cuda()
        nn.init.xavier_normal_(self.h1); nn.init.xavier_normal_(self.h2)
    def forward(self, x):
        out, hn = self.gru1(x, self.h1)
        out2, hn = self.gru2(out, self.h2)
        out = self.fc(out2[:, -1, :])   
        return self.sigmoid(out), out2

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def load_xcp(pth = None, template=False, dct=False):
    md = Xception(template = template, dct = dct)
    if pth:
        md = load_partial_pth(pth, md)
    md.last_linear = Identity()
    return md

class model(nn.Module):
    def __init__(self, template = False, dct = False, frame = False):  
        super(model, self).__init__()
        self.xcp = load_xcp(template = template, dct = dct)
        self.gru = GRU(1)
        self.template = template
        self.frame = frame
        if frame:
            self.fmd = f_model()
        else:
            self.fmd = None
    def forward(self, x):
        if self.template:
            feature, mask, _ = self.xcp(x)
        else:
            feature = self.xcp(x)
        x, out = self.gru(feature.view(1, -1, 2048))
        if self.frame:
            x = self.fmd(out).squeeze()
        if self.template:
            return x, mask
        else:
            return x

class f_model(nn.Module): # predict frame level AUC
    def __init__(self):
        super(f_model, self).__init__()
        self.fc= nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.fc(x)
        return self.sigmoid(out)