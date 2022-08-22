from hanmodel import common
import torch
import torch.nn as nn
import pdb

def make_model(args, parent=False):
    return HAN(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class LAM_Module(nn.Module):
    """ Layer attention module"""
    def __init__(self, in_dim):
        super(LAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out + x
        out = out.view(m_batchsize, -1, height, width)
        return out
class Style(nn.Module):
    def __init__(self, dim):
        super().__init__()
        fc = []
        for i in range(8):
            fc.append(nn.Linear(512,512))
            fc.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*fc)
        self.affine = nn.Linear(512, dim)
    def forward(self,z):
        w = self.mapping(z)
        style = self.affine(w)
        
        return style.view((style.shape[0], style.shape[1],1,1))


class CSAM_Module(nn.Module):
    """ Channel-Spatial attention module"""
    def __init__(self, in_dim, neptune):
        super(CSAM_Module, self).__init__()
        self.chanel_in = in_dim

        # self.style = Style(in_dim)

        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        
            
        self.neptune = neptune
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))
        
        # proj_query = x.view(m_batchsize, N, -1)
        # proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        # energy = torch.bmm(proj_query, proj_key)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        # attention = self.softmax(energy_new)
        # proj_value = x.view(m_batchsize, N, -1)

        # out = torch.bmm(attention, proj_value)
        # out = out.view(m_batchsize, N, C, height, width)

        # z = x.new_empty(x.shape[0],512).normal_()
        # style = self.style(z)

        # out = self.gamma*out
        

        out = out
        out = out.view(m_batchsize, -1, height, width)#*style
        x = x * out# + x
        return x

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
class NoiseInjection(nn.Module):
    def __init__(self, channel, std):
        super().__init__()
        self.channel = channel
        self.std = std
        self.w = torch.nn.Parameter(torch.zeros(1,self.channel, 1,1))
        
        
        
        
    def forward(self, image):
        batch, channel, height, width = image.shape
        assert channel==self.channel
        noise = image.new_empty(batch, 1, height, width).normal_()*self.std
        noise = noise.expand((-1,channel,-1,-1))
        
        
     
        return image + self.w*noise


## Holistic Attention Network (HAN)
class HAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(HAN, self).__init__()
        
        n_resgroups = args.n_resgroups
        n_resblocks = 20
        n_feats = 128
        kernel_size = 3
        reduction = 16
        scale = args.upscale_factor
        act = nn.ReLU(True)
        self.gpu=args.gpu
        self.args = args
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(3, n_feats, kernel_size, stride=2), conv(n_feats, n_feats, kernel_size, stride=2)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module


        # modules_tail = [
        #     common.Upsampler(conv, scale, n_feats, act=False),
        #     conv(n_feats, args.n_colors, kernel_size)]

        modules_tail = [conv(n_feats, 3, kernel_size)] # Upsample block removed

        # self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.noises = nn.ModuleList([NoiseInjection(n_feats, args.noise_std) for x in range(args.n_resgroups + 2)])
        self.csa = CSAM_Module(n_feats, False)
        self.la = LAM_Module(n_feats)
        self.last_conv = nn.Conv2d(n_feats*11, n_feats, 3, 1, 1)
        self.last = nn.Conv2d(n_feats*2, n_feats, 3, 1, 1)
        self.tail = nn.Sequential(*modules_tail)
        

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)
        res = x
        #pdb.set_trace()
        idx = 0
        for name, midlayer in self.body._modules.items():
            res = midlayer(res)
            if self.args.use_noise:
                res = self.noises[idx](res)
                idx+=1
            #print(name)
            if name=='0':
                res1 = res.unsqueeze(1)
            else:
                res1 = torch.cat([res.unsqueeze(1),res1],1)
        #res = self.body(x)
        out1 = res
        #res3 = res.unsqueeze(1)
        #res = torch.cat([res1,res3],1)
        res = self.la(res1)
        out2 = self.last_conv(res)

       
        if self.args.use_noise: out1 = self.noises[idx](out1)
        out1 = self.csa(out1)
       
        out = torch.cat([out1, out2], 1)
        res = self.last(out)
        
        res += x
        #res = self.csa(res)

        x = self.tail(res)
        # x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))