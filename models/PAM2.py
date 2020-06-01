import torch
import torch.nn.functional as F
from torch import nn


class _PAMBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input/Output:
        N * C  *  H  *  (2*W)
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to partition the input feature maps
        ds                : downsampling scale
    '''
    def __init__(self, in_channels, key_channels, value_channels, scale=1, ds=1):
        super(_PAMBlock, self).__init__()
        self.scale = scale
        self.ds = ds
        self.pool = nn.AvgPool2d(self.ds)
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels)
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels)
        )
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
            kernel_size=1, stride=1, padding=0)



    def forward(self, input):
        x = input
        if self.ds != 1:
            x = self.pool(input)
        # input shape: b,c,h,2w
        batch_size, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)//2


        local_y = []
        local_x = []
        step_h, step_w = h//self.scale, w//self.scale
        for i in range(0, self.scale):
            for j in range(0, self.scale):
                start_x, start_y = i*step_h, j*step_w
                end_x, end_y = min(start_x+step_h, h), min(start_y+step_w, w)
                if i == (self.scale-1):
                    end_x = h
                if j == (self.scale-1):
                    end_y = w
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]

        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)

        value = torch.stack([value[:, :, :, :w], value[:,:,:,w:]], 4)  # B*N*H*W*2
        query = torch.stack([query[:, :, :, :w], query[:,:,:,w:]], 4)  # B*N*H*W*2
        key = torch.stack([key[:, :, :, :w], key[:,:,:,w:]], 4)  # B*N*H*W*2


        local_block_cnt = 2*self.scale*self.scale

        #  self-attention func
        def func(value_local, query_local, key_local):
            batch_size_new = value_local.size(0)
            h_local, w_local = value_local.size(2), value_local.size(3)
            value_local = value_local.contiguous().view(batch_size_new, self.value_channels, -1)

            query_local = query_local.contiguous().view(batch_size_new, self.key_channels, -1)
            query_local = query_local.permute(0, 2, 1)
            key_local = key_local.contiguous().view(batch_size_new, self.key_channels, -1)

            sim_map = torch.bmm(query_local, key_local)  # batch matrix multiplication
            sim_map = (self.key_channels**-.5) * sim_map
            sim_map = F.softmax(sim_map, dim=-1)

            context_local = torch.bmm(value_local, sim_map.permute(0,2,1))
            # context_local = context_local.permute(0, 2, 1).contiguous()
            context_local = context_local.view(batch_size_new, self.value_channels, h_local, w_local, 2)
            return context_local

        #  Parallel Computing to speed up
        #  reshape value_local, q, k
        v_list = [value[:,:,local_x[i]:local_x[i+1],local_y[i]:local_y[i+1]] for i in range(0, local_block_cnt, 2)]
        v_locals = torch.cat(v_list,dim=0)
        q_list = [query[:,:,local_x[i]:local_x[i+1],local_y[i]:local_y[i+1]] for i in range(0, local_block_cnt, 2)]
        q_locals = torch.cat(q_list,dim=0)
        k_list = [key[:,:,local_x[i]:local_x[i+1],local_y[i]:local_y[i+1]] for i in range(0, local_block_cnt, 2)]
        k_locals = torch.cat(k_list,dim=0)
        # print(v_locals.shape)
        context_locals = func(v_locals,q_locals,k_locals)

        context_list = []
        for i in range(0, self.scale):
            row_tmp = []
            for j in range(0, self.scale):
                left = batch_size*(j+i*self.scale)
                right = batch_size*(j+i*self.scale) + batch_size
                tmp = context_locals[left:right]
                row_tmp.append(tmp)
            context_list.append(torch.cat(row_tmp, 3))

        context = torch.cat(context_list, 2)
        context = torch.cat([context[:,:,:,:,0],context[:,:,:,:,1]],3)


        if self.ds !=1:
            context = F.interpolate(context, [h*self.ds, 2*w*self.ds])

        return context


class PAMBlock(_PAMBlock):
    def __init__(self, in_channels, key_channels=None, value_channels=None, scale=1, ds=1):
        if key_channels == None:
            key_channels = in_channels//8
        if value_channels == None:
            value_channels = in_channels
        super(PAMBlock, self).__init__(in_channels,key_channels,value_channels,scale,ds)


class PAM(nn.Module):
    """
        PAM module
    """

    def __init__(self, in_channels, out_channels, sizes=([1]), ds=1):
        super(PAM, self).__init__()
        self.group = len(sizes)
        self.stages = []
        self.ds = ds  # output stride
        self.value_channels = out_channels
        self.key_channels = out_channels // 8


        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, self.key_channels, self.value_channels, size, self.ds)
             for size in sizes])
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_channels * self.group, out_channels, kernel_size=1, padding=0,bias=False),
            # nn.BatchNorm2d(out_channels),
        )

    def _make_stage(self, in_channels, key_channels, value_channels, size, ds):
        return PAMBlock(in_channels,key_channels,value_channels,size,ds)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]

        #  concat
        context = []
        for i in range(0, len(priors)):
            context += [priors[i]]
        output = self.conv_bn(torch.cat(context, 1))

        return output
