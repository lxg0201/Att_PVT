import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class MIE(nn.Module):
    def __init__(self, c=32):
        super(RMIE, self).__init__()

        self.cv4 = nn.Sequential(
            nn.Conv2d(c, 32, 3, 1, 9, dilation=9),
            nn.BatchNorm2d(32),
                 )
        self.cv = nn.Sequential(
            nn.Conv2d(c, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.cv6 = nn.Sequential(
            nn.Conv2d(32 * 3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
        )
        self.ca = ChannelAttention(32)
        self.sa = SpatialAttention()

    def forward(self, x, y):
        y = self.cv(y)
        x1 = self.ca(y)*y
        x2 = self.sa(y)*y
        x4 = self.cv4(x)
        xs = torch.cat([x1, x2, y], 1)
        x = F.relu(x4 + self.cv6(xs))
        return x

    def init_weight(self):
        weight_init(self)
        
        

class MIEs(nn.Module):
    def __init__(self):
        super(MIEs, self).__init__()
        self.MIE2, self.MIE3, self.MIE4, self.MIE5  = MIE(64), \
                                                     MIE(128), \
                                                     MIE(320), \
    def forward(self, x2, x3, x4, x5):
        out2, out3, out4, out5  = self.MIE2(x2, x2), \
                                 self.MIE3(x3, x3), self.MIE4(x4, x4), \
                                 self.MIE5(x5, x5) 

        return out2, out3, out4, out5

class CCI(nn.Module):
    def __init__(self, c =32):
        super(CCI, self).__init__()
        self.cv1 = nn.Conv2d(c, c, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(c)
        self.cv2 = nn.Conv2d(c, c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(c)
        self.cv3 = nn.Conv2d(c, c, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(c)
        
        self.cv6 = nn.Sequential(
            nn.Conv2d(32 * 4, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
        )
 

    def forward(self, left, mid, right):
        left_ = self.bn3(self.cv3(left))
        right = F.interpolate(right, size=left.shape[2:], mode='bilinear')
        right = self.bn1(self.cv1(right))
        mid = F.interpolate(mid, size=left.shape[2:], mode='bilinear')
        mid = self.bn2(self.cv2(mid))

        lm = left_ * mid
        lr = left_ * right
        mr = mid * right
        lmr = left * mid* right

        cat1 = torch.cat([lm, lr, mr, lmr], 1)
        cat = self.cv6(cat1)
        return cat

    def init_weight(self):
        weight_init(self)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features*4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class MFF(nn.Module):
    def __init__(self, num_in=32, plane_mid=16, mids=4, normalize=False):
        super(MFF, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_edge = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.Mlp = Mlp(16)
        self.conv_extend = nn.Conv2d(num_in+self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        edge = F.upsample(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        edge = self.conv_edge(edge).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)

        x_anchor = torch.matmul(edge, x_proj.reshape(n, self.num_s, -1).permute(0, 2, 1))
        x_anchor = torch.nn.functional.softmax(x_anchor, dim=-1)
        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)
        

        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        x_n_rel = self.Mlp(x_n_state)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        x_state = torch.cat([x, x_state], 1)
        out =self.conv_extend(x_state)

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Att_PVT(nn.Module):
    def __init__(self, channel=32):
        super(Att_PVT, self).__init__()

        self.backbone = pvt_v2_b2()   
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        self.MIEs = MIEs()
        self.CCI = CCI(32)

        
        self.MFF = MFF()
        
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out1 = nn.Conv2d(channel, 1, 1)
        self.out2 = nn.Conv2d(channel, 1, 1)


    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        
        x1, x2, x3, x4 = self.MIEs(x1, x2, x3, x4)
        C1 = self.CCI(x2, x3, x4)
 
        C2 = self.down(C1)
        M2 = self.MFF(x1, C2)

        prediction1 = self.out1(C2 )
        prediction2 = self.out2(M2 )

        prediction1= F.interpolate(prediction1, scale_factor=8, mode='bilinear') 
        prediction2= F.interpolate(prediction2, scale_factor=8, mode='bilinear')  
        return prediction1, prediction2


if __name__ == '__main__':
    model = Att_PVT().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1, prediction2 = model(input_tensor)
    print(prediction1.size(), prediction2.size())
