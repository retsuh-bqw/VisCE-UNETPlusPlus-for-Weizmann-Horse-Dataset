import torch
import torchvision
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict



class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(Conv_BN_ReLU, self).__init__()
        self.basic_conv = nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                           padding=padding, bias=False)),
                        ('bn', nn.BatchNorm2d(num_features =out_planes)),
                        ('act', nn.ReLU(inplace=False))   
        ]))

    def forward(self, x):
        return self.basic_conv(x)



class BasicDecoderBlock(nn.Module):
    def __init__(self,
                 in_channels=512,
                 out_channels=256,
                 ):
        super().__init__()
        self.decoder_block = nn.Sequential(
                            Conv_BN_ReLU(in_channels, out_channels // 2, kernel_size=3, stride=1, padding=1),
                            Conv_BN_ReLU(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1),
                            Conv_BN_ReLU(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1)
        )
 
    def forward(self, x):
        return self.decoder_block(torch.cat(x, dim=1))
 
 
def upsample(x,scale_factor=2):
        x = F.interpolate(input=x, scale_factor=scale_factor, mode='nearest')
        return x


class UnetPP(nn.Module):
    def __init__(self,
                 num_channels=3,
                 num_class=1,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 ):
        super().__init__()
        backbone = models.resnet34(weights=None)
        out_channels = [64, 64, 128, 256]
        self.weights = nn.Parameter(torch.ones(4))

        
        # Head
        self.head_conv = backbone.conv1
        self.head_bn = backbone.bn1
        self.head_relu = backbone.relu
        self.maxpool = backbone.maxpool

        # Encoder Blocks
        self.encoder_blocks = nn.ModuleList([backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4])
 
        # Decoder_Layer1
        self.decoder0_1 = BasicDecoderBlock(in_channels=64+64, out_channels=out_channels[0])

        # Decoder_Layer2
        self.decoder1_1 = BasicDecoderBlock(in_channels=128+64, out_channels=out_channels[1])
        self.decoder0_2 = BasicDecoderBlock(in_channels=64+64+64, out_channels=out_channels[1])

        # Decoder_Layer3
        self.decoder2_1 = BasicDecoderBlock(in_channels=128+256, out_channels=out_channels[2])
        self.decoder1_2 = BasicDecoderBlock(in_channels=64+64+128, out_channels=out_channels[2])
        self.decoder0_3 = BasicDecoderBlock(in_channels=64+64+64+128, out_channels=out_channels[2])

        # Decoder_Layer4
        self.decoder3_1 = BasicDecoderBlock(in_channels=512+256,out_channels=out_channels[3])
        self.decoder2_2 = BasicDecoderBlock(in_channels=128+128+256,out_channels=out_channels[3])
        self.decoder1_3 = BasicDecoderBlock(in_channels=64+64+128+256,out_channels=out_channels[3])
        self.decoder0_4 = BasicDecoderBlock(in_channels=64+64+64+128+256,out_channels=out_channels[3])

        # final output
        self.get_output = nn.ModuleList([nn.Conv2d(c, num_class, kernel_size=1) for c in out_channels])

 
    def forward(self, x):
        b,_, H,W = x.shape
        x = self.head_conv(x)  
        x = self.head_bn(x)
        x_ = self.head_relu(x)  

        # Encoder
        x = self.maxpool(x_)
        e1 = self.encoder_blocks[0](x)  
        e2 = self.encoder_blocks[1](e1)  
        e3 = self.encoder_blocks[2](e2)  
        e4 = self.encoder_blocks[3](e3)  

        x0_0 = x_
        x1_0 = e1
        x2_0 = e2
        x3_0 = e3
        x4_0 = e4
        
        # Layer1
        x0_1 = self.decoder0_1([x0_0, upsample(x1_0)])  

        # Layer2
        x1_1 = self.decoder1_1([x1_0, upsample(x2_0)])
        x0_2 = self.decoder0_2([x0_0, x0_1,  upsample(x1_1)])

        # Layer3
        x2_1 = self.decoder2_1([x2_0, upsample(x3_0)])
        x1_2 = self.decoder1_2([x1_0, x1_1, upsample(x2_1)])
        x0_3 = self.decoder0_3([x0_0, x0_1, x0_2,  upsample(x1_2)])

        # Layer4
        x3_1 = self.decoder3_1([x3_0, upsample(x4_0)])
        x2_2 = self.decoder2_2([x2_0, x2_1, upsample(x3_1)])
        x1_3 = self.decoder1_3([x1_0, x1_1, x1_2, upsample(x2_2)])
        x0_4 = self.decoder0_4([x0_0, x0_1, x0_2,  x0_3,  upsample(x1_3)])

        
        pred = torch.zeros((b,1,H//2,W//2)).cuda()
        for idx, v in enumerate([x0_1, x0_2, x0_3, x0_4]):
            pred += self.weights[idx] * self.get_output[idx](v)

        return F.interpolate(pred, size=(H,W), mode='bilinear', align_corners=False)

   