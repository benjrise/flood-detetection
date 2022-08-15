"""
FROM CANNAB SP7
"""
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)


class EfficientNet_Unet(nn.Module):
    def __init__(self, 
                name='efficientnet-b0', 
                pretrained=True, 
                in_channels=3,
                mode="foundation",
                **kwargs):
        super(EfficientNet_Unet, self).__init__()

        enc_sizes = {
            'efficientnet-b0': [16, 24, 40, 112, 1280],
            'efficientnet-b1': [16, 24, 40, 112, 1280],
            'efficientnet-b2': [16, 24, 48, 120, 1408],
            'efficientnet-b3': [24, 32, 48, 136, 1536],
            'efficientnet-b4': [24, 32, 56, 160, 1792],
            'efficientnet-b5': [24, 40, 64, 176, 2048],
            'efficientnet-b6': [32, 40, 72, 200, 2304],
            'efficientnet-b7': [32, 48, 80, 224, 2560],
            'efficientnet-b8': [32, 56, 88, 248, 2816]
        }

        encoder_filters = enc_sizes[name]
        decoder_filters = np.asarray([48, 64, 128, 160, 320]) 

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
        
        self.mode = mode
        if self.mode == "foundation":
            self.road_layer = nn.Conv2d(decoder_filters[-5], 8, 1, stride=1, padding=0)
            self.building_layer = nn.Conv2d(decoder_filters[-5], 1, 1, stride=1, padding=0)
        else:
            self.flood_layer = self.make_final_classifier(decoder_filters[-5], num_classes=5)

        self._initialize_weights()

        if pretrained:
            self.encoder = EfficientNet.from_pretrained(name)
        else:    
            self.encoder = EfficientNet.from_name(name)

        if in_channels != 3:
            weights = self.encoder._conv_stem.weight.clone()
            out_channels = self.encoder._conv_stem.out_channels
            # Need to set image size, doesn't make any difference here, but if production version needed probably need to change
            # to correct image input size or to Conv2dDynamicSamePadding
            new_in_layer = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=(3,3), stride=(2,2), bias=False, image_size=240)
            new_in_layer.weight.data = nn.init.kaiming_normal_(new_in_layer.weight.data)
            new_in_layer.weight[:, :3, :, :].data[...] = Variable(weights, requires_grad=True)
            self.encoder._conv_stem = new_in_layer
        elif in_channels < 3:
            raise NotImplementedError

    def extract_features(self, inp):
        out = []

        # Stem
        x = self.encoder._swish(self.encoder._bn0(self.encoder._conv_stem(inp)))

        # Blocks
        for idx, block in enumerate(self.encoder._blocks):
            drop_connect_rate = self.encoder._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.encoder._blocks)
            y = block(x, drop_connect_rate=drop_connect_rate)
            if y.size()[-1] != x.size()[-1]:
                out.append(x)
            x = y
        
        # Head
        x = self.encoder._swish(self.encoder._bn1(self.encoder._conv_head(x)))
        out.append(x)

        return out


    def forward(self, x):
        batch_size, C, H, W = x.shape

        enc1, enc2, enc3, enc4, enc5 = self.extract_features(x)

        y_out, x_out = enc4.shape[2], enc4.shape[3]
        dec6 = self.conv6(F.interpolate(enc5, size=(y_out, x_out)))
        dec6 = self.conv6_2(torch.cat([dec6, enc4
                ], 1))

        y_out, x_out = enc3.shape[2], enc3.shape[3]
        dec7 = self.conv7(F.interpolate(dec6, size=(y_out, x_out)))
        dec7 = self.conv7_2(torch.cat([dec7, enc3
                ], 1))
        
        y_out, x_out = enc2.shape[2], enc2.shape[3]
        dec8 = self.conv8(F.interpolate(dec7, size=(y_out, x_out)))
        dec8 = self.conv8_2(torch.cat([dec8, enc2
                ], 1))

        y_out, x_out = enc1.shape[2], enc1.shape[3]
        dec9 = self.conv9(F.interpolate(dec8, size=(y_out, x_out)))
        dec9 = self.conv9_2(torch.cat([dec9, 
                enc1
                ], 1))
        
        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2, ))

        if self.mode == "foundation":
            building = self.building_layer(dec10)
            road = self.road_layer(dec10)
            return building, road
        else:
            flood = self.flood_layer(dec10)
            return flood


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_final_classifier(self, in_filters, num_classes):
        return nn.Sequential(
            nn.Conv2d(in_filters, num_classes, 3, padding=1)
        )

    def make_final_classifier2(self, in_filters, num_classes):
        return nn.Sequential(nn.Conv2d(in_filters, 32, 3, padding=1),
                            nn.Conv2d(32, num_classes, 3, padding=1))

class EfficientNet_Unet_Double(nn.Module):
    def __init__(self, 
                name='efficientnet-b0', 
                pretrained=True, 
                num_classes=5,
                in_channels=3):
        super(EfficientNet_Unet_Double, self).__init__()

        enc_sizes = {
            'efficientnet-b0': [16, 24, 40, 112, 1280],
            'efficientnet-b1': [16, 24, 40, 112, 1280],
            'efficientnet-b2': [16, 24, 48, 120, 1408],
            'efficientnet-b3': [24, 32, 48, 136, 1536],
            'efficientnet-b4': [24, 32, 56, 160, 1792],
            'efficientnet-b5': [24, 40, 64, 176, 2048],
            'efficientnet-b6': [32, 40, 72, 200, 2304],
            'efficientnet-b7': [32, 48, 80, 224, 2560],
            'efficientnet-b8': [32, 56, 88, 248, 2816]
        }

        encoder_filters = enc_sizes[name]
        decoder_filters = np.asarray([48, 64, 128, 160, 320])

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
        
        self.res = nn.Conv2d(decoder_filters[-5] * 2, num_classes, 1, stride=1, padding=0)

        self._initialize_weights()

        if pretrained:
            self.encoder = EfficientNet.from_pretrained(name)
        else:    
            self.encoder = EfficientNet.from_name(name)


    def extract_features(self, inp):
        out = []

        # Stem
        x = self.encoder._swish(self.encoder._bn0(self.encoder._conv_stem(inp)))

        # Blocks
        for idx, block in enumerate(self.encoder._blocks):
            drop_connect_rate = self.encoder._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.encoder._blocks)
            y = block(x, drop_connect_rate=drop_connect_rate)
            if y.size()[-1] != x.size()[-1]:
                out.append(x)
            x = y
        
        # Head
        x = self.encoder._swish(self.encoder._bn1(self.encoder._conv_head(x)))
        out.append(x)

        return out


    def forward1(self, x):
        batch_size, C, H, W = x.shape

        enc1, enc2, enc3, enc4, enc5 = self.extract_features(x)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4
                ], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3
                ], 1))
        
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2
                ], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, 
                enc1
                ], 1))
        
        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return dec10


    def forward(self, preimg, postimg, foundation=None):

        dec10_0 = self.forward1(preimg)
        dec10_1 = self.forward1(postimg)

        if foundation is not None:
            dec10 = torch.cat([dec10_0, dec10_1, foundation], 1)
        else:
            dec10 = torch.cat([dec10_0, dec10_1], 1)

        return self.res(dec10)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == "__main__":
    efficient = EfficientNet.from_name("efficientnet-b1")
    #print(efficient)
    # new_in_layer = Conv2dStaticSamePadding()
    print(efficient._conv_stem)
    weights = efficient._conv_stem.weight.clone()
    new_in_layer = Conv2dStaticSamePadding(6, 32, kernel_size=(3,3), stride=(2,2), bias=False, image_size=240)
    new_in_layer.weight.data = nn.init.kaiming_normal_(new_in_layer.weight.data)
    new_in_layer.weight[:, :3, :, :].data[...] = Variable(weights, requires_grad=True)
    # print(new_in_layer)


    x = torch.rand(1, 3, 512, 512)
    efficient._conv_stem = new_in_layer
    print(efficient._conv_stem)
    #sprint(x)