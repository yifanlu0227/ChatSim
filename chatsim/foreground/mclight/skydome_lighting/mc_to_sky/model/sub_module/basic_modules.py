import torch
import torch.nn as nn
from torch import Tensor
from mc_to_sky.model.sub_module.residual import build_layer, build_up_layer


class EncoderNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        layer_num = len(args["layer_channels"])
        layer_channels = args["layer_channels"]
        kernel_size = args["kernel_size"]
        strides = args["strides"]
        block_nums = args["block_nums"]
        use_bn = args["use_bn"]
        act = args["act"]
        inplanes = args["in_ch"]

        module_list = []
        for i in range(layer_num):
            module_list.append(
                build_layer(
                    inplanes,
                    layer_channels[i],
                    kernel_size,
                    strides[i],
                    block_nums[i],
                    act,
                    use_bn,
                )
            )
            inplanes = layer_channels[i]

        self.model = nn.Sequential(*module_list)

    def forward(self, x):
        return self.model(x)


class DecoderNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        layer_num = len(args["layer_channels"])
        layer_channels = args["layer_channels"]
        kernel_size = args["kernel_size"]
        upstrides = args["upstrides"]
        block_nums = args["block_nums"]
        use_bn = args["use_bn"]
        act = args["act"]
        inplanes = args["in_ch"]

        module_list = []
        for i in range(layer_num):
            module_list.append(
                build_up_layer(
                    inplanes,
                    layer_channels[i],
                    kernel_size,
                    upstrides[i],
                    block_nums[i],
                    act,
                    use_bn,
                )
            )
            inplanes = layer_channels[i]

        self.model = nn.Sequential(*module_list)

    def forward(self, x):
        return self.model(x)

class UNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        layer_num = len(args["layer_channels"])
        self.layer_num = layer_num

        # down sample branch
        layer_channels = args["layer_channels"]
        strides = args["strides"]
        block_nums = args["block_nums"]

        # upsample branch
        up_in_channels = args["up_in_channels"]
        up_layer_channels = args["up_layer_channels"]
        up_strides = args["up_strides"]
        up_block_nums = args["up_block_nums"]

        kernel_size = args["kernel_size"]
        use_bn = args["use_bn"]
        act = args["act"]

        # merge latent vector
        self.inject_latent = args['inject_latent']

        inplanes = args["in_ch"]
        self.final_conv_to_RGB = args['final_conv_to_RGB']
        final_act = args.get('final_conv_to_RGB_act', "relu")
        
        if final_act == 'relu':
            self.final_act = nn.ReLU()
        elif final_act == 'sigmoid':
            self.final_act = nn.Sigmoid()

        self.module_dict = nn.ModuleDict()
        for i in range(layer_num):
            self.module_dict[f'down{i}'] = build_layer(inplanes,
                                                       layer_channels[i],
                                                       kernel_size,
                                                       strides[i],
                                                       block_nums[i],
                                                       act,
                                                       use_bn)
            inplanes = layer_channels[i]

            self.module_dict[f'up{i}'] = build_up_layer(up_in_channels[i],
                                                     up_layer_channels[i],
                                                     kernel_size,
                                                     up_strides[i],
                                                     up_block_nums[i],
                                                     act,
                                                     use_bn)        

        # to HDR RGB
        if self.final_conv_to_RGB:
            self.final_conv = nn.Sequential(nn.Conv2d(up_layer_channels[-1], 3, kernel_size=3, stride=1, padding=1),
                                            self.final_act)           

    def forward(self, x, latent_feature=None):
        """
        Args:
            x : tensor 
                shape [N, C, H, W], C = 7
            latent_feature : tensor
                shape [N, C2, h, w], the small feature map from LDR encoder
        """
        # down branch
        x_downs = []
        for i in range(self.layer_num):
            x = self.module_dict[f'down{i}'](x)
            x_downs.append(x)
        

        # fuse with latent vector
        if self.inject_latent:
            x = torch.cat([x, latent_feature], dim = 1)
        x = self.module_dict[f'up0'](x)

        # up branch
        for i in range(1, self.layer_num): # reverse order

            x = torch.cat([x, x_downs[self.layer_num - 1 - i]], dim = 1)
            x = self.module_dict[f'up{i}'](x)

        if self.final_conv_to_RGB:
            x = self.final_conv(x)

        return x


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        layer_num = len(args["layer_channels"])
        layer_channels = args["layer_channels"]
        act = args['act']
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'selu':
            self.act = nn.SELU(inplace=True)
        elif act == 'elu':
            self.act = nn.ELU(inplace=True)
        elif act == 'none':
            self.act = nn.Identity()

        inplanes = args['in_ch']

        module_list = []
        for i in range(layer_num):
            module_list.append(nn.Linear(inplanes, layer_channels[i]))
            module_list.append(self.act)
            inplanes = layer_channels[i]
        
        self.model = nn.Sequential(*module_list)

    def forward(self, x):
        return self.model(x)
