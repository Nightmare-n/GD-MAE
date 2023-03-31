import numpy as np
import torch
import torch.nn as nn


class SSTBEVBackbone(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        input_channels = model_cfg.NUM_FILTER
        self.conv_shortcut = model_cfg.CONV_SHORTCUT
        conv_kwargs = model_cfg.CONV_KWARGS
        conv_list = []
        for i in range(len(conv_kwargs)):
            conv_kwargs_i = conv_kwargs[i]
            conv_list.append(nn.Sequential(
                nn.Conv2d(input_channels, **conv_kwargs_i, bias=False),
                nn.BatchNorm2d(conv_kwargs_i['out_channels'], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True)
            ))
            input_channels = conv_kwargs_i['out_channels']
        self.conv_layer = nn.ModuleList(conv_list)

        self.num_bev_features = input_channels

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        output = data_dict['spatial_features']
        if len(self.conv_layer) > 0:
            for i, conv in enumerate(self.conv_layer):
                temp = conv(output)
                if temp.shape == output.shape and i in self.conv_shortcut:
                    output = temp + output
                else:
                    output = temp

        data_dict['spatial_features_2d'] = output
        return data_dict