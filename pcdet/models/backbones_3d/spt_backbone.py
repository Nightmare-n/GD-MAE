import numpy as np
import torch
import torch.nn as nn
from ..model_utils.sst_basic_block import BasicShiftBlockV2
from ..model_utils import sst_utils
from ...ops.sst_ops import sst_ops_utils
from functools import partial
from ...utils.spconv_utils import replace_feature, spconv, post_act_block, SparseBasicBlock


class SSTInputLayer(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.window_shape = model_cfg.WINDOW_SHAPE
        self.shuffle_voxels = model_cfg.SHUFFLE_VOXELS
        drop_info = model_cfg.DROP_INFO['train' if self.training else 'test']
        self.drop_info = {int(k):v for k, v in drop_info.items()}
        self.pos_temperature = model_cfg.POS_TEMPERATURE
        self.normalize_pos = model_cfg.NORMALIZE_POS
        assert self.window_shape[2] == 1

    def window_partition(self, coors, grid_size):
        voxel_info = {}
        for i in range(2):
            batch_win_inds, coors_in_win, _ = sst_utils.get_window_coors(coors, grid_size, self.window_shape, i == 1)
            voxel_info[f'batch_win_inds_shift{i}'] = batch_win_inds
            voxel_info[f'coors_in_win_shift{i}'] = coors_in_win

        return voxel_info

    def drop_single_shift(self, batch_win_inds):
        drop_info = self.drop_info
        drop_lvl_per_voxel = -torch.ones_like(batch_win_inds)
        inner_win_inds = sst_ops_utils.get_inner_win_inds(batch_win_inds)
        bincount = torch.bincount(batch_win_inds)
        num_per_voxel_before_drop = bincount[batch_win_inds]
        target_num_per_voxel = torch.zeros_like(batch_win_inds)

        for dl in drop_info:
            max_tokens = drop_info[dl]['max_tokens']
            lower, upper = drop_info[dl]['drop_range']
            range_mask = (num_per_voxel_before_drop >= lower) & (num_per_voxel_before_drop < upper)
            target_num_per_voxel[range_mask] = max_tokens
            drop_lvl_per_voxel[range_mask] = dl

        assert (target_num_per_voxel > 0).all()
        assert (drop_lvl_per_voxel >= 0).all()

        keep_mask = inner_win_inds < target_num_per_voxel
        return keep_mask, drop_lvl_per_voxel

    def drop_voxel(self, voxel_info, num_shifts):
        '''
        To make it clear and easy to follow, we do not use loop to process two shifts.
        '''

        batch_win_inds_s0 = voxel_info['batch_win_inds_shift0']
        num_all_voxel = batch_win_inds_s0.shape[0]

        voxel_keep_inds = torch.arange(num_all_voxel, device=batch_win_inds_s0.device, dtype=torch.long)

        keep_mask_s0, drop_lvl_s0 = self.drop_single_shift(batch_win_inds_s0)

        drop_lvl_s0 = drop_lvl_s0[keep_mask_s0]
        voxel_keep_inds = voxel_keep_inds[keep_mask_s0]
        batch_win_inds_s0 = batch_win_inds_s0[keep_mask_s0]

        if num_shifts == 1:
            voxel_info['voxel_keep_inds'] = voxel_keep_inds
            voxel_info['voxel_drop_level_shift0'] = drop_lvl_s0
            voxel_info['batch_win_inds_shift0'] = batch_win_inds_s0
            return voxel_info

        batch_win_inds_s1 = voxel_info['batch_win_inds_shift1']
        batch_win_inds_s1 = batch_win_inds_s1[keep_mask_s0]

        keep_mask_s1, drop_lvl_s1 = self.drop_single_shift(batch_win_inds_s1)

        # drop data in first shift again
        drop_lvl_s0 = drop_lvl_s0[keep_mask_s1]
        voxel_keep_inds = voxel_keep_inds[keep_mask_s1]
        batch_win_inds_s0 = batch_win_inds_s0[keep_mask_s1]

        drop_lvl_s1 = drop_lvl_s1[keep_mask_s1]
        batch_win_inds_s1 = batch_win_inds_s1[keep_mask_s1]

        voxel_info['voxel_keep_inds'] = voxel_keep_inds
        voxel_info['voxel_drop_level_shift0'] = drop_lvl_s0
        voxel_info['batch_win_inds_shift0'] = batch_win_inds_s0
        voxel_info['voxel_drop_level_shift1'] = drop_lvl_s1
        voxel_info['batch_win_inds_shift1'] = batch_win_inds_s1
        voxel_keep_inds = voxel_info['voxel_keep_inds']

        voxel_num_before_drop = len(voxel_info['voxel_coords'])
        voxel_info['voxel_features'] = voxel_info['voxel_features'][voxel_keep_inds]
        voxel_info['voxel_coords'] = voxel_info['voxel_coords'][voxel_keep_inds]

        # Some other variables need to be dropped, e.g., coors_in_win_shift0 and coors_in_win_shift1.
        for k, v in voxel_info.items():
            if isinstance(v, torch.Tensor) and len(v) == voxel_num_before_drop:
                voxel_info[k] = v[voxel_keep_inds]

        return voxel_info

    def forward(self, input_dict):
        voxel_features = input_dict['voxel_features']
        voxel_coords = input_dict['voxel_coords'].long()
        voxel_shuffle_inds = input_dict['voxel_shuffle_inds']
        grid_size = input_dict['grid_size']

        if self.shuffle_voxels:
            # shuffle the voxels to make the drop process uniform.
            shuffle_inds = torch.randperm(len(voxel_features))
            voxel_features = voxel_features[shuffle_inds]
            voxel_coords = voxel_coords[shuffle_inds]
            voxel_shuffle_inds = voxel_shuffle_inds[shuffle_inds]

        voxel_info = self.window_partition(voxel_coords, grid_size)
        voxel_info['voxel_features'] = voxel_features
        voxel_info['voxel_coords'] = voxel_coords
        voxel_info['voxel_shuffle_inds'] = voxel_shuffle_inds
        voxel_info = self.drop_voxel(voxel_info, 2) # voxel_info is updated in this function

        for i in range(2):
            voxel_info[f'flat2win_inds_shift{i}'] = \
                sst_utils.get_flat2win_inds_v2(voxel_info[f'batch_win_inds_shift{i}'], voxel_info[f'voxel_drop_level_shift{i}'], self.drop_info)

            voxel_info[f'pos_dict_shift{i}'] = \
                self.get_pos_embed(voxel_info[f'flat2win_inds_shift{i}'], voxel_info[f'coors_in_win_shift{i}'], voxel_info['voxel_features'].size(1))

            voxel_info[f'key_mask_shift{i}'] = \
                self.get_key_padding_mask(voxel_info[f'flat2win_inds_shift{i}'])

        return voxel_info

    def get_pos_embed(self, inds_dict, coors_in_win, feat_dim):
        '''
        Args:
        coors_in_win: shape=[N, 3], order: z, y, x
        '''
        # [N,]
        window_shape = self.window_shape
        assert window_shape[-1] == 1
        ndim = 2
        win_x, win_y = window_shape[:2]

        assert coors_in_win.size(1) == 3
        y, x = coors_in_win[:, 1] - win_y / 2, coors_in_win[:, 2] - win_x / 2
        assert (x >= -win_x / 2 - 1e-4).all()
        assert (x <= win_x / 2 - 1 + 1e-4).all()

        if self.normalize_pos:
            x = x / win_x * 2 * 3.1415  # [-pi, pi]
            y = y / win_y * 2 * 3.1415  # [-pi, pi]

        pos_length = feat_dim // ndim
        # [pos_length]
        inv_freq = torch.arange(
            pos_length, dtype=torch.float32, device=coors_in_win.device
        )
        inv_freq = self.pos_temperature ** (2 * (torch.div(inv_freq, 2, rounding_mode='floor')) / pos_length)

        # [num_tokens, pos_length]
        embed_x = x[:, None] / inv_freq[None, :]
        embed_y = y[:, None] / inv_freq[None, :]

        # [num_tokens, pos_length]
        embed_x = torch.stack([embed_x[:, ::2].sin(), embed_x[:, 1::2].cos()], dim=-1).flatten(1)
        embed_y = torch.stack([embed_y[:, ::2].sin(), embed_y[:, 1::2].cos()], dim=-1).flatten(1)

        # [num_tokens, c]
        pos_embed_2d = torch.cat([embed_x, embed_y], dim=-1)
        
        gap = feat_dim - pos_embed_2d.size(1)
        assert gap == 0 and ndim == 2

        pos_embed_dict = sst_utils.flat2window_v2(
            pos_embed_2d, inds_dict
        )

        return pos_embed_dict

    def get_key_padding_mask(self, inds_dict):
        num_all_voxel = len(inds_dict['voxel_drop_level'])
        key_padding = torch.ones((num_all_voxel, 1)).to(inds_dict['voxel_drop_level'].device).bool()

        window_key_padding_dict = sst_utils.flat2window_v2(key_padding, inds_dict)

        # logical not. True means masked
        for key, value in window_key_padding_dict.items():
            window_key_padding_dict[key] = value.logical_not().squeeze(2)

        return window_key_padding_dict


class SSTBlockV1(nn.Module):
    def __init__(self, model_cfg, input_channels, indice_key, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        encoder_cfg = model_cfg.ENCODER
        d_model = encoder_cfg.D_MODEL
        stride = encoder_cfg.STRIDE
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if stride > 1:
            self.conv_down = post_act_block(input_channels, d_model, 3, norm_fn=norm_fn, stride=stride, padding=1, indice_key=f'{indice_key}_spconv', conv_type='spconv', dim=2)
        else:
            self.conv_down = None
        self.sst_input_layer = SSTInputLayer(model_cfg.PREPROCESS)
        block_list=[]
        for i in range(encoder_cfg.NUM_BLOCKS):
            block_list.append(
                BasicShiftBlockV2(d_model, encoder_cfg.NHEAD, encoder_cfg.DIM_FEEDFORWARD,
                    encoder_cfg.DROPOUT, encoder_cfg.ACTIVATION, batch_first=False, layer_cfg=encoder_cfg.LAYER_CFG)
            )
        self.encoder_blocks = nn.ModuleList(block_list)
        self.conv_out = post_act_block(d_model, d_model, 3, norm_fn=norm_fn, indice_key=f'{indice_key}_subm', dim=2)

    def decouple_sp_tensor(self, sp_tensor):
        voxel_features = sp_tensor.features
        voxel_coords = sp_tensor.indices.long()
        voxel_coords = torch.cat([voxel_coords[:, 0:1], torch.zeros_like(voxel_coords[:, 0:1]), voxel_coords[:, 1:]], dim=-1)  # (bs_idx, 0, y, x)
        grid_size = sp_tensor.spatial_shape
        grid_size = [grid_size[1], grid_size[0], 1]  # [X, Y, 1]
        return voxel_features, voxel_coords, grid_size

    def encoder_forward(self, voxel_features, voxel_coords, grid_size):
        voxel_shuffle_inds = torch.arange(voxel_coords.shape[0], device=voxel_coords.device, dtype=torch.long)

        preprocess_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_shuffle_inds': voxel_shuffle_inds,
            'grid_size': grid_size
        }
        voxel_info = self.sst_input_layer(preprocess_dict)

        num_shifts = 2
        voxel_features = voxel_info['voxel_features']
        voxel_coords = voxel_info['voxel_coords']
        voxel_shuffle_inds = voxel_info['voxel_shuffle_inds']
        ind_dict_list = [voxel_info[f'flat2win_inds_shift{i}'] for i in range(num_shifts)]
        padding_mask_list = [voxel_info[f'key_mask_shift{i}'] for i in range(num_shifts)]
        pos_embed_list = [voxel_info[f'pos_dict_shift{i}'] for i in range(num_shifts)]

        output = voxel_features
        for i, block in enumerate(self.encoder_blocks):
            output = block(
                output, pos_embed_list, ind_dict_list, padding_mask_list
            )
        voxel_features = output

        return voxel_features, voxel_coords, voxel_shuffle_inds

    def forward(self, sp_tensor):
        if self.conv_down is not None:
            sp_tensor = self.conv_down(sp_tensor)
        voxel_features, voxel_coords, grid_size = self.decouple_sp_tensor(sp_tensor)
        voxel_features_shuffle, voxel_coords_shuffle, voxel_shuffle_inds = self.encoder_forward(voxel_features, voxel_coords, grid_size)
        voxel_features_unshuffle = torch.zeros_like(voxel_features)
        voxel_features_unshuffle[voxel_shuffle_inds] = voxel_features_shuffle
        sp_tensor = replace_feature(sp_tensor, voxel_features + voxel_features_unshuffle)
        sp_tensor = self.conv_out(sp_tensor)
        return sp_tensor


class SPTBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.sparse_shape = grid_size[[1, 0]]
        in_channels = input_channels

        sst_block_list = model_cfg.SST_BLOCK_LIST
        self.sst_blocks = nn.ModuleList()
        for sst_block_cfg in sst_block_list:
            self.sst_blocks.append(SSTBlockV1(sst_block_cfg, in_channels, sst_block_cfg.NAME))
            in_channels = sst_block_cfg.ENCODER.D_MODEL
        
        in_channels = 0
        self.deblocks = nn.ModuleList()
        for src in model_cfg.FEATURES_SOURCE:
            conv_cfg = model_cfg.FUSE_LAYER[src]
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(
                    conv_cfg.NUM_FILTER, conv_cfg.NUM_UPSAMPLE_FILTER,
                    conv_cfg.UPSAMPLE_STRIDE,
                    stride=conv_cfg.UPSAMPLE_STRIDE, bias=False
                ),
                nn.BatchNorm2d(conv_cfg.NUM_UPSAMPLE_FILTER, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True)
            ))
            in_channels += conv_cfg.NUM_UPSAMPLE_FILTER

        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // len(self.deblocks), 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // len(self.deblocks), eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.num_point_features = in_channels // len(self.deblocks)

    def forward(self, batch_dict):
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        assert torch.all(voxel_coords[:, 1] == 0)
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords[:, [0, 2, 3]].contiguous().int(),  # (bs_idx, y_idx, x_idx)
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = input_sp_tensor
        x_hidden = []
        for sst_block in self.sst_blocks:
            x = sst_block(x)
            x_hidden.append(x)

        batch_dict.update({
            'encoded_spconv_tensor': x_hidden[-1],
            'encoded_spconv_tensor_stride': 2 ** len(x_hidden)
        })

        multi_scale_3d_features, multi_scale_3d_strides = {}, {}
        for i in range(len(x_hidden)):
            multi_scale_3d_features[f'x_conv{i + 1}'] = x_hidden[i]
            multi_scale_3d_strides[f'x_conv{i + 1}'] = 2 ** (i + 1)

        spatial_features = []
        spatial_features_stride = []
        for i, src in enumerate(self.model_cfg.FEATURES_SOURCE):
            per_features = multi_scale_3d_features[src].dense()
            B, Y, X = per_features.shape[0], per_features.shape[-2], per_features.shape[-1]
            spatial_features.append(self.deblocks[i](per_features.view(B, -1, Y, X)))
            spatial_features_stride.append(multi_scale_3d_strides[src] // self.model_cfg.FUSE_LAYER[src].UPSAMPLE_STRIDE)
        spatial_features = self.conv_out(torch.cat(spatial_features, dim=1))  # (B, C, Y, X)
        spatial_features_stride = spatial_features_stride[0]

        assert spatial_features.shape[0] == batch_size and spatial_features.shape[2] == self.grid_size[1] and spatial_features.shape[3] == self.grid_size[0]
        batch_dict['multi_scale_3d_features'] = multi_scale_3d_features
        batch_dict['multi_scale_3d_strides'] = multi_scale_3d_strides
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = spatial_features_stride
        return batch_dict
