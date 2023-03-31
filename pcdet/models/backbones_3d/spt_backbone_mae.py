import numpy as np
import torch
import torch.nn as nn
from ...utils.spconv_utils import spconv
from .spt_backbone import SSTBlockV1
from ...utils import common_utils
from ...ops.sst_ops import sst_ops_utils
from pytorch3d.loss import chamfer_distance


class SPTBackboneMAE(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.sparse_shape = grid_size[[1, 0]]
        in_channels = input_channels

        self.mask_cfg = self.model_cfg.get('MASK_CONFIG', None)
        self.mask_ratio = self.mask_cfg.RATIO if self.mask_cfg is not None else 0.0

        sst_block_list = model_cfg.SST_BLOCK_LIST
        self.sst_blocks = nn.ModuleList()
        for sst_block_cfg in sst_block_list:
            self.sst_blocks.append(SSTBlockV1(sst_block_cfg, in_channels, sst_block_cfg.NAME))
            in_channels = sst_block_cfg.ENCODER.D_MODEL
        
        in_channels = 0
        self.decoder_deblocks = nn.ModuleList()
        for src in model_cfg.FEATURES_SOURCE:
            conv_cfg = model_cfg.FUSE_LAYER[src]
            self.decoder_deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(
                    conv_cfg.NUM_FILTER, conv_cfg.NUM_UPSAMPLE_FILTER,
                    conv_cfg.UPSAMPLE_STRIDE,
                    stride=conv_cfg.UPSAMPLE_STRIDE, bias=False
                ),
                nn.BatchNorm2d(conv_cfg.NUM_UPSAMPLE_FILTER, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True)
            ))
            in_channels += conv_cfg.NUM_UPSAMPLE_FILTER
        
        self.decoder_conv_out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // len(self.decoder_deblocks), 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // len(self.decoder_deblocks), eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        in_channels = in_channels // len(self.decoder_deblocks)

        self.decoder_pred = nn.Linear(in_channels, self.mask_cfg.NUM_PRD_POINTS * 3, bias=True)
        self.forward_ret_dict = {}

        self.num_point_features = in_channels

    def target_assigner(self, batch_dict):
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        voxel_shuffle_inds = batch_dict['voxel_shuffle_inds']
        points = batch_dict['points']
        point_inverse_indices = batch_dict['point_inverse_indices']
        voxel_mae_mask = batch_dict['voxel_mae_mask']
        # road_plane = batch_dict['road_plane']
        batch_size = batch_dict['batch_size']

        gt_points = sst_ops_utils.group_inner_inds(points[:, 1:4], point_inverse_indices, self.mask_cfg.NUM_GT_POINTS)
        gt_points = gt_points[voxel_shuffle_inds]
        voxel_centers = common_utils.get_voxel_centers(
            voxel_coords[:, 1:], 1, self.voxel_size, self.point_cloud_range, dim=3
        )  # (N, 3)
        norm_gt_points = gt_points - voxel_centers.unsqueeze(1)
        mask = voxel_mae_mask[voxel_shuffle_inds]
        pred_points = self.decoder_pred(voxel_features).view(voxel_features.shape[0], -1, 3)

        forward_ret_dict = {
            'pred_points': pred_points,  # (N, P1, 3)
            'gt_points': norm_gt_points,  # (N, P2, 3)
            'mask': mask  # (N,)
        }
        return forward_ret_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        # (N, K, 3)
        gt_points, pred_points, mask = \
            self.forward_ret_dict['gt_points'], self.forward_ret_dict['pred_points'], self.forward_ret_dict['mask']
        loss, _ = chamfer_distance(pred_points, gt_points, weights=mask)
        return loss, tb_dict

    def forward(self, batch_dict):
        all_voxel_features, all_voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        assert torch.all(all_voxel_coords[:, 1] == 0)

        voxel_mae_mask = []
        for bs_idx in range(batch_size):
            voxel_mae_mask.append(common_utils.random_masking(1, (all_voxel_coords[:, 0] == bs_idx).sum().item(), self.mask_ratio, all_voxel_coords.device)[0])
        voxel_mae_mask = torch.cat(voxel_mae_mask, dim=0)
        batch_dict['voxel_mae_mask'] = voxel_mae_mask

        input_sp_tensor = spconv.SparseConvTensor(
            features=all_voxel_features[voxel_mae_mask == 0],
            indices=all_voxel_coords[voxel_mae_mask == 0][:, [0, 2, 3]].contiguous().int(),  # (bs_idx, y_idx, x_idx)
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
            'encoded_spconv_tensor_stride': self.sparse_shape[0] // x_hidden[-1].spatial_shape[0]
        })

        multi_scale_3d_features, multi_scale_3d_strides = {}, {}
        for i in range(len(x_hidden)):
            multi_scale_3d_features[f'x_conv{i + 1}'] = x_hidden[i]
            multi_scale_3d_strides[f'x_conv{i + 1}'] = self.sparse_shape[0] // x_hidden[i].spatial_shape[0]
        
        spatial_features = []
        spatial_features_stride = []
        for i, src in enumerate(self.model_cfg.FEATURES_SOURCE):
            per_features = multi_scale_3d_features[src].dense()
            B, Y, X = per_features.shape[0], per_features.shape[-2], per_features.shape[-1]
            spatial_features.append(self.decoder_deblocks[i](per_features.view(B, -1, Y, X)))
            spatial_features_stride.append(multi_scale_3d_strides[src] // self.model_cfg.FUSE_LAYER[src].UPSAMPLE_STRIDE)
        spatial_features = self.decoder_conv_out(torch.cat(spatial_features, dim=1))  # (B, C, Y, X)
        spatial_features_stride = spatial_features_stride[0]
        
        batch_dict['multi_scale_3d_features'] = multi_scale_3d_features
        batch_dict['multi_scale_3d_strides'] = multi_scale_3d_strides
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = spatial_features_stride
        
        assert spatial_features.shape[0] == batch_size and spatial_features.shape[2] == self.grid_size[1] and spatial_features.shape[3] == self.grid_size[0]
        all_voxel_shuffle_inds = torch.arange(all_voxel_coords.shape[0], device=all_voxel_coords.device, dtype=torch.long)
        slices = [all_voxel_coords[:, i].long() for i in [0, 2, 3]]
        all_pyramid_voxel_features = spatial_features.permute(0, 2, 3, 1)[slices]

        target_dict = {
            'voxel_features': all_pyramid_voxel_features,
            'voxel_coords': all_voxel_coords,
            'voxel_shuffle_inds': all_voxel_shuffle_inds
        }
        batch_dict.update(target_dict)
        self.forward_ret_dict = self.target_assigner(batch_dict)

        return batch_dict
