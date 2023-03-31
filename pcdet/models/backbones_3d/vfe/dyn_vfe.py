import torch
import torch.nn as nn
from torch.nn import functional as F
from .vfe_template import VFETemplate
import torch_scatter
from ...model_utils.network_utils import make_fc_layers
from functools import partial
from ....utils import common_utils


class DynVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, grid_size, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.sample_type = model_cfg.get('TYPE', 'mean')

        mlps = model_cfg.get('MLPS', None)
        self.dvfe_mlps = None
        if mlps is not None:
            self.with_distance = self.model_cfg.WITH_DISTANCE
            self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
            self.use_cluster_xyz = self.model_cfg.USE_CLUSTER_XYZ
            input_channels = num_point_features
            if self.use_cluster_xyz:
                input_channels += 3
            if self.use_absolute_xyz:
                input_channels += 3
            if self.with_distance:
                input_channels += 1

            norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
            self.dvfe_mlps = nn.ModuleList()
            for i in range(len(mlps)):
                self.dvfe_mlps.append(make_fc_layers(mlps[i], input_channels, norm_fn=norm_fn))
                input_channels = mlps[i][-1] if i == len(mlps) - 1 else mlps[i][-1] * 2
        else:
            input_channels = num_point_features
        agg_mlp = model_cfg.get('AGGREGATION_MLPS', None)
        self.aggregation_mlp = None
        if agg_mlp is not None:
            self.aggregation_mlp = make_fc_layers(agg_mlp, input_channels)
            input_channels = agg_mlp[-1]

        self.num_point_features = input_channels
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.grid_size = grid_size

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        points = batch_dict['points']
        pc_range = points.new_tensor(self.point_cloud_range)
        voxel_size = points.new_tensor(self.voxel_size)
        keep, coords = common_utils.get_in_range_mask(points, pc_range, voxel_size, self.grid_size)
        points, coords = points[keep], coords[keep]
        coords = torch.cat([points[:, 0:1].long(), torch.flip(coords, dims=[-1])], dim=-1)  # [bs_idx, Z, Y, X]
        sampled_coords, inverse_indices = coords.unique(sorted=False, return_inverse=True, dim=0)

        if self.sample_type == 'random':
            _, sampled_indices = torch_scatter.scatter_max(torch.arange(len(coords)).to(coords.device), inverse_indices, dim=0)
            x = points[:, 1:][sampled_indices]
        elif self.sample_type == 'nearest':
            points_mean = torch_scatter.scatter(points[:, 1:4], inverse_indices, dim=0, reduce='mean')
            _, sampled_indices = torch_scatter.scatter_min(
                torch.linalg.norm(points[:, 1:4] - points_mean[inverse_indices], dim=-1), 
                inverse_indices, dim=0
            )
            x = points[:, 1:][sampled_indices]
        elif self.sample_type == 'mean':
            x = torch_scatter.scatter(points[:, 1:], inverse_indices, dim=0, reduce='mean')
        else:
            raise NotImplementedError

        if self.dvfe_mlps is not None:
            sampled_xyz = x[:, :3]
            f_cluster = points[:, 1:4] - sampled_xyz[inverse_indices]

            f_center = torch.zeros_like(f_cluster)
            f_center[:, 0] = points[:, 1] - ((coords[:, 3] + 0.5) * voxel_size[0] + pc_range[0])
            f_center[:, 1] = points[:, 2] - ((coords[:, 2] + 0.5) * voxel_size[1] + pc_range[1])
            f_center[:, 2] = points[:, 3] - ((coords[:, 1] + 0.5) * voxel_size[2] + pc_range[2])

            x = [f_center]
            if self.use_absolute_xyz:
                x.append(points[:, 1:])
            else:
                x.append(points[:, 4:])
            if self.use_cluster_xyz:
                x.append(f_cluster)

            if self.with_distance:
                points_dist = torch.linalg.norm(points[:, 1:4], dim=-1, keepdim=True)
                x.append(points_dist)
            x = torch.cat(x, dim=-1)

            for k in range(len(self.dvfe_mlps)):
                x = self.dvfe_mlps[k](x)
                x_max = torch_scatter.scatter_max(x, inverse_indices, dim=0)[0]
                if k == len(self.dvfe_mlps) - 1:
                    x = x_max
                else:
                    x = torch.cat([x, x_max[inverse_indices]], dim=-1)
            if self.aggregation_mlp is not None:
                x = self.aggregation_mlp(x)            

        batch_dict['points'] = points
        batch_dict['point_coords'] = coords
        batch_dict['point_inverse_indices'] = inverse_indices
        batch_dict['voxel_coords'] = sampled_coords  # (M1+M2+..., 4)
        batch_dict['pillar_features'] = x
        batch_dict['voxel_features'] = x  # (M1+M2+..., C)

        return batch_dict
