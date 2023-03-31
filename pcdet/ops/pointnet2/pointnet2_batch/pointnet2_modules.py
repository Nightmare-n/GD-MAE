from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pointnet2_utils


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, new_xyz=None):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if new_xyz is None:
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped,
                pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            ).transpose(1, 2).contiguous() if self.npoint is not None else None

        for i in range(len(self.groupers)):
            idx_cnt, new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            idx_cnt_mask = (idx_cnt > 0).float().unsqueeze(1).unsqueeze(-1)
            new_features *= idx_cnt_mask
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], 
                 use_xyz: bool = True, pool_method='max_pool'):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))

        self.pool_method = pool_method


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer"""

    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None,
                 bn: bool = True, use_xyz: bool = True, pool_method='max_pool'):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__(
            mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz,
            pool_method=pool_method
        )


class _PointnetSAModuleFSBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.groupers = None
        self.mlps = None
        self.npoint_list = []
        self.sample_range_list = [[0, -1]]
        self.sample_method_list = ['d-fps']
        self.radii = []

        self.pool_method = 'max_pool'
        self.dilated_radius_group = False
        self.weight_gamma = 1.0

        self.aggregation_mlp = None

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, new_xyz=None, scores=None):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the features
        :param new_xyz:
        :param scores: (B, N) tensor of confidence scores of points, required when using s-fps
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        batch_size = xyz.shape[0]
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if new_xyz is None:
            sample_idx_list = []
            for i in range(len(self.sample_method_list)):
                range_start, range_end = self.sample_range_list[i]
                src_xyz_slice = xyz[:, range_start: range_end, :].contiguous()
                src_features_slice = features[:, :, range_start: range_end].permute(0, 2, 1).contiguous() \
                    if features is not None else None  # (B, N, C)

                fps_type = self.sample_method_list[i]
                xyz_slice = src_xyz_slice
                features_slice = src_features_slice
                indices_slice = None
                npoints = self.npoint_list[i] 

                if fps_type == 'd-fps':
                    sample_idx = pointnet2_utils.furthest_point_sample(xyz_slice, npoints)
                elif fps_type == 'f-fps':
                    dist_matrix = pointnet2_utils.calc_dist_matrix_for_sampling(xyz_slice, features_slice, self.weight_gamma)
                    sample_idx = pointnet2_utils.furthest_point_sample_matrix(dist_matrix, npoints)
                else:
                    raise NotImplementedError

                if indices_slice is not None:
                    # (B, npoints)
                    sample_idx = torch.gather(indices_slice, -1, sample_idx.long()).view(-1, self.npoint_list[i]).int()
                sample_idx_list.append(sample_idx + range_start)

            sample_idx = torch.cat(sample_idx_list, dim=-1)
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, sample_idx).transpose(1, 2).contiguous()  # (B, npoint, 3)

        for i in range(len(self.groupers)):
            idx_cnt, new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            idx_cnt_mask = (idx_cnt > 0).float().unsqueeze(1).unsqueeze(-1)  # (B, 1, npoint, 1)

            new_features = new_features * idx_cnt_mask

            if self.pool_method == 'max_pool':
                pooled_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                pooled_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError
            
            new_features_list.append(pooled_features.squeeze(-1))  # (B, mlp[-1], npoint)

        new_features = torch.cat(new_features_list, dim=1)
        if self.aggregation_mlp is not None:
            new_features = self.aggregation_mlp(new_features)

        return new_xyz, new_features, None


class PointnetSAModuleFSMSG(_PointnetSAModuleFSBase):
    """Pointnet set abstraction layer with fusion sampling and multiscale grouping"""

    def __init__(self, *,
                 npoint_list: List[int] = None,
                 sample_range_list: List[List[int]] = None,
                 sample_method_list: List[str] = None,
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],
                 use_xyz: bool = True,
                 pool_method='max_pool',
                 dilated_radius_group: bool = False,
                 weight_gamma: float = 1.0,
                 aggregation_mlp: List[int] = None):
        """
        :param npoint_list: list of int, number of samples for every sampling method
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling method
        :param sample_method_list: list of str, list of used sampling method, d-fps or f-fps
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param dilated_radius_group: whether to use radius dilated group
        :param skip_connection: whether to add skip connection
        :param weight_gamma: gamma for s-fps, default: 1.0
        :param aggregation_mlp: list of int, spec aggregation mlp
        """
        super().__init__()

        assert npoint_list is None or len(npoint_list) == len(sample_range_list) == len(sample_method_list)
        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint_list = npoint_list
        self.sample_range_list = sample_range_list
        self.sample_method_list = sample_method_list
        self.radii = radii
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        former_radius = 0.0
        out_channels = 0
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            if dilated_radius_group:
                self.groupers.append(
                    pointnet2_utils.QueryAndGroupDilated(former_radius, radius, nsample, use_xyz=use_xyz)
                )
            else:
                self.groupers.append(
                    pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                )
            former_radius = radius
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlp = []
            for k in range(len(mlp_spec) - 1):
                shared_mlp.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlp))
            out_channels += mlp_spec[-1]

        self.pool_method = pool_method
        self.dilated_radius_group = dilated_radius_group
        self.weight_gamma = weight_gamma

        if aggregation_mlp is not None and len(aggregation_mlp) > 0:
            shared_mlp = []
            for k in range(len(aggregation_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels, aggregation_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(aggregation_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = aggregation_mlp[k]
            self.aggregation_mlp = nn.Sequential(*shared_mlp)
        else:
            self.aggregation_mlp = None


class PointnetSAModuleFS(PointnetSAModuleFSMSG):
    """Pointnet set abstraction layer with fusion sampling"""

    def __init__(self, *,
                 mlp: List[int],
                 npoint_list: List[int] = None,
                 sample_range_list: List[List[int]] = None,
                 sample_method_list: List[str] = None,
                 radius: float = None,
                 nsample: int = None,
                 use_xyz: bool = True,
                 pool_method='max_pool',
                 dilated_radius_group: bool = False,
                 weight_gamma: float = 1.0,
                 aggregation_mlp: List[int] = None):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint_list: list of int, number of samples for every sampling method
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling method
        :param sample_method_list: list of str, list of used sampling method, d-fps, f-fps or c-fps
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param dilated_radius_group: whether to use radius dilated group
        :param skip_connection: whether to add skip connection
        :param weight_gamma: gamma for s-fps, default: 1.0
        :param aggregation_mlp: list of int, spec aggregation mlp
        """
        super().__init__(
            mlps=[mlp], npoint_list=npoint_list, sample_range_list=sample_range_list,
            sample_method_list=sample_method_list, radii=[radius], nsamples=[nsample],
            use_xyz=use_xyz, pool_method=pool_method, dilated_radius_group=dilated_radius_group,
            weight_gamma=weight_gamma, aggregation_mlp=aggregation_mlp
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another"""

    def __init__(self, *, mlp: List[int], bn: bool = True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()

        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([
                nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp[k + 1]),
                nn.ReLU()
            ])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


if __name__ == "__main__":
    pass
