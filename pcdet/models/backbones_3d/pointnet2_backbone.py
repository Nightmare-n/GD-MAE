import torch
import torch.nn as nn
from ..model_utils.network_utils import make_fc_layers
from ...ops.pointnet2.pointnet2_batch import pointnet2_modules


class PointNet2MSG(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3

        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules.PointnetSAModuleMSG(
                    npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(
                pointnet2_modules.PointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
                )
            )

        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )  # (B, C, N)

        point_features = l_features[0].permute(0, 2, 1).contiguous()  # (B, N, C)
        batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1])
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0].view(-1, 3)), dim=1)
        return batch_dict


class PointNet2SAMSG(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3

        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules.PointnetSAModuleFSMSG(
                    npoint_list=self.model_cfg.SA_CONFIG.NPOINTS[k],
                    sample_range_list=self.model_cfg.SA_CONFIG.SAMPLE_RANGE[k],
                    sample_method_list=self.model_cfg.SA_CONFIG.SAMPLE_METHOD[k],
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                    dilated_radius_group=self.model_cfg.SA_CONFIG.get('DILATED_RADIUS_GROUP', False),
                    weight_gamma=self.model_cfg.SA_CONFIG.get('WEIGHT_GAMMA', 1.0),
                    aggregation_mlp=self.model_cfg.SA_CONFIG.AGGREGATION_MLPS[k],
                )
            )
            agg_mlps = self.model_cfg.SA_CONFIG.AGGREGATION_MLPS[k]
            if len(agg_mlps) > 0:
                channel_out = agg_mlps[-1]
            channel_in = channel_out

        self.num_point_features = channel_out

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:

        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        # (B, N, 3)
        xyz = xyz.view(batch_size, -1, 3)
        # (B, C, N)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None

        l_xyz, l_features, l_scores = [xyz], [features], [None]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_scores = self.SA_modules[i](l_xyz[i], features=l_features[i], scores=l_scores[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_scores.append(li_scores)
        
        aux_points_list = []
        aux_cls_preds_list = []
        for li_xyz, li_scores in zip(l_xyz, l_scores):
            if li_scores is None:
                continue
            li_bs_idx = torch.arange(batch_size).view(-1, 1, 1).repeat(1, li_xyz.shape[1], 1).to(li_xyz.device)
            aux_points_list.append(torch.cat([li_bs_idx, li_xyz], dim=-1).view(-1, 4))
            aux_cls_preds_list.append(li_scores.view(-1, 1))

        li_xyz, li_features = l_xyz[-1], l_features[-1]
        point_coords = torch.cat([
            torch.arange(batch_size).view(-1, 1, 1).repeat(1, li_xyz.shape[1], 1).to(li_xyz.device),
            li_xyz
        ], dim=-1).view(-1, 4)
        point_features = li_features.permute(0, 2, 1).contiguous().view(-1, li_features.shape[1])
        
        # (B, N, 4)
        point_coords = point_coords.view(batch_size, -1, 4)
        # (B, C, N)
        point_features = point_features.view(batch_size, -1, point_features.shape[-1]).permute(0, 2, 1).contiguous()

        batch_dict['point_coords'] = point_coords
        batch_dict['point_features'] = point_features
        batch_dict['aux_points_list'] = aux_points_list
        batch_dict['aux_cls_preds_list'] = aux_cls_preds_list
        return batch_dict
