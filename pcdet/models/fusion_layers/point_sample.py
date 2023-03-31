import torch
import torch.nn as nn
from ...utils import transform_utils


def img_grid_sample(pts_2d, img_shape, img_feats):
    """
    Args:
        pts_2d: (N, 2)
        img_shape: [H, W]
        img_feats: (1, C, H', W')
    """
    coor_x, coor_y = torch.split(pts_2d, 1, dim=1)
    h, w = img_shape
    coor_y = coor_y / (h - 1) * 2 - 1
    coor_x = coor_x / (w - 1) * 2 - 1
    grid = torch.cat([coor_x, coor_y], dim=1).unsqueeze(0).unsqueeze(0)

    point_features = torch.nn.functional.grid_sample(
        img_feats,
        grid,
        align_corners=True)
    point_features = point_features.squeeze().t()
    return point_features


class PointSample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_dict):
        """
        Args:
            img_feats: (B, C, H/4, W/4)
            sampled_points: (N1+N2+..., 4+C), (B, N, 3+C)
        Return:
            point_feats: (N1+N2+..., C), (B, N, C)
        """
        batch_size = batch_dict['batch_size']
        img_feats = batch_dict['image_features']
        raw_points = batch_dict['sampled_points'].clone()
        point_feats = []
        for index in range(batch_size):
            if len(raw_points.shape) == 2:
                cur_points = raw_points[raw_points[:, 0] == index][:, 1:4]
            else:
                cur_points = raw_points[index][:, :3]
            proj_mat = batch_dict['trans_cam_to_img'][index] @ batch_dict['trans_lidar_to_cam'][index]

            if batch_dict.get('transformation_3d_list', None) is not None:
                cur_3d_trans_list = batch_dict['transformation_3d_list'][index]
                cur_3d_trans_params = batch_dict['transformation_3d_params'][index]
                for key in cur_3d_trans_list[::-1]:
                    cur_points, _ = getattr(transform_utils, key)(cur_3d_trans_params[key], reverse=True, points_3d=cur_points)

            cur_points_2d = transform_utils.points_lidar2img(cur_points, proj_mat)

            cur_2d_trans_list = batch_dict['transformation_2d_list'][index]
            cur_2d_trans_params = batch_dict['transformation_2d_params'][index]
            for key in cur_2d_trans_list:
                cur_points_2d, _ = getattr(transform_utils, key)(cur_2d_trans_params[key], reverse=False, points_2d=cur_points_2d)

            img_shape = batch_dict['image'].shape[2:]
            cur_img_feats = img_feats[index: index + 1]
            cur_point_feats = img_grid_sample(cur_points_2d, img_shape, cur_img_feats)

            point_feats.append(cur_point_feats)
        if len(raw_points.shape) == 2:
            point_feats = torch.cat(point_feats, dim=0)  # (N1+N2+..., C)
        else:
            point_feats = torch.stack(point_feats, dim=0)  # (B, N, C)
        return point_feats
