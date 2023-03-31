import torch
import numpy as np
from . import common_utils


def random_world_flip(params, reverse=False, points_3d=None, boxes_3d=None):
    if reverse:
        params = params[::-1]
    for cur_axis in params:
        if cur_axis == 'x':
            if points_3d is not None:
                points_3d[:, 1] = -points_3d[:, 1]
            if boxes_3d is not None:
                boxes_3d[:, 1] = -boxes_3d[:, 1]
                boxes_3d[:, 6] = -boxes_3d[:, 6]
        elif cur_axis == 'y':
            if points_3d is not None:
                points_3d[:, 0] = -points_3d[:, 0]
            if boxes_3d is not None:
                boxes_3d[:, 0] = -boxes_3d[:, 0]
                boxes_3d[:, 6] = -(boxes_3d[:, 6] + np.pi)
        else:
            raise NotImplementedError
    return points_3d, boxes_3d


def random_world_rotation(params, reverse=False, points_3d=None, boxes_3d=None):
    if reverse:
        params = -params
    if points_3d is not None:
        points_3d = common_utils.rotate_points_along_z(points_3d.unsqueeze(0), points_3d.new_tensor([params]))[0]
    if boxes_3d is not None:
        boxes_3d[:, 0:3] = common_utils.rotate_points_along_z(boxes_3d[:, 0:3].unsqueeze(0), boxes_3d.new_tensor([params]))[0]
        boxes_3d[:, 6] += params
    return points_3d, boxes_3d


def random_world_scaling(params, reverse=False, points_3d=None, boxes_3d=None):
    if reverse:
        params = 1.0 / params
    if points_3d is not None:
        points_3d[:, :3] *= params
    if boxes_3d is not None:
        boxes_3d[:, :6] *= params
    return points_3d, boxes_3d


def random_world_translation(params, reverse=False, points_3d=None, boxes_3d=None):
    if reverse:
        params = -params
    if points_3d is not None:
        points_3d[:, :3] += points_3d.new_tensor(params)
    if boxes_3d is not None:
        boxes_3d[:, :3] += boxes_3d.new_tensor(params)
    return points_3d, boxes_3d


def imrescale(params, reverse=False, points_2d=None, boxes_2d=None):
    w_scale, h_scale = params
    if reverse:
        w_scale, h_scale = 1.0 / w_scale, 1.0 / h_scale
    if points_2d is not None:
        points_2d[:, 0:2] *= points_2d.new_tensor([w_scale, h_scale])
    if boxes_2d is not None:
        boxes_2d[:, :4] *= boxes_2d.new_tensor([w_scale, h_scale, w_scale, h_scale])
    return points_2d, boxes_2d


def imflip(params, reverse=False, points_2d=None, boxes_2d=None):
    enable_x, rescale_w = params
    if enable_x:
        if points_2d is not None:
            points_2d[:, 0] = rescale_w - 1 - points_2d[..., 0]
        if boxes_2d is not None:
            flipped = boxes_2d.clone()
            flipped[:, 0] = rescale_w - 1 - boxes_2d[..., 2]
            flipped[:, 2] = rescale_w - 1 - boxes_2d[..., 0]
            boxes_2d = flipped
    return points_2d, boxes_2d


def points_lidar2img(points_3d, proj_mat, with_depth=False):
    """Project points from lidar coordicates to image coordinates.

    Args:
        points_3d (torch.Tensor): Points in shape (N, 3).
        proj_mat (torch.Tensor): (3, 4), transformation matrix between coordinates(left R).
        with_depth (bool, optional): Whether to keep depth in the output.
            Defaults to False.

    Returns:
        torch.Tensor: Points in image coordinates with shape [N, 2].
    """
    # (N, 4)
    points_4 = torch.cat([points_3d, points_3d.new_ones((points_3d.shape[0], 1))], dim=-1)
    point_2d = torch.matmul(points_4, proj_mat.t())
    point_2d_res = point_2d[..., :2] / torch.clamp(point_2d[..., 2:3], min=1e-5, max=1e5)

    if with_depth:
        return torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)
    return point_2d_res
