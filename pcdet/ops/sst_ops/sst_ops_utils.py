import torch
from . import sst_ops_cuda


def get_inner_win_inds(group_inds):
    """
    Args:
        group_inds: (N,)
    """
    out_inds = torch.zeros_like(group_inds) - 1
    sst_ops_cuda.ingroup_inds_wrapper(group_inds.contiguous(), out_inds)
    return out_inds


def group_inner_inds(points, inverse_inds, K):
    """
    Args:
        points: (N, C)
        inverse_inds: (N, )
    Return:
        group_points: (valid_voxel_num + 1, K, C)
    """
    valid_voxel_num = inverse_inds.max().item()
    group_inds = torch.full((valid_voxel_num + 1, K), -1, dtype=torch.long, device=points.device)
    sst_ops_cuda.group_inner_inds_wrapper(inverse_inds.contiguous(), group_inds)
    group_points = points[group_inds]
    return group_points
