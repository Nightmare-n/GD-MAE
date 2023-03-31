from functools import partial

import numpy as np
import cv2
from ...utils import box_utils, common_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points))
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def imrescale(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.imrescale, config=config)
        img = data_dict['image']
        h, w = img.shape[:2]
        img_scales = config.IMAGE_SCALES[self.mode]  # must be [(w, h)]
        if len(img_scales) > 1:
            img_scale_long = [max(s) for s in img_scales]
            img_scale_short = [min(s) for s in img_scales]
            long_edge = np.random.randint(
                min(img_scale_long),
                max(img_scale_long) + 1)
            short_edge = np.random.randint(
                min(img_scale_short),
                max(img_scale_short) + 1)
            size = [long_edge, short_edge]
        else:
            size = img_scales[0]
        if config.KEEP_RATIO:
            max_long_edge = max(size)
            max_short_edge = min(size)
            scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
            new_size = (int(w * scale_factor + 0.5), int(h * scale_factor + 0.5))
        else:
            new_size = size
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        w_scale, h_scale = new_size[0] / w, new_size[1] / h
        data_dict['image'] = img
        data_dict['image_rescale_shape'] = img.shape[:2]
        data_dict['transformation_2d_list'].append('imrescale')
        data_dict['transformation_2d_params']['imrescale'] = (w_scale, h_scale)
        if data_dict.get('gt_boxes2d', None) is not None:
            gt_boxes2d = data_dict['gt_boxes2d']
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
            gt_boxes2d[:, :4] *= scale_factor
            # TODO: should -1?
            gt_boxes2d[:, [0, 2]] = np.clip(gt_boxes2d[:, [0, 2]], 0, img.shape[1] - 1)
            gt_boxes2d[:, [1, 3]] = np.clip(gt_boxes2d[:, [1, 3]], 0, img.shape[0] - 1)
            data_dict['gt_boxes2d'] = gt_boxes2d
        return data_dict

    def imflip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.imflip, config=config)
        flip_ratio = config.FLIP_RATIO[self.mode]
        enable = np.random.choice([False, True], replace=False, p=[1 - flip_ratio, flip_ratio])
        if enable:
            img = data_dict['image']
            img = np.flip(img, axis=1)
            data_dict['image'] = img
            data_dict['transformation_2d_list'].append('imflip')
            data_dict['transformation_2d_params']['imflip'] = (enable, img.shape[1])
            if data_dict.get('gt_boxes2d', None) is not None:
                gt_boxes2d = data_dict['gt_boxes2d']
                w = img.shape[1]
                flipped = gt_boxes2d.copy()
                flipped[..., 0] = w - 1 - gt_boxes2d[..., 2]
                flipped[..., 2] = w - 1 - gt_boxes2d[..., 0]
                data_dict['gt_boxes2d'] = flipped
        return data_dict

    def imnormalize(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.imnormalize, config=config)
        img = data_dict['image']
        img = img.copy().astype(np.float32)
        mean = np.array(config.MEAN, dtype=np.float64).reshape(1, -1)
        stdinv = 1 / np.array(config.STD, dtype=np.float64).reshape(1, -1)
        if config.TO_RGB:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        data_dict['image'] = img
        return data_dict

    def impad(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.impad, config=config)
        img = data_dict['image']
        pad_h = int(np.ceil(img.shape[0] / config.SIZE_DIVISOR)) * config.SIZE_DIVISOR
        pad_w = int(np.ceil(img.shape[1] / config.SIZE_DIVISOR)) * config.SIZE_DIVISOR
        padding = (0, 0, pad_w - img.shape[1], pad_h - img.shape[0])
        img = cv2.copyMakeBorder(
            img,
            padding[1],
            padding[3],
            padding[0],
            padding[2],
            cv2.BORDER_CONSTANT,
            value=0)
        data_dict['image'] = img
        data_dict['image_pad_shape'] = img.shape[:2]
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        data_dict['transformation_2d_list'] = []
        data_dict['transformation_2d_params'] = {}
        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
