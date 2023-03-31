from functools import partial

import numpy as np

from ...utils import common_utils
from . import database_sampler
import cv2


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def random_world_drop(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_drop, config=config)
        points = data_dict['points']
        enable = np.random.choice([False, True], replace=False, p=[1 - config['PROBABILITY'], config['PROBABILITY']])
        drop_ratio = config['DROP_RATIO'] if enable else 0.0

        choice = np.arange(0, len(points), dtype=np.int32)
        choice = np.random.choice(choice, int((1 - drop_ratio) * len(points)), replace=False)
        points = points[choice]
        data_dict['points'] = points
        return data_dict

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        points = data_dict['points']
        gt_boxes = data_dict['gt_boxes'] if 'gt_boxes' in data_dict else None
        params = []
        for cur_axis in config['ALONG_AXIS_LIST']:
            enable = np.random.choice([False, True], replace=False, p=[1 - config['PROBABILITY'], config['PROBABILITY']])
            if enable:
                params.append(cur_axis)
        if 'random_world_flip' in data_dict['transformation_3d_list']:
            params = data_dict['transformation_3d_params']['random_world_flip']

        for cur_axis in params:
            if cur_axis == 'x':
                points[:, 1] = -points[:, 1]
                if 'gt_boxes' in data_dict:
                    gt_boxes[:, 1] = -gt_boxes[:, 1]
                    gt_boxes[:, 6] = -gt_boxes[:, 6]
                    if gt_boxes.shape[1] > 7:
                        gt_boxes[:, 8] = -gt_boxes[:, 8]
            elif cur_axis == 'y':
                points[:, 0] = -points[:, 0]
                if 'gt_boxes' in data_dict:
                    gt_boxes[:, 0] = -gt_boxes[:, 0]
                    gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
                    if gt_boxes.shape[1] > 7:
                        gt_boxes[:, 7] = -gt_boxes[:, 7]
            else:
                raise NotImplementedError
        if 'random_world_flip' not in data_dict['transformation_3d_list']:
            data_dict['transformation_3d_list'].append('random_world_flip')
            data_dict['transformation_3d_params']['random_world_flip'] = params
        data_dict['points'] = points
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        points = data_dict['points']
        gt_boxes = data_dict['gt_boxes'] if 'gt_boxes' in data_dict else None

        enable = np.random.choice([False, True], replace=False, p=[1 - config['PROBABILITY'], config['PROBABILITY']])
        rot_range = config['WORLD_ROT_ANGLE'] if enable else [0.0, 0.0]

        noise_rotation = np.random.uniform(rot_range[0], rot_range[1]) \
            if 'random_world_rotation' not in data_dict['transformation_3d_list'] \
            else data_dict['transformation_3d_params']['random_world_rotation']
        points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
        if 'gt_boxes' in data_dict:
            gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
            gt_boxes[:, 6] += noise_rotation
            if gt_boxes.shape[1] > 7:
                gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
                    np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
                    np.array([noise_rotation])
                )[0][:, 0:2]
        if 'random_world_rotation' not in data_dict['transformation_3d_list']:
            data_dict['transformation_3d_list'].append('random_world_rotation')
            data_dict['transformation_3d_params']['random_world_rotation'] = noise_rotation
        data_dict['points'] = points
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        points = data_dict['points']
        gt_boxes = data_dict['gt_boxes'] if 'gt_boxes' in data_dict else None

        enable = np.random.choice([False, True], replace=False, p=[1 - config['PROBABILITY'], config['PROBABILITY']])
        scale_range = config['WORLD_SCALE_RANGE'] if enable else [1.0, 1.0]

        noise_scale = np.random.uniform(scale_range[0], scale_range[1]) \
            if 'random_world_scaling' not in data_dict['transformation_3d_list'] \
            else data_dict['transformation_3d_params']['random_world_scaling']
        points[:, :3] *= noise_scale
        if 'gt_boxes' in data_dict:
            gt_boxes[:, :6] *= noise_scale
        if 'random_world_scaling' not in data_dict['transformation_3d_list']:
            data_dict['transformation_3d_list'].append('random_world_scaling')
            data_dict['transformation_3d_params']['random_world_scaling'] = noise_scale
        data_dict['points'] = points
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def random_world_translation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_translation, config=config)
        points = data_dict['points']
        gt_boxes = data_dict['gt_boxes'] if 'gt_boxes' in data_dict else None

        enable = np.random.choice([False, True], replace=False, p=[1 - config['PROBABILITY'], config['PROBABILITY']])
        noise_translate_std = config['NOISE_TRANSLATE_STD'] if enable else [0.0, 0.0, 0.0]
        if not isinstance(noise_translate_std, list):
            noise_translate_std = [noise_translate_std, noise_translate_std, noise_translate_std]

        noise_translate = np.array([
            np.random.normal(0, noise_translate_std[0], 1),
            np.random.normal(0, noise_translate_std[1], 1),
            np.random.normal(0, noise_translate_std[2], 1),
        ]).T if 'random_world_translation' not in data_dict['transformation_3d_list'] \
        else data_dict['transformation_3d_params']['random_world_translation']
        points[:, :3] += noise_translate
        if 'gt_boxes' in data_dict:
            gt_boxes[:, :3] += noise_translate
        if 'random_world_translation' not in data_dict['transformation_3d_list']:
            data_dict['transformation_3d_list'].append('random_world_translation')
            data_dict['transformation_3d_params']['random_world_translation'] = noise_translate
        data_dict['points'] = points
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def photo_metric_distortion(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.photo_metric_distortion, config=config)
        img = data_dict['image']
        assert img.dtype == np.float32, \
            'PhotoMetricDistortion needs the input image of dtype ' \
            'np.float32, please set "to_float32=True" in ' \
            '"LoadImageFromFile" pipeline'
        
        brightness_delta = config['BRIGHTNESS_DELTA']
        contrast_lower, contrast_upper = config['CONTRAST_RANGE']
        saturation_lower, saturation_upper = config['SATURATION_RANGE']
        hue_delta = config['HUE_DELTA']

        # random brightness
        if np.random.randint(2):
            delta = np.random.uniform(-brightness_delta, brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            if np.random.randint(2):
                alpha = np.random.uniform(contrast_lower, contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # random saturation
        if np.random.randint(2):
            img[..., 1] *= np.random.uniform(saturation_lower, saturation_upper)

        # random hue
        if np.random.randint(2):
            img[..., 0] += np.random.uniform(-hue_delta, hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        # random contrast
        if mode == 0:
            if np.random.randint(2):
                alpha = np.random.uniform(contrast_lower, contrast_upper)
                img *= alpha

        # randomly swap channels
        if np.random.randint(2):
            img = img[..., np.random.permutation(3)]

        data_dict['image'] = img
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        data_dict['transformation_3d_list'] = data_dict.get('transformation_3d_list', [])
        data_dict['transformation_3d_params'] = data_dict.get('transformation_3d_params', {})

        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
                data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
            )
        if 'calib' in data_dict:
            data_dict.pop('calib')
        # if 'road_plane' in data_dict:
        #     data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            if 'gt_boxes2d' in data_dict:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][gt_boxes_mask]

            data_dict.pop('gt_boxes_mask')
        return data_dict