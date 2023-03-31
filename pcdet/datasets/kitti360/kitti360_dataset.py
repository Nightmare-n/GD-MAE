import numpy as np
from ..dataset import DatasetTemplate
import glob
import os


class Kitti360Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

        self.kitti_infos = []
        self.include_kitti_data()

    def include_kitti_data(self):
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')
        
        self.kitti_infos = list(self.client.list_dir_or_file(
                os.path.join(self.root_path, 'data_3d_raw'),
                list_dir=False, recursive=True, suffix='.bin'
        ))

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(self.kitti_infos)))

    def get_lidar(self, lidar_file):
        return self.client.load_to_numpy(str(self.root_path / 'data_3d_raw' / lidar_file), dtype=np.float32).reshape(-1, 4)

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        return len(self.kitti_infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        lidar_path = self.kitti_infos[index]

        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])
        
        path_split = str(lidar_path).split('/')
        input_dict = {
            'frame_id': path_split[-4] + '_' + path_split[-1][:-4],
        }

        if "points" in get_item_list:
            points = self.get_lidar(lidar_path)
            input_dict['points'] = points

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict
