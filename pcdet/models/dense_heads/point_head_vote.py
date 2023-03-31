import torch
import torch.nn as nn
import torch.nn.functional as F
from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...utils import box_coder_utils, box_utils, loss_utils
from .point_head_template import PointHeadTemplate


class PointHeadVote(PointHeadTemplate):
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )

        self.vote_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.VOTE_CONFIG.MLPS,
            input_channels=input_channels,
            output_channels=3,
            linear=False
        )

        mlps = self.model_cfg.VOTE_SA_CONFIG.MLPS.copy()
        channel_out = 0
        for idx in range(mlps.__len__()):
            mlps[idx] = [input_channels] + mlps[idx]
            channel_out += mlps[idx][-1]
        self.vote_SA_module = pointnet2_modules.PointnetSAModuleFSMSG(
            radii=self.model_cfg.VOTE_SA_CONFIG.RADIUS,
            nsamples=self.model_cfg.VOTE_SA_CONFIG.NSAMPLE,
            mlps=mlps,
            use_xyz=True
        )

        self.shared_conv = self.make_fc_layers(
            fc_cfg=self.model_cfg.SHARED_FC,
            input_channels=channel_out,
            linear=True
        )
        channel_out = self.model_cfg.SHARED_FC[-1]
        self.cls_conv = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=channel_out,
            output_channels=num_class,
            linear=True
        )
        self.box_conv = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=channel_out,
            output_channels=self.box_coder.code_size,
            linear=True
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def build_losses(self, losses_cfg):
        cls_loss_type = 'WeightedBinaryCrossEntropyLoss' \
            if losses_cfg.CLS_LOSS.startswith('WeightedBinaryCrossEntropyLoss') else losses_cfg.CLS_LOSS
        self.cls_loss_func = getattr(loss_utils, cls_loss_type)(
            **losses_cfg.get('CLS_LOSS_CONFIG', {})
        )

        reg_loss_type = losses_cfg.REG_LOSS
        self.reg_loss_func = getattr(loss_utils, reg_loss_type)(
            code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None),
            **losses_cfg.get('REG_LOSS_CONFIG', {})
        )

        aux_cls_loss_type = losses_cfg.get('AUX_CLS_LOSS', None)
        if aux_cls_loss_type is not None:
            self.aux_cls_loss_func = getattr(loss_utils, aux_cls_loss_type)(
                **losses_cfg.get('AUX_CLS_LOSS_CONFIG', {})
            )

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        gt_boxes = input_dict['gt_boxes']
  
        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        
        central_radius = self.model_cfg.TARGET_CONFIG.get('GT_CENTRAL_RADIUS', 2.0)
        vote_targets_dict = self.assign_stack_targets(
            points=input_dict['votes'], gt_boxes=gt_boxes, 
            set_ignore_flag=False, use_ball_constraint=True,
            ret_part_labels=False, ret_box_labels=True, central_radius=central_radius
        )

        seed_targets_dict = {
            'seed_cls_labels_list': [],
            'gt_box_of_fg_seeds_list': []
        }
        for i, seeds in enumerate(input_dict['seeds_list']):
            cur_seed_targets_dict = self.assign_stack_targets(
                points=seeds, gt_boxes=extend_gt_boxes,
                set_ignore_flag=False, use_ball_constraint=False,
                ret_part_labels=False, ret_box_labels=False
            )
            seed_targets_dict['seed_cls_labels_list'].append(cur_seed_targets_dict['point_cls_labels'])
            seed_targets_dict['gt_box_of_fg_seeds_list'].append(cur_seed_targets_dict['gt_box_of_fg_points'])

        aux_points_targets_dict = {
            'aux_points_cls_labels_list': [],
            'gt_box_idx_of_fg_aux_points_list': []
        }
        aux_extra_width = self.model_cfg.TARGET_CONFIG.get('AUX_GT_EXTRA_WIDTH', None)
        if aux_extra_width is not None:
            extend_gt_boxes = box_utils.enlarge_box3d(
                gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=aux_extra_width
            ).view(batch_size, -1, gt_boxes.shape[-1])
        for i, pts in enumerate(input_dict['aux_points_list']):
            cur_targets_dict = self.assign_stack_targets(
                points=pts, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                set_ignore_flag=True, use_ball_constraint=False,
                ret_part_labels=False, ret_box_labels=False
            )
            aux_points_targets_dict['aux_points_cls_labels_list'].append(cur_targets_dict['point_cls_labels'])
            aux_points_targets_dict['gt_box_idx_of_fg_aux_points_list'].append(cur_targets_dict['gt_box_idx_of_fg_points'])

        targets_dict = {
            'vote_cls_labels': vote_targets_dict['point_cls_labels'],
            'vote_box_labels': vote_targets_dict['point_box_labels'],
            'gt_box_of_fg_votes': vote_targets_dict['gt_box_of_fg_points'],
            'seed_cls_labels_list': seed_targets_dict['seed_cls_labels_list'],
            'gt_box_of_fg_seeds_list': seed_targets_dict['gt_box_of_fg_seeds_list'],
            'aux_points_cls_labels_list': aux_points_targets_dict['aux_points_cls_labels_list'],
            'gt_box_idx_of_fg_aux_points_list': aux_points_targets_dict['gt_box_idx_of_fg_aux_points_list']
        }
        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        aux_cls_loss, tb_dict = self.get_aux_cls_loss(tb_dict)
        seed_reg_loss, tb_dict = self.get_seed_reg_loss(tb_dict)
        vote_cls_loss, tb_dict = self.get_vote_cls_loss(tb_dict)
        vote_reg_loss, tb_dict = self.get_vote_reg_loss(tb_dict)
        vote_corner_loss, tb_dict = self.get_vote_corner_loss(tb_dict)
        point_loss = aux_cls_loss + seed_reg_loss + vote_cls_loss + vote_reg_loss + vote_corner_loss
        return point_loss, tb_dict

    def get_aux_single_cls_loss(self, point_cls_labels, point_cls_preds, gt_box_idx_of_fg_points, index, tb_dict):
        positives = point_cls_labels > 0
        negatives = point_cls_labels == 0
        cls_weights = negatives * 1.0 + positives * 1.0
        pos_normalizer = positives.sum().float() if self.model_cfg.LOSS_CONFIG.AUX_CLS_POS_NORM else cls_weights.sum()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        num_class = 1
        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels > 0).unsqueeze(-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]

        cls_loss_src = self.aux_cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()

        point_loss_cls = point_loss_cls * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['aux_cls_weight_list'][index]
        tb_dict.update({
            f'aux_points_cls_loss_{index}': point_loss_cls.item(),
            f'aux_points_pos_num_{index}': int(positives.sum().item() / self.forward_ret_dict['batch_size'])
        })
        return point_loss_cls, tb_dict

    def get_aux_cls_loss(self, tb_dict):
        point_cls_labels_list = self.forward_ret_dict['aux_points_cls_labels_list']
        point_cls_preds_list = self.forward_ret_dict['aux_cls_preds_list']
        gt_box_idx_of_fg_points_list = self.forward_ret_dict['gt_box_idx_of_fg_aux_points_list']
        aux_cls_loss_list = []
        for i in range(len(point_cls_labels_list)):
            point_loss_cls, tb_dict = self.get_aux_single_cls_loss(
                point_cls_labels_list[i], 
                point_cls_preds_list[i], 
                gt_box_idx_of_fg_points_list[i],
                i,
                tb_dict 
            )
            aux_cls_loss_list.append(point_loss_cls)
        return sum(aux_cls_loss_list), tb_dict

    def get_seed_single_reg_loss(self, votes, seed_cls_labels, gt_box_of_fg_seeds, index, tb_dict=None):
        pos_mask = seed_cls_labels > 0
        seed_center_labels = gt_box_of_fg_seeds[:, 0:3]
        seed_center_loss = self.reg_loss_func(
            votes[pos_mask][:, 1:], seed_center_labels
        ).sum(dim=-1).mean()
        seed_center_loss = seed_center_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['seed_reg_weight_list'][index]

        tb_dict.update({
            f'seed_reg_loss_{index}': seed_center_loss.item(),
            f'seed_pos_num_{index}': int(pos_mask.sum().item() / self.forward_ret_dict['batch_size'])
        })
        return seed_center_loss, tb_dict

    def get_seed_reg_loss(self, tb_dict=None):
        seed_cls_labels_list = self.forward_ret_dict['seed_cls_labels_list']
        gt_box_of_fg_seeds_list = self.forward_ret_dict['gt_box_of_fg_seeds_list']
        votes_list = self.forward_ret_dict['votes_list']
        seed_center_loss_list = []
        for i in range(len(votes_list)):
            seed_center_loss, tb_dict = self.get_seed_single_reg_loss(
                votes_list[i],
                seed_cls_labels_list[i],
                gt_box_of_fg_seeds_list[i],
                i,
                tb_dict
            )
            seed_center_loss_list.append(seed_center_loss)
        return sum(seed_center_loss_list), tb_dict

    def get_vote_cls_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['vote_cls_labels']
        point_cls_preds = self.forward_ret_dict['vote_cls_preds']

        positives = point_cls_labels > 0
        negatives = point_cls_labels == 0
        cls_weights = negatives * 1.0 + positives * 1.0
        pos_normalizer = positives.float().sum() if self.model_cfg.LOSS_CONFIG.CLS_POS_NORM else cls_weights.sum()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]

        if 'WithCenterness' in self.model_cfg.LOSS_CONFIG.CLS_LOSS:
            votes = self.forward_ret_dict['votes'].detach()
            gt_box_of_fg_votes = self.forward_ret_dict['gt_box_of_fg_votes']
            pos_centerness = box_utils.generate_centerness_mask(votes[positives][:, 1:], gt_box_of_fg_votes)
            centerness_mask = positives.new_zeros(positives.shape).float()
            centerness_mask[positives] = pos_centerness
            one_hot_targets = one_hot_targets * centerness_mask.unsqueeze(-1)

        cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()

        point_loss_cls = point_loss_cls * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_cls_weight']
        tb_dict.update({
            'vote_cls_loss': point_loss_cls.item(),
            'vote_pos_num': int(positives.sum().item() / self.forward_ret_dict['batch_size'])
        })
        return point_loss_cls, tb_dict

    def get_vote_reg_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['vote_cls_labels'] > 0
        point_box_labels = self.forward_ret_dict['vote_box_labels']
        point_box_preds = self.forward_ret_dict['vote_box_preds']

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        xyzlwh_preds = point_box_preds[:, :6]
        xyzlwh_labels = point_box_labels[:, :6]
        point_loss_xyzlwh = self.reg_loss_func(xyzlwh_preds, xyzlwh_labels, reg_weights).sum()

        angle_bin_num = self.box_coder.angle_bin_num
        dir_cls_preds = point_box_preds[:, 6:6 + angle_bin_num]
        dir_cls_labels = point_box_labels[:, 6:6 + angle_bin_num]
        point_loss_dir_cls = F.cross_entropy(dir_cls_preds, dir_cls_labels.argmax(dim=-1), reduction='none')
        point_loss_dir_cls = (point_loss_dir_cls * reg_weights).sum()

        dir_res_preds = point_box_preds[:, 6 + angle_bin_num:6 + 2 * angle_bin_num]
        dir_res_labels = point_box_labels[:, 6 + angle_bin_num:6 + 2 * angle_bin_num]
        
        dir_res_preds = torch.sum(dir_res_preds * dir_cls_labels, dim=-1)
        dir_res_labels = torch.sum(dir_res_labels * dir_cls_labels, dim=-1)
        point_loss_dir_res = self.reg_loss_func(dir_res_preds, dir_res_labels, weights=reg_weights)
        point_loss_dir_res = point_loss_dir_res.sum()

        point_loss_velo = 0
        if hasattr(self.box_coder, 'pred_velo') and self.box_coder.pred_velo:
            point_loss_velo = self.reg_loss_func(
                point_box_preds[:, 6 + 2 * angle_bin_num:8 + 2 * angle_bin_num],
                point_box_labels[:, 6 + 2 * angle_bin_num:8 + 2 * angle_bin_num],
                reg_weights
            ).sum()
            tb_dict.update({
                'vote_reg_velo_loss': point_loss_velo.item()
            })

        point_loss_box = point_loss_xyzlwh + point_loss_dir_cls + point_loss_dir_res + point_loss_velo
        point_loss_box = point_loss_box * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_reg_weight']
        tb_dict.update({'vote_reg_loss': point_loss_box.item()})
        return point_loss_box, tb_dict

    def get_vote_corner_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['vote_cls_labels'] > 0
        gt_boxes = self.forward_ret_dict['gt_box_of_fg_votes']
        pred_boxes = self.forward_ret_dict['point_box_preds']
        pred_boxes = pred_boxes[pos_mask]
        loss_corner = loss_utils.get_corner_loss_lidar(
            pred_boxes[:, 0:7],
            gt_boxes[:, 0:7],
            p=self.model_cfg.LOSS_CONFIG.CORNER_LOSS_TYPE
        ).mean()
        loss_corner = loss_corner * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_corner_weight']
        tb_dict.update({'vote_corner_loss': loss_corner.item()})
        return loss_corner, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']
        range_start, range_end = self.model_cfg.VOTE_CONFIG.SAMPLE_RANGE
        seeds = point_coords[:, range_start: range_end, :].contiguous()  # (B, K, 4)
        seed_features = point_features[:, :, range_start: range_end].contiguous()  # (B, C, K)

        vote_offsets = self.vote_layers(seed_features).permute(0, 2, 1).contiguous()  # (B, K, 3)
        vote_xyz_range = vote_offsets.new_tensor(self.model_cfg.VOTE_CONFIG.VOTE_XYZ_RANGE).view(1, 1, 3)
        vote_offsets = torch.max(vote_offsets, -vote_xyz_range)
        vote_offsets = torch.min(vote_offsets, vote_xyz_range)
        votes = torch.cat([
            seeds[..., 0:1], seeds[..., 1:] + vote_offsets
        ], dim=-1)

        _, vote_features, _ = self.vote_SA_module(
            point_coords[..., 1:].contiguous(),
            point_features,
            new_xyz=votes[..., 1:].contiguous()
        )
        
        vote_features = vote_features.permute(0, 2, 1).contiguous().view(-1, vote_features.shape[1])
        votes = votes.view(-1, 4)
        seeds = seeds.view(-1, 4)

        vote_features = self.shared_conv(vote_features)
        vote_cls_preds = self.cls_conv(vote_features)
        vote_box_preds = self.box_conv(vote_features)

        ret_dict = {
            'vote_cls_preds': vote_cls_preds,
            'vote_box_preds': vote_box_preds,
            'votes': votes,
            'votes_list': [votes],
            'seeds_list': [seeds],
            'aux_points_list': batch_dict['aux_points_list'],
            'aux_cls_preds_list': batch_dict['aux_cls_preds_list'],
            'batch_size': batch_dict['batch_size']
        }

        batch_dict.update({
            'votes_list': ret_dict['votes_list'],
            'seeds_list': ret_dict['seeds_list'],
            'votes': ret_dict['votes']
        })

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training or \
                self.model_cfg.LOSS_CONFIG.PREDICT_BOXES:

            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['votes'][:, 1:4],
                point_cls_preds=vote_cls_preds, point_box_preds=vote_box_preds
            )

            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = batch_dict['votes'][:, 0].contiguous()
            batch_dict['cls_preds_normalized'] = False

            ret_dict['point_box_preds'] = point_box_preds

        self.forward_ret_dict = ret_dict

        return batch_dict
