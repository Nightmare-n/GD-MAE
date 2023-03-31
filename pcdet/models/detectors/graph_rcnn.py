from .detector3d_template import Detector3DTemplate


class GraphRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, logger):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, logger=logger)
        self.module_list = self.build_networks()
        if self.model_cfg.get('FREEZE_LAYERS', None) is not None:
            self.freeze(self.model_cfg.FREEZE_LAYERS)

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict, tb_dict = {}, {}
        loss = 0
        if self.model_cfg.get('FREEZE_LAYERS', None) is None:
            if self.dense_head is not None:
                loss_rpn, tb_dict = self.dense_head.get_loss(tb_dict)
            else:
                loss_rpn, tb_dict = self.point_head.get_loss(tb_dict)
            loss += loss_rpn

        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        loss += loss_rcnn
        
        return loss, tb_dict, disp_dict
