from .detector3d_template import Detector3DTemplate
from .pointpillar import PointPillar
from .second_net import SECONDNet
from .centerpoint import CenterPoint
from .ssd3d import SSD3D
from .graph_rcnn import GraphRCNN
from .gd_mae import GDMAE


__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PointPillar': PointPillar,
    'CenterPoint': CenterPoint,
    'GraphRCNN': GraphRCNN,
    'SSD3D': SSD3D,
    'GraphRCNN': GraphRCNN,
    'GDMAE': GDMAE
}


def build_detector(model_cfg, num_class, dataset, logger):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset, logger=logger
    )

    return model
