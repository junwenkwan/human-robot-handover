from detectron2.config import CfgNode as CN


def add_config(cfg):
    """
    Add config for densepose head.
    """
    _C = cfg

    _C.HEAD_POSE = CN()
    _C.HEAD_POSE.PRETRAINED = "./pretrained/head-pose-pretrained.pkl"
    _C.HEAD_POSE.GPU_ID = 0
