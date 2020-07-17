from detectron2.config import CfgNode as CN


def add_custom_config(cfg):
    """
    Add config for head pose estimation
    """
    _C = cfg

    _C.HEAD_POSE = CN()
    _C.HEAD_POSE.PRETRAINED = "./pretrained-weights/head-pose-pretrained.pkl"
    _C.HEAD_POSE.GPU_ID = 0

    _C.MLP = CN()
    _C.MLP.PRETRAINED = "./pretrained-weights/MLP_localized.pth"
    _C.MLP.PRETRAINED_NONLOCALIZED = "./pretrained-weights/MLP_nonlocalized.pth"
