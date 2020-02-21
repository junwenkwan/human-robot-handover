# Human-Robot Handover
## Requirements
* [detectron2](https://github.com/facebookresearch/detectron2) (Follow [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) to install detectron2)
* [mtcnn](https://github.com/ipazc/mtcnn)

## Pretrained models
Download the pretrained models for [object detection](https://drive.google.com/file/d/1gx6beqSOwh0mTkATEDe3tdKdya-vPZSZ/view?usp=sharing) and [head pose estimation](https://drive.google.com/file/d/1kY2nfpnFsows14TLKTOd-8PYftOAeomh/view?usp=sharing).

## Running inference
```bash
python3 run_inference.py \
        --cfg-keypoint ./configs/keypoint_rcnn_R_101_FPN_3x.yaml \
        --cfg-object ./configs/object_faster_rcnn_R_101_FPN_3x.yaml \
        --obj-weights ./pretrained-weights/Apple_Faster_RCNN_R_101_FPN_3x.pth \
        --video-input [VIDEO_INPUT] \
        --output [OUTPUT]
```


