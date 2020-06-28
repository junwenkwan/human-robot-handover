# Human-Robot Handover
<div align="center">
<img src="./teaser/demo.gif" width="700"/><br>
</div>

## Requirements
* [detectron2](https://github.com/facebookresearch/detectron2) (Follow [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) to install detectron2)
* [mtcnn](https://github.com/ipazc/mtcnn)

## Pretrained models
Download the pretrained models for [object detection](https://drive.google.com/file/d/1gx6beqSOwh0mTkATEDe3tdKdya-vPZSZ/view?usp=sharing), [head pose estimation](https://drive.google.com/file/d/1kY2nfpnFsows14TLKTOd-8PYftOAeomh/view?usp=sharing) and [MLP](https://drive.google.com/file/d/1D192ELxRDVeyuI81r86G2PgVBLQNdhZk/view?usp=sharing). Place them in ```./pretrained-weights```.

## Custom dataset
The handover dataset used to obtain the pretrained weights can be found [here](https://drive.google.com/open?id=1NYRohLw1iWMH33qNJtft1tMSjXfnNyDk).


## Run system for training
```bash
python3 run_inference.py \
        --cfg-keypoint ./configs/keypoint_rcnn_R_101_FPN_3x.yaml \
        --cfg-object ./configs/object_faster_rcnn_R_101_FPN_3x.yaml \
        --obj-weights ./pretrained-weights/Apple_Faster_RCNN_R_101_FPN_3x.pth \
        --video-input [VIDEO_INPUT] \
        --output [OUTPUT] \
        --train
```

## Generate a JSON file for MLP training
```bash
python3 utils/json_utils.py --json-path [JSON_FOLDER] --csv-path [classes.csv] \
                      --output-json sample_robot.json
```

## Train MLP network
```bash
python3 train_MLP_localize.py --json-path [JSON_FILE] --weights-path [PATH_TO_WEIGHTS]
```

## Deployment
```bash
python3 run_inference.py \
        --cfg-keypoint ./configs/keypoint_rcnn_R_101_FPN_3x.yaml \
        --cfg-object ./configs/object_faster_rcnn_R_101_FPN_3x.yaml \
        --obj-weights ./pretrained-weights/Apple_Faster_RCNN_R_101_FPN_3x.pth \
        --video-input [VIDEO_INPUT] \
        --output [OUTPUT]
