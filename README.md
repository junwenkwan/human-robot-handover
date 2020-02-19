# robot-handover
## Requirements
* detectron2

*Follow [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) to install detectron2

* mtcnn
```bash
pip3 install mtcnn
```

## Pretrained weights
The pretrained weights for the object detection network can be found in the [Google Drive](https://drive.google.com/file/d/1gx6beqSOwh0mTkATEDe3tdKdya-vPZSZ/view?usp=sharing) 

## Running inference on pretrained keypoint detection weights
```
sudo python3 run_inference.py --config-file configs/keypoint-RCNN-FPN.yaml \
  --video-input [VIDEO_INPUT] \
  --confidence-threshold [CONFIDENCE_TRHESHOLD]
  --output OUTPUT
```

