# robot-handover
## Requirements
* detectron2

Install Detectron 2 following [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

## Pretrained weights
The pretrained weights for the object detection network can be found in the [Google Drive](https://drive.google.com/file/d/1gx6beqSOwh0mTkATEDe3tdKdya-vPZSZ/view?usp=sharing) 

## Running inference on pretrained keypoint detection weights
```
sudo python3 demo/demo.py --config-file configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml \
  --video-input VIDEO_INPUT \
  --confidence-threshold CONFIDENCE_TRHESHOLD
  --output OUTPUT
  --opts MODEL.WEIGHTS detectron2://COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x/138363331/model_final_997cc7.pkl
```
