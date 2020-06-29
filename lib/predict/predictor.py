# This code is modified from detectron2 by facebook research
# Link to github repo: https://github.com/facebookresearch/detectron2

import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
from torchvision import transforms
import torch.nn as nn
import numpy as np

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.structures.instances import Instances

from lib.headpose import module_init, head_pose_estimation
from lib.headpose.utils import draw_axis, plot_pose_cube
from mtcnn.mtcnn import MTCNN
import json

class VisualizationDemo(object):
    def __init__(self, cfg_object, cfg_keypoint, instance_mode=ColorMode.IMAGE):
        """
        Initialize all required modules
        """
        self.metadata_object = MetadataCatalog.get(
            "__unused"
        )

        self.metadata_keypoint = MetadataCatalog.get(
            cfg_keypoint.DATASETS.TEST[0] if len(cfg_keypoint.DATASETS.TEST) else "__unused"
        )

        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.predictor_object = DefaultPredictor(cfg_object)
        self.predictor_keypoint = DefaultPredictor(cfg_keypoint)

        self.head_pose_module = module_init(cfg_keypoint)
        self.mtcnn = MTCNN()
        self.transformations = transforms.Compose([transforms.Resize(224), \
                                        transforms.CenterCrop(224), transforms.ToTensor(), \
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.softmax = nn.Softmax(dim=1).cuda()

        idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(idx_tensor).cuda()
        self.data_json = {}
        self.data_json['object_detection'] = {}
        self.data_json['keypoint_detection'] = {}
        self.data_json['head_pose_estimation'] = {}
        self.frame_count = 0

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Process the input video
        """
        video_visualizer_object = VideoVisualizer(self.metadata_object, self.instance_mode)
        video_visualizer_keypoint = VideoVisualizer(self.metadata_keypoint, self.instance_mode)

        def process_predictions(frame, predictions_object, predictions_keypoint):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            blank_image = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)

            # object detection
            if "instances" in predictions_object:
                predictions_object = predictions_object["instances"].to(self.cpu_device)
                boxes_area = predictions_object.get('pred_boxes').area()

                if boxes_area.nelement() != 0:
                    max_val, max_idx = torch.max(boxes_area,dim=0)

                    pred_boxes_object = predictions_object.get('pred_boxes')[max_idx.item()]
                    scores_object = predictions_object.get('scores')[max_idx.item()]
                    pred_classes_object = predictions_object.get('pred_classes')[max_idx.item()]
                    
                    # create a detectron2 instance for visualisation
                    draw_instance_object = Instances([1280,720])
                    draw_instance_object.set('pred_boxes',pred_boxes_object)
                    draw_instance_object.set('scores', torch.unsqueeze(scores_object, dim=0))
                    draw_instance_object.set('pred_classes', torch.unsqueeze(pred_classes_object, dim=0))

                    # update json file
                    self.data_json['object_detection']['pred_boxes'] = predictions_object.get('pred_boxes')[max_idx.item()].tensor.numpy().tolist()
                    self.data_json['object_detection']['scores'] = predictions_object.get('scores')[max_idx.item()].numpy().tolist()
                    vis_frame = video_visualizer_object.draw_instance_predictions(blank_image, draw_instance_object)
                else:
                    self.data_json['object_detection']['pred_boxes'] = []
                    self.data_json['object_detection']['scores'] = []
                    vis_frame = video_visualizer_object.draw_instance_predictions(blank_image, predictions_object)

            # keypoint detection
            if "instances" in predictions_keypoint:
                predictions_keypoint = predictions_keypoint["instances"].to(self.cpu_device)
                boxes_area = predictions_keypoint.get('pred_boxes').area()

                if boxes_area.nelement() != 0:
                    max_val, max_idx = torch.max(boxes_area,dim=0)

                    pred_boxes_keypoint = predictions_keypoint.get('pred_boxes')[max_idx.item()]
                    scores_keypoint = predictions_keypoint.get('scores')[max_idx.item()]
                    pred_classes_keypoint = predictions_keypoint.get('pred_classes')[max_idx.item()]
                    pred_keypoints_keypoint = predictions_keypoint.get('pred_keypoints')[max_idx.item()]

                    # create a detectron2 instance for visualisation
                    draw_instance_keypoint = Instances([1280,720])
                    draw_instance_keypoint.set('pred_boxes', pred_boxes_keypoint)
                    draw_instance_keypoint.set('scores', torch.unsqueeze(scores_keypoint, dim=0))
                    draw_instance_keypoint.set('pred_classes', torch.unsqueeze(pred_classes_keypoint, dim=0))
                    draw_instance_keypoint.set('pred_keypoints', torch.unsqueeze(pred_keypoints_keypoint, dim=0))

                    # update json file
                    self.data_json['keypoint_detection']['pred_boxes'] = predictions_keypoint.get('pred_boxes')[max_idx.item()].tensor.numpy().tolist()
                    self.data_json['keypoint_detection']['scores'] = predictions_keypoint.get('scores')[max_idx.item()].numpy().tolist()
                    self.data_json['keypoint_detection']['pred_keypoints'] = predictions_keypoint.get('pred_keypoints')[max_idx.item()].numpy().tolist()
                    vis_frame = video_visualizer_keypoint.draw_instance_predictions(vis_frame.get_image(), draw_instance_keypoint)
                else:
                    self.data_json['keypoint_detection']['pred_boxes'] = []
                    self.data_json['keypoint_detection']['scores'] = []
                    self.data_json['keypoint_detection']['pred_keypoints'] = []
                    vis_frame = video_visualizer_keypoint.draw_instance_predictions(vis_frame.get_image(), predictions_keypoint)

            # head pose estimation
            predictions, bounding_box, face_keypoints, w, face_area = head_pose_estimation(frame, self.mtcnn, self.head_pose_module, self.transformations, self.softmax, self.idx_tensor)
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)

            if len(face_area) != 0:
                max_val, max_idx = torch.max(torch.Tensor(face_area),dim=0)

                self.data_json['head_pose_estimation']['predictions'] = predictions[max_idx.item()]
                self.data_json['head_pose_estimation']['pred_boxes'] = bounding_box[max_idx.item()]

                plot_pose_cube(vis_frame, predictions[max_idx.item()][0], predictions[max_idx.item()][1], predictions[max_idx.item()][2], \
                                tdx = (face_keypoints[max_idx.item()][0] + face_keypoints[max_idx.item()][2]) / 2, \
                                tdy= (face_keypoints[max_idx.item()][1] + face_keypoints[max_idx.item()][3]) / 2, \
                                size = w[max_idx.item()])

            data_json = self.data_json
            self.data_json['frame'] = self.frame_count
            self.frame_count += 1
            return vis_frame, data_json

        frame_gen = self._frame_from_video(video)

        for frame in frame_gen:

            yield process_predictions(frame, self.predictor_object(frame), self.predictor_keypoint(frame))
