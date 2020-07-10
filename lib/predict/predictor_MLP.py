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

from lib.headpose import module_init, head_pose_estimation
from lib.headpose.utils import draw_axis, plot_pose_cube
from mtcnn.mtcnn import MTCNN
from model.mlp import MLP
from torch.autograd import Variable
import json

class VisualizationDemoMLP(object):
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

        self.mlp_model = MLP(input_size=26, output_size=1).cuda()
        self.mlp_model.load_state_dict(torch.load(cfg_keypoint.MLP.PRETRAINED))
        self.mlp_model.eval()

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

        def get_parameters(annos):
            if annos["object_detection"]["pred_boxes"]:
                temp = annos["object_detection"]["pred_boxes"][0]
                obj_det = [1]
                temp = np.asarray(temp)
                temp = temp.flatten()

                key_det = annos["keypoint_detection"]["pred_keypoints"][0]
                key_det = np.asarray(key_det)
                # select relevant keypoints
                key_det = key_det[0:11, 0:2]
                # localization
                key_det = np.subtract(key_det, temp[0:2])
                key_det = key_det.flatten()

            else:
                obj_det = [-999]
                obj_det = np.asarray(obj_det)

                key_det = annos["keypoint_detection"]["pred_keypoints"][0]
                key_det = np.asarray(key_det)
                key_det = key_det[0:11, 0:2]
                key_det = key_det.flatten()

            if annos["head_pose_estimation"]["predictions"]:
                hp_est = annos["head_pose_estimation"]["predictions"][0]
                hp_est = np.asarray(hp_est)
            else:
                hp_est = np.asarray([-999, -999, -999])

            anno_list = np.concatenate((obj_det, key_det, hp_est))
            return anno_list

        def process_predictions(frame, predictions_object, predictions_keypoint):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            blank_image = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
            
            # object detection
            if "instances" in predictions_object:
                predictions_object = predictions_object["instances"].to(self.cpu_device)
                self.data_json['object_detection']['pred_boxes'] = predictions_object.get('pred_boxes').tensor.numpy().tolist()
                self.data_json['object_detection']['scores'] = predictions_object.get('scores').numpy().tolist()
                vis_frame = video_visualizer_object.draw_instance_predictions(frame, predictions_object)

            # keypoint detection
            if "instances" in predictions_keypoint:
                predictions_keypoint = predictions_keypoint["instances"].to(self.cpu_device)
                self.data_json['keypoint_detection']['pred_boxes'] = predictions_keypoint.get('pred_boxes').tensor.numpy().tolist()
                self.data_json['keypoint_detection']['scores'] = predictions_keypoint.get('scores').numpy().tolist()
                self.data_json['keypoint_detection']['pred_keypoints'] = predictions_keypoint.get('pred_keypoints').numpy().tolist()
                vis_frame = video_visualizer_keypoint.draw_instance_predictions(vis_frame.get_image(), predictions_keypoint)

            # head pose estimation
            predictions, bounding_box, face_keypoints, w, face_area = head_pose_estimation(frame, self.mtcnn, self.head_pose_module, self.transformations, self.softmax, self.idx_tensor)
            self.data_json['head_pose_estimation']['predictions'] = predictions
            self.data_json['head_pose_estimation']['pred_boxes'] = bounding_box

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)

            for i in range(len(predictions)):
                plot_pose_cube(vis_frame, predictions[i][0], predictions[i][1], predictions[i][2], \
                                tdx = (face_keypoints[i][0] + face_keypoints[i][2]) / 2, \
                                tdy= (face_keypoints[i][1] + face_keypoints[i][3]) / 2, \
                                size = w[i])

            data_json = self.data_json
            self.data_json['frame'] = self.frame_count
            self.frame_count += 1

            inputs_MLP = get_parameters(self.data_json)
            inputs_MLP = Variable(torch.from_numpy(inputs_MLP)).float().cuda()
            outputs_MLP = self.mlp_model(inputs_MLP)
            predicted_MLP = (outputs_MLP >= 0.5)

            if predicted_MLP.item():
                cv2.putText(vis_frame,str(predicted_MLP.item()), (10,700), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 10)
            else:
                cv2.putText(vis_frame,str(predicted_MLP.item()), (10,700), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 10)


            return vis_frame, data_json

        frame_gen = self._frame_from_video(video)

        for frame in frame_gen:

            yield process_predictions(frame, self.predictor_object(frame), self.predictor_keypoint(frame))
