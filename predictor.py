# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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

class VisualizationDemo(object):
    def __init__(self, cfg_object, cfg_keypoint, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
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

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)


        if "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)
            vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer_object = VideoVisualizer(self.metadata_object, self.instance_mode)
        video_visualizer_keypoint = VideoVisualizer(self.metadata_keypoint, self.instance_mode)

        def process_predictions(frame, predictions_object, predictions_keypoint):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            blank_image = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)

            if "instances" in predictions_object:
                predictions_object = predictions_object["instances"].to(self.cpu_device)
                vis_frame = video_visualizer_object.draw_instance_predictions(blank_image, predictions_object)

            if "instances" in predictions_keypoint:
                predictions_keypoint = predictions_keypoint["instances"].to(self.cpu_device)
                vis_frame = video_visualizer_keypoint.draw_instance_predictions(vis_frame.get_image(), predictions_keypoint)

            # head pose estimation
            predictions, bounding_box, face_keypoints, w = head_pose_estimation(frame, self.mtcnn, self.head_pose_module, self.transformations, self.softmax, self.idx_tensor)

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)

            for i in range(len(predictions)):
                plot_pose_cube(vis_frame, predictions[i][0], predictions[i][1], predictions[i][2], \
                                tdx = (face_keypoints[i][0] + face_keypoints[i][2]) / 2, \
                                tdy= (face_keypoints[i][1] + face_keypoints[i][3]) / 2, \
                                size = w[i])
                # draw_axis(vis_frame, predictions[i][0], predictions[i][1], predictions[i][2], \
                #                 tdx = (face_keypoints[i][0] + face_keypoints[i][2]) / 2, \
                #                 tdy= (face_keypoints[i][1] + face_keypoints[i][3]) / 2, \
                #                 size = w[i])


            return vis_frame

        frame_gen = self._frame_from_video(video)

        for frame in frame_gen:
            yield process_predictions(frame, self.predictor_object(frame), self.predictor_keypoint(frame))
