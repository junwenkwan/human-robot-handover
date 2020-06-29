# This code is modified from detectron2 by facebook research
# Link to github repo: https://github.com/facebookresearch/detectron2

import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from lib.predict.predictor import VisualizationDemo
from lib.predict.predictor_MLP import VisualizationDemoMLP
from utils.add_custom_config import *
import json
import copy

# constants
WINDOW_NAME = "Human-to-robots handovers"


def setup_cfg(args):
    # load default config from file and command-line arguments
    cfg_object = get_cfg()
    cfg_keypoint = get_cfg()
    # add additional config for head pose estimation
    add_custom_config(cfg_keypoint)

    cfg_object.merge_from_file(args.cfg_object)
    cfg_keypoint.merge_from_file(args.cfg_keypoint)
    
    # set threshold for object detection
    cfg_object.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg_object.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # set threshold for keypoint detection
    cfg_keypoint.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold

    cfg_object.MODEL.WEIGHTS = args.obj_weights
    cfg_keypoint.MODEL.WEIGHTS = args.keypoint_weights

    cfg_object.freeze()
    cfg_keypoint.freeze()

    return cfg_object, cfg_keypoint


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--cfg-keypoint",
        default="./configs/keypoint_rcnn_R_101_FPN_3x.yaml",
        metavar="FILE",
        help="path to keypoint config file",
    )
    parser.add_argument(
        "--cfg-object",
        default="./configs/object_faster_rcnn_R_101_FPN_3x.yaml",
        metavar="FILE",
        help="path to object detection config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")

    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--out-json",
        default="./default.json",
        metavar="FILE",
        help="A file or directory to save output json. "
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.85,
        help="Minimum score for instance predictions to be shown",
    )

    parser.add_argument(
        "--obj-weights",
        type=str,
        default="./pretrained-weights/Apple_Faster_RCNN_R_101_FPN_3x.pth",
        help="Path to the object detection weights",
    )

    parser.add_argument(
        "--keypoint-weights",
        type=str,
        default="detectron2://COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x/138363331/model_final_997cc7.pkl",
        help="Path to the keypoint detection weights",
    )

    parser.add_argument("--train", action="store_true", help="Run training.")

    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg_object, cfg_keypoint = setup_cfg(args)
    
    # database in json
    database_json = {}
    database_json['annotation'] = {}
    database_arr = []

    if args.train:
        demo = VisualizationDemo(cfg_object, cfg_keypoint)
    else:
        demo = VisualizationDemoMLP(cfg_object, cfg_keypoint)
    frame = 0

    if args.webcam:
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,

                fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame, data_json in tqdm.tqdm(demo.run_on_video(video), total=num_frames):

            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit

            database_arr.append(copy.deepcopy(data_json))
        database_json["annotation"] = database_arr

        with open(args.out_json, 'w') as json_file:
            json.dump(database_json, json_file)

        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
