from torchvision import transforms
import torchvision

from model.hopenet import Hopenet
import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable

__all__ = ['head_pose_estimation', 'module_init']

def crop_image(cv2_frame, x1, y1, x2, y2, w, h, ad, img_w, img_h):
    xw1 = max(int(x1 - ad * w), 0)
    yw1 = max(int(y1 - ad * h), 0)
    xw2 = min(int(x2 + ad * w), img_w - 1)
    yw2 = min(int(y2 + ad * h), img_h - 1)

    # Crop image
    img = cv2_frame[yw1:yw2 + 1, xw1:xw2 + 1, :]
    return img, xw1, yw1, xw2, yw2


def img_transform(img, transformations):
    img = transformations(img)
    img_shape = img.size()
    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
    img = Variable(img).cuda()
    return img


def module_init(cfg):
    # ResNet50 structure
    model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # Load snapshot
    gpu = cfg.HEAD_POSE.GPU_ID
    pretrained_path = cfg.HEAD_POSE.PRETRAINED
    saved_state_dict = torch.load(pretrained_path)
    model.load_state_dict(saved_state_dict)

    model.cuda(gpu)
    model.eval()

    return model


def head_pose_estimation(cv2_frame, mtcnn, model, transformations, softmax, idx_tensor):
    detected = mtcnn.detect_faces(cv2_frame)
    img_h, img_w, _ = np.shape(cv2_frame)
    ad = 0.2
    predictions_arr = []
    bounding_box_arr = []
    face_keypoints_arr = []
    w_arr = []

    for i, d in enumerate(detected):
        if d['confidence'] > 0.95:
            x1, y1, w, h = d['box']
            x2 = x1+w
            y2 = y1+h

            img, xw1, yw1, xw2, yw2 = crop_image(cv2_frame, x1, y1, x2, y2, w, h, ad, img_w, img_h)

            img = Image.fromarray(img)

            # Transform
            img = img_transform(img, transformations)

            yaw, pitch, roll = model(img)

            yaw_predicted = softmax(yaw)
            pitch_predicted = softmax(pitch)
            roll_predicted = softmax(roll)
            # Get continuous predictions in degrees.
            yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
            roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

            bounding_box = d['box']
            predictions = [yaw_predicted, pitch_predicted, roll_predicted]
            face_keypoints = [xw1, yw1, xw2, yw2]

            predictions_arr.append(predictions)
            bounding_box_arr.append(bounding_box)
            face_keypoints_arr.append(face_keypoints)
            w_arr.append(w)

    return predictions_arr, bounding_box_arr, face_keypoints_arr, w_arr
