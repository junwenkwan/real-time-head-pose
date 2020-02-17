import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

import hopenet, utils
from mtcnn.mtcnn import MTCNN
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained', help='Path of model snapshot.',
          default='', type=str)
    parser.add_argument('--video', dest='video_path', help='Path of video')
    parser.add_argument('--output_string', dest='output_string', help='String appended to output file')
    parser.add_argument('--fps', dest='fps', help='Frames per second of source video', type=float, default=30.)
    args = parser.parse_args()
    return args


def crop_image(cv2_frame, x1, x2, w, h, ad):
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
    img = Variable(img).cuda(gpu)
    return img


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    gpu = args.gpu_id
    snapshot_path = args.pretrained
    out_dir = 'output'
    video_path = args.video_path

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(args.video_path):
        sys.exit('Video does not exist')

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # Dlib face detection model
    mtcnn = MTCNN()
    ad = 0.2

    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    print('Loaded pretrained model successfully')

    transformations = transforms.Compose([transforms.Resize(224), \
                                    transforms.CenterCrop(224), transforms.ToTensor(), \
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.cuda(gpu)

    # Test the Model
    model.eval() 
    print('Evaluation mode')

    softmax = nn.Softmax(dim=1).cuda(gpu)
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    video = cv2.VideoCapture(video_path)
    video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output/%s.avi' % args.output_string, fourcc, args.fps, (width, height))

    frame_num = 1

    while frame_num <= video_length:
        print('frame_num',frame_num)

        ret,frame = video.read()
        if ret == False:
            break

        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        dets = mtcnn.detect_faces(cv2_frame)
        img_h, img_w, _ = np.shape(frame)

        for i, d in enumerate(dets):
            if d['confidence'] > 0.95:
                x1, y1, w, h = d['box']
                x2 = x1+w
                y2 = y1+h

                # Crop image
                img, xw1, yw1, xw2, yw2 = crop_image(cv2_frame, x1, x2, w, h, ad)

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

                # Print new frame with cube and axis
                # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (xw1 + xw2) / 2, (yw1 + yw2) / 2, size = w)
                utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (xw1 + xw2) / 2, tdy= (yw1 + yw2) / 2, size = w/2)

                bounding_box = d['box']

                cv2.rectangle(frame,(bounding_box[0], bounding_box[1]), \
                              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]), \
                              (140,255,255), 3)

        out.write(frame)
        frame_num += 1

    out.release()
    video.release()
