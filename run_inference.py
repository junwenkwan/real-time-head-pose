import sys, os, argparse

import cv2

import torch.backends.cudnn as cudnn
import torchvision

import utils
from lib.headpose import module_init, head_pose_estimation

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained', help='Path of model snapshot.',
          default='', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    model, mtcnn, transformations, softmax, idx_tensor = module_init(args)

    print('Loaded pretrained model successfully')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        # Convert to numpy.ndarray
        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        predictions, bounding_box, face_keypoints, w = head_pose_estimation(cv2_frame, mtcnn, model, transformations, softmax, idx_tensor)

        # Print new frame with cube and axis
        for i in range(len(predictions)):

            utils.draw_axis(frame, predictions[i][0], predictions[i][1], predictions[i][2], \
                            tdx = (face_keypoints[i][0] + face_keypoints[i][2]) / 2, tdy= (face_keypoints[i][1] + face_keypoints[i][3]) / 2, size = w[i]/2)

            cv2.rectangle(frame,(bounding_box[i][0], bounding_box[i][1]), \
                          (bounding_box[i][0]+bounding_box[i][2], bounding_box[i][1] + bounding_box[i][3]), \
                          (140,255,255), 3)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow("INFERENCE", frame)

    cap.release()
    cv2.destroyWindow("INFERENCE")
