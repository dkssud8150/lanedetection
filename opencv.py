import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import torchvision.models as models
import torchsummary

import sklearn
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm.notebook import tqdm

import cv2, argparse, random

# import sys
# sys.path('./yolov5')

# from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords
# from models.experimental import attempt_load
# from models.common import DetectMultiBackend
# from utils.torch_utils import select_device



device ="cpu"#= select_device('')
print("device : ", device)

random.seed(34)

CUDA_LAUNCH_BLOCKING = 1

os.makedirs('./weights', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--img', default='./data/road_video1.mp4')
parser.add_argument('--weights',default='./yolov5/yolov5s.onnx')
parser.add_argument('--yolofile',default='./yolov5/yolov5s.pt')

args = parser.parse_args()


weight = args.weights
model_param = args.yolofile

threshold = 0.7

def start():
    global weight

    detector = cv2.dnn.readNet(weight)
    layers = detector.getLayerNames()
    output_layer = detector.getUnconnectedOutLayers()[0]

    print(output_layer)

    cap = cv2.VideoCapture(args.img if args.img else 0)

    while True:
        ok, frame = cap.read()

        if not ok: break

        dnnframe = frame.copy()

        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(dnnframe, 1.0, (640,640),
                                    [104,117,123],
                                    swapRB=True, crop=False)
        
        detector.setInput(blob)
        detections = detector.forward()
        print("detection shape : {}\nshape0 : {}\nshape00 : {}\nshape000 : {}".format(
                detections.shape, detections[0].shape,detections[0][0].shape,detections[0][:10],
            )
        )
        
        colors = (random.randint(0,225),random.randint(0,225),random.randint(0,225))

        boxes=[]
        #for i in range(detections.shape[2]):

        
        cv2.rectangle(frame, (0,0), (255,255), colors, 3)

        cv2.imshow("src", frame)
        if cv2.waitKey(1) == 27: break









if __name__ == '__main__':
    start()