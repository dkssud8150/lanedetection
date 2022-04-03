import cv2
import argparse
import torch, torchvision
from torchvision import datasets, models, transforms
import onnx


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]

    # blobFromImage(iamge, scalefactor = 1, size, mean, swapRB=false,crop=false,ddepth=CV_32F)
    # 4차원 이미지 blob로 만드는 함수로, 
    # image : 첫번째 인자로 리사이즈하거나 자르고
    # scalefactor : scale 변환
    # size : 출력 사이즈
    # mean : 에 의해 나누고
    # swapRB : 이름 그대로 Rcolor와 Bcolor 자리를 바꿈
    # crop : 이미지를 자를지말지
    # 출력 blob의 깊이
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (224, 224), [104, 117, 123], True, False)

    # setInput(blob, name = "", scalefactor=1.0,mean=scalar())
    # blob : 입력 영상
    # name : 입력 레이어의 이름
    # scalefactor : 정규화 스케일 지저
    # mean : 평균화 수치
    net.setInput(blob)

    # forward 계산
    detections=net.forward()
    print("detection : ",detections)

    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


parser = argparse.ArgumentParser()
parser.add_argument('--img')

args = parser.parse_args()

torchmodel = "./weight/model_best_epoch.pt"
onnxmodel = './weight/animalface_detector.onnx'
pbmodel = "./weight/opencv_face_detector_uint8.pb"
pbparam = "./weight/opencv_face_detector.pbtxt"


detector = cv2.dnn.readNet(pbmodel,pbparam)

video = cv2.VideoCapture(args.img if args.img else 0)

while True:
    hasframe, frame = video.read()

    if not hasframe:
        break

    resultImg, faceBoxes = highlightFace(detector, frame)
    cv2.imshow("Detection", resultImg)

    if cv2.waitKey(10) == 27: break
    
'''
https://justkode.kr/deep-learning/pytorch-save
https://velog.io/@ailab/PyTorch-%EB%AA%A8%EB%8D%B8%EC%9D%84-ONNX%EB%A5%BC-%ED%86%B5%ED%95%B4-OpenCV%EC%97%90%EC%84%9C-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0
https://ys-cs17.tistory.com/24
https://docs.opencv.org/4.x/dc/d70/pytorch_cls_tutorial_dnn_conversion.html
https://tutorials.pytorch.kr/advanced/super_resolution_with_onnxruntime.html
'''