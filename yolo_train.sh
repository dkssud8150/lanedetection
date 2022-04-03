mkdir weights/ --parents

cd yolov5

python train.py --data lane.yaml --weights yolov5s.pt --img 124 --epochs 10 --batch 2 --project ../weights/ --name model_yolov5_lane --device 0 --exist-ok



# convert pt to onnx
# https://github.com/ultralytics/yolov5/blob/master/export.py
#pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime openvino-dev tensorflow-cpu # for cpu
#pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime-gpu openvino-dev tensorflow # for gpu
#python export.py --weights yolov5s.pt --include onnx