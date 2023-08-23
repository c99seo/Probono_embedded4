# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import math
import time
import board
import requests

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / '/home/pi/yolov5/data/custom_data/best.pt',  # model path or triton URL
        # weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / '/home/pi/yolov5/data/custom_data/custom_dataset.yaml',  # dataset.yaml path
        # data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(480, 480),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    result = []

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    
    
    for path, im, im0s, vid_cap, s in dataset:
        resp = requests.get("http://probono.codedbyjst.com/robotState/1").json()
        robotState = resp["robotState"]
        
        
        
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # im.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    
                    # img_height, img_width = frame.shape[:2]
        
                    # img_center_x = img_width // 2
                    # img_center_y = img_height // 2
                    
                    marker_size = 20
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    #     with open(f'{txt_path}.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    center_point = round((c1[0]+c2[0])/2), round((c1[1]+c2[1])/2)
                    # width = c2[0] - c1[0]
                    # center_point = list(center_point)
                    
                    x_pos = (c1[0]+c2[0])/2 # bound box center x
                    y_pos = c2[0] - c1[0]   # = bound box width
                    
                    cv2.circle(im0,center_point,5,(0,255,0),2)
                    
                    # cv2.rectangle(frame, (img_center_x - marker_size, img_center_y - marker_size),
                    #   (img_center_x + marker_size, img_center_y + marker_size), (0, 255, 0), 2)
    
                    
                    cv2.putText(im0,str(center_point),center_point,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                    # print(y_pos)
                    
                    
                    if robotState[1] == '_':
                        room, mode, report = robotState.split('_')
                        def replace_none_with_zero(value):
                            if value is None:
                                return 0
                            return value
                        
                        post_done = False
                        
                        if mode == 'object' and room == '1':
                            if not post_done:
                                requests.post("http://probono.codedbyjst.com/reportLogData", json={"xpos": x_pos, "ypos": y_pos, "roomId": 1, "reportId": int(report)})
                                print("===================done1===================")
                                post_done = True
                                # resp2 = requests.get("http://probono.codedbyjst.com/reportLogData/recent/1").json()
                                # recent_xpos1 = resp2["xpos"]
                                # recent_ypos1 = resp2["ypos"]
                                
                            # print(x_pos)
                            # print(y_pos)
                            
                            requests.put("http://probono.codedbyjst.com/robotState/1", json={"robotState":"SECOND_MOVE"})
                            print("===================exit1===================")
                            post_done = False  
                            
                        elif mode == 'object' and room == '2':
                            if not post_done:
                                requests.post("http://probono.codedbyjst.com/reportLogData", json={"xpos": x_pos, "ypos": y_pos, "roomId": 2, "reportId": int(report)})
                                print("===================done2===================")
                                post_done = True
                                    
                            requests.put("http://probono.codedbyjst.com/robotState/1", json={"robotState":"THIRD_MOVE"})
                            print("===================exit2===================")   
                            post_done = False  
                            
                        elif mode == 'object' and room == '3':
                            if not post_done:
                                resp2 = requests.get("http://probono.codedbyjst.com/reportLogData/recent/1/{roomId}?roomId=3").json()
                                recent_xpos3 = resp2["xpos"]
                                recent_ypos3 = resp2["ypos"]
                                
                                print(recent_xpos3)
                                print(recent_ypos3)
                                
                                print("===================done3.0===================")
                                
                                requests.post("http://probono.codedbyjst.com/reportLogData", json={"xpos": x_pos, "ypos": y_pos, "roomId": 3, "reportId": int(report)})
                                
                                print(x_pos)
                                print(y_pos)
                                print("===================done3.1===================")
                                
                                # 처음 최근 데이터를 불러오면 데이터 저장된게 없기 때문에 None값을 불러옴
                                # None을 0으로 변환
                                if recent_xpos3 == None and recent_ypos3 == None:
                                    recent_xpos3 = replace_none_with_zero(recent_xpos3)
                                    recent_ypos3 = replace_none_with_zero(recent_ypos3)
                                    print(recent_xpos3)
                                    print(recent_ypos3)
                                    print("===================done3.1.1===================")
                                
                                # 현재 - 최신 = 이동 데이터 계산
                                xcalc3 = x_pos - recent_xpos3
                                ycalc3 = y_pos - recent_ypos3
                                print(xcalc3)
                                print(ycalc3)
                                print("===================done3.2===================")
                                
                                # 현재 - 최신 = 이동 데이터 계산
                                # 이동 데이터 계산 // 픽셀 값 = 움직인 좌우 길이, 움직인 앞뒤 길이
                                diffxpos3 = xcalc3 // 12
                                diffypos3 = ycalc3 // 2
                                print(diffxpos3)
                                print(diffypos3)
                                print("===================done3.3===================")
  
                                # 움직이지 않았을 경우
                                if abs(xcalc3) == x_pos and abs(ycalc3) == y_pos:
                                    print("move nothing")
                                else:
                                    pass


                                # diffxpos3 = xcalc3 // 12 # center_x의 변위값 cm
                                # diffypos3 = ycalc3 // 8 # bound box_width의 변위값 cm
                                
                                requests.post("http://probono.codedbyjst.com/reportLogData",json={"diffXPos":diffxpos3, "diffYPos":diffypos3, "roomId": 3, "reportId" : int(report)})
                    
                                print("===================done3===================")
                                post_done = True 
                                
                                
                            requests.put("http://probono.codedbyjst.com/robotState/1", json={"robotState":"COMEBACK_BASE"})
                            print("===================exit3===================")   
                            post_done = False  
                            
                        else:
                            break   
                            
                            
                            # print('bounding box = ', center_x, center_y, width, height) # bounding box 좌표
                        
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        
                        # if save_txt:
                        #     with open(f'{txt_path}.txt', 'w') as f:
                        #         for *xyxy, conf, cls in reversed(det):
                        #             c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        #             center_point = round((c1[0] + c2[0]) / 2), round((c1[1] + c2[1]) / 2
                        
                    # if save_crop:
                        # save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                        

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond


        # Print time (inference-only)
        # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))

    
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
    
    
    
## pixel계산
# center_point 기준
# 318 -> 195 10cm - 123 pixel차이 
# 318 -> 256 5cm - 62 pixel차이

# 1cm - 12 pixel 이동 오차 +-2

# y기준
# 8pixel -> 10cm
# 0.8(=1)pixel -> 1cm