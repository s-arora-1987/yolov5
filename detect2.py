#!/usr/bin/env python3

########################################################################################################
# Authored by Farah Saeed (farah.saeed@uga.edu) and Saurabh Arora (sa08751@uga.edu), as a part of project of thinc lab at Computer Science
# Dept at the university of Georgia. 
########################################################################################################

import cv_bridge
import rospy
import argparse

import torch.backends.cudnn as cudnn
#from google.colab.patches import cv2_imshow
from utils import google_utils
from utils.datasets import *
from utils.utils import *

import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.utils import xyxy2xywh, xywh2xyxy

import rospy
#from sa_net_robo.srv import sanet_srv
from sensor_msgs.msg import Image
from numpy import save as savenpy
from cv_bridge import CvBridge, CvBridgeError
import os

import sys
sys.path.append('src/beginner_tutorials/scripts')

from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Float32MultiArray
from collections import OrderedDict

rospy.init_node('detect2', anonymous=True)
import time

global pubCentBlem,pubCentUnBlem,pubLocOrderedPreds,pubCentLocOrd
#publishing float array of centroids on 2 separate topics: centroids_blemished, centroids_unblemished
pubCentLocOrd = rospy.Publisher('location_ordered_centroids', Float32MultiArray, queue_size=2)
pubCentBlem = rospy.Publisher('centroids_blemished', Float32MultiArray, queue_size=2)
pubCentUnBlem = rospy.Publisher('centroids_unblemished', Float32MultiArray, queue_size=2)
pubLocOrderedPreds = rospy.Publisher('location_ordered_predictions', Int32MultiArray, queue_size=2)

def grabarrimg_rgb(msg):

    flat_arr = np.array(msg.data)
    rospy.loginfo("I received flat_array of shape %s .",flat_arr.shape)
    # kinectv2
    rgb_mem = np.reshape(flat_arr,(1080, 1920, 3))
    #sawyer head-camera
    #rgb_mem = np.reshape(flat_arr,(800, 1280, 3))
    rospy.loginfo("I received arrimg of shape %s . starting a call for prediction.",rgb_mem.shape)
    with torch.no_grad():
        det_bboxes = detect(rgb_mem)
    return

def grabarrimg_rgb_sd(msg):

    flat_arr = np.array(msg.data)
    rospy.loginfo("sd image: I received flat_array of shape %s .",flat_arr.shape)
    # kinectv2
    rgb_mem = np.reshape(flat_arr,(424, 512, 3))
    #sawyer head-camera
    #rgb_mem = np.reshape(flat_arr,(800, 1280, 3))
    rospy.loginfo("sd image: I received arrimg of shape %s . starting a call for prediction.",rgb_mem.shape)
    with torch.no_grad():
        det_bboxes = detect(rgb_mem)
    return

def detect(input_image,save_img=True):

    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    global frame_num,model
    global pubCentBlem,pubCentUnBlem

    # Initialize
    #device = torch_utils.select_device(opt.device)
    #if os.path.exists(out):
    #    shutil.rmtree(out)  # delete output folder
    #os.makedirs(out)  # make new output folder

    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    print("class names array ", names)
    # names = ['blemished', 'unblemished', 'glove', 'belt', 'bin', 'head']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    bounding_boxes_all_images = []
    
    img0 = input_image.astype('float32')
    #img0 = cv2.resize(img0, (640,480), interpolation = cv2.INTER_AREA)
    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    dataset = [("frame_num"+str(frame_num)+'.jpg', img, img0, None)]

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # pred = model(img)[0]

        t2 = torch_utils.time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = torch_utils.time_synchronized()


        # needed specific to each image
        Loc2arrCent = {}
        arrCentBlem = []
        arrCentUnBlem = []
        Loc2Cls = {}

        # Process detections

        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            bounding_boxes = {}

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                minx = 5000 
                miny = 5000
                maxx = 0
                maxy = 0

                box_num = 0
                bounding_boxes = {}
                for *xyxy, conf, cls in det:
                    box_num+=1
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #print("Box num:", box_num, " label: ",names[int(cls)], " xyxy: ", int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), "xywh: ", float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3]))
                    if (bounding_boxes.get(names[int(cls)], None) == None):
                        bounding_boxes[names[int(cls)]] = [[int(xyxy[0]), int(xyxy[1]), abs(int(xyxy[2])-int(xyxy[0])), abs(int(xyxy[3])-int(xyxy[1]))]]
                    else:
                        bounding_boxes[names[int(cls)]].append([int(xyxy[0]), int(xyxy[1]), abs(int(xyxy[2])-int(xyxy[0])), abs(int(xyxy[3])-int(xyxy[1]))])


                    tlx,tly, brx, bry = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    if tlx < minx: 
                        minx = tlx 
                    if tly < miny:
                        miny = tly
                    if bry > maxy:
                        maxy = bry
                    if brx > maxx:
                        maxx = brx #crop_img = img[y:y+h, x:x+w]


                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        # print("label input to plot_one_box ",label)
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    # append onion centroid to list of centroids
                    if names[int(cls)]=='blemished' or names[int(cls)]=='unblemished':
                        centx = (tlx+brx)/2
                        centy = (tly+bry)/2

                        if centx > 250:
                            # FOR CAMERA LOOKING FROM FRONT
                            # if center is on the belt
                            if names[int(cls)]=='blemished':
                                Loc2Cls[centx] = 0
                            else:
                                Loc2Cls[centx] = 1

                            Loc2arrCent[centx] = (centx,centy)
                            if names[int(cls)]=='blemished' and centx not in arrCentBlem:
                                arrCentBlem.append(centx)
                                arrCentBlem.append(centy)
                            else:
                                if centx not in arrCentUnBlem:
                                    arrCentUnBlem.append(centx)
                                    arrCentUnBlem.append(centy)

############################################### all bounding boxes for this frames: "bounding_boxes" ########################################################################################################

            bounding_boxes_all_images.append(bounding_boxes)
            print("Frame number = ", frame_num, "Bounding boxes: ", bounding_boxes)  
            frame_num+=1   
            # Print time (inference + NMS)
            print('%s Inference. (%.3fs)' % (s, t2 - t1))
            print('%s NMS. (%.3fs)' % (s, t3 - t2))
            #print('im0.shape before',im0.shape)
############################################### cropping
            #im0 = im0[miny:maxy, minx:maxx]
            #print('im0.shape after',im0.shape)


            # Stream results
            view_img= True
            if view_img:
                #cv2_imshow( im0)
                not_showing = True
                #cv2.imshow(p, im0)
                
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            save_img = True
            if save_img: #'src/beginner_tutorials/scripts/yolov5/inference/output'
                #save_path = '/home/psuresh/src/beginner_tutorials/scripts/yolov5/inference/output/frame1.jpg'
                cv2.imwrite(save_path, im0) #if dataset.mode == 'images':
########################################################## comment this line to avoid saving images  ########################################################################################################

        # publish centroids for current image 
        print("publishing arrCentBlem ",arrCentBlem )
        print("publishing arrCentUnBlem ",arrCentUnBlem )
        pubCentBlem.publish(Float32MultiArray(data=arrCentBlem))
        pubCentUnBlem.publish(Float32MultiArray(data=arrCentUnBlem))
        
        # make array LocOrderedPreds 
        LocOrderedPreds = list(OrderedDict(sorted(Loc2Cls.items(), key=lambda t: t[0])).values())
        print("publishing LocOrderedPreds ",LocOrderedPreds) 
        pubLocOrderedPreds.publish(Int32MultiArray(data=LocOrderedPreds))
        arrCentxyOrd = list(OrderedDict(sorted(Loc2arrCent.items(), key=lambda t: t[0])).values())
        print("arrCentxyOrd ",arrCentxyOrd) 
        arrCentLocOrd = []
        for i in range(len(arrCentxyOrd)):
            arrCentLocOrd.append(arrCentxyOrd[i][0])
            arrCentLocOrd.append(arrCentxyOrd[i][1])
        print("publishing arrCentLocOrd ",arrCentLocOrd) 
        pubCentLocOrd.publish(Float32MultiArray(data=arrCentLocOrd))


    #global img_processed
    #img_processed = True

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    return bounding_boxes_all_images

######################################################################################


import os
#print("hereeeeeeeeeeeeeeeeeeee:",os.getcwd())

class Options:
    def __init__(self):
      self.weights = 'src/beginner_tutorials/scripts/yolov5/weights/best.pt'
      self.source = 'src/beginner_tutorials/scripts/yolov5/inference/images'
      self.output = 'src/beginner_tutorials/scripts/yolov5/inference/output'
      self.img_size = 640
      self.conf_thres = 0.4
      self.iou_thres = 0.5
      self.fourcc = 'mp4v'
      self.device = ''
      self.view_img = False
      self.agnostic_nms = False
      self.augment = False      
      self.save_txt = False
      self.classes = None

opt = Options()
# # if __name__ == '__main__':
# parser = argparse.ArgumentParser()
# parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='model.pt path')
# parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
# parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
# parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
# parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
# parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
# parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
# parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
# parser.add_argument('--view-img', action='store_true', help='display results')
# parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
# parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
# parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
# parser.add_argument('--augment', action='store_true', help='augmented inference')
# opt = parser.parse_args()
opt.img_size = check_img_size(opt.img_size)
print(opt)

opt.weights='src/beginner_tutorials/scripts/yolov5/weights/best.pt'
opt.view_img=False
# Initialize
frame_num = 0
device = torch_utils.select_device(opt.device)
# Load model
#google_utils.attempt_download(opt.weights)
model = torch.load(opt.weights, map_location=device)['model'].float()  # load to FP32
model.to(device).eval()

rospy.Subscriber("arrimg", Int32MultiArray, grabarrimg_rgb)
rospy.Subscriber("arrimg_sd", Int32MultiArray, grabarrimg_rgb_sd)


rospy.loginfo("real-time yolov5 prediction started")


rospy.spin()

