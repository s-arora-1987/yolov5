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

cvb = CvBridge()

mem_action = 0
mem_x = None
mem_y = None
mem_theta = None
rgb_mem = np.array([])
print("np.isempty(rgb_mem) ",rgb_mem.size == 0)
msg_received = False
img_processed = True
depth_mem = None
same_flag = 0


help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']

rospy.init_node('detect1', anonymous=True)
import time

def grabarrimg_rgb(msg):
	global msg_received,img_processed
	while not img_processed:
		pass
	msg_received = True
	img_processed = False
	flat_arr = msg.data
	global rgb_mem
	rgb_mem = np.reshape(flat_arr,(1080, 1920, 3))
	rospy.loginfo("I received arrimg of shape %s",rgb_mem.shape)
	return

class LoadImages1:  # for inference
    def __init__(self, path, session, img_size=416):
        path = str(Path(path))  # os-agnostic
        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, '*.*')))
        elif os.path.isfile(path):
            files = [path]

        # images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        images = []
        if session == 'session1':          
            for i in range(353):
                images.append(os.path.join(path,'session1 ('+str(i+1)+').jpg') )

        elif session == 'session2':          
            for i in range(386):
                images.append(os.path.join(path,'session2 ('+str(i+1)+').jpg') )


        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        nI, nV = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        # off line data loaging
        #self.nF = nI + nV  # number of files
        # real-time from kinect
        self.nF = 3
        self.video_flag = [False] * nI + [True] * nV
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nF > 0, 'No images or videos found in %s. Supported formats are:\nimages: %s\nvideos: %s' % \
                            (path, img_formats, vid_formats)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        print("called next of LoadImages1 ")
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nF:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nF, self.frame, self.nframes, path), end='')

        else:
            # Read image
            self.count += 1
########################################################################################### input image here ###############################################################################################
            #from sa_net_robo.script.sanet_service2 import rgb_mem
            # real-time from kinect

            global rgb_mem
            global msg_received, img_processed
            while not msg_received or rgb_mem.size == 0:
                if not msg_received:
                    print("image not changed")
                if rgb_mem.size == 0:
                    print("rgb_mem.size == 0")
                #pass
            img0 = rgb_mem            
            msg_received = False 
            
            # off line
            #img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nF, path), end='')

        # Padded resize
        print("input image shape :",img0.shape)
        img0 = img0.astype('float32')
        #img0 = cv2.resize(img0, (640,480), interpolation = cv2.INTER_AREA)
	
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

########################################################################### input number of frames to run it for ###########################################################################################
    def __len__(self):
        return self.nF  # number of files # return 10 if frames = 10


def detect(source_path,session,save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        #dataset = LoadImages(source, session, img_size=imgsz)
        #dataset = LoadImages(source, img_size=imgsz)
        dataset = LoadImages1(source,session, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    bounding_boxes_all_images = []
    frame_num = 0


    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                minx = 0 
                miny = 0
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
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

############################################### all bounding boxes for this frames: "bounding_boxes" ########################################################################################################
            global img_processed
            img_processed = True

            bounding_boxes_all_images.append(bounding_boxes)
            print("Frame number = ", frame_num, "Bounding boxes: ", bounding_boxes)  
            frame_num+=1   
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            im0 = im0[miny:maxy, minx:maxx]


            # Stream results
            view_img= True
            if view_img:
                #cv2_imshow( im0)
                not_showing = True
                #cv2.imshow(p, im0)
                
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
########################################################## comment this line to avoid saving images  ########################################################################################################
                    cv2.imwrite(save_path, im0)

                    not_saving = True
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

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

#from yolov5.detect1 import * 
#print("cv_image from yolov5 detect1 \n",np.array(detect1.cv_image))
rospy.Subscriber("arrimg", Int32MultiArray, grabarrimg_rgb)

#rospy.init_node("sanet_server")
# for kinect v2
#rospy.Subscriber("/kinect2/hd/image_depth_rect", Image, grabdepth)
# for kinect v1
#rospy.Subscriber("/camera/depth_registered/image_raw", Image, grabdepth)
# for kinect v2
#rospy.Subscriber("/kinect2/hd/image_color", Image, grabrgb)
# for kinect v1
#rospy.Subscriber("/camera/rgb/image_raw", Image, grabrgb)
rospy.loginfo("sa-net started")

# det_bboxes = None
# source_path = 'inference/images/session2 (210).jpg'
source_path = '../../jpg_files'
with torch.no_grad():
    det_bboxes = detect(source_path,'session1')

        # Update all models
        # for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
        #    detect()
        #    create_pretrained(opt.weights, opt.weights)

#service = rospy.Service("/get_state_action", sanet_srv, getsa)
rospy.spin()

