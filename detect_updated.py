#!/usr/bin/env python3

########################################################################################################
# Authored by Farah Saeed (farah.saeed@uga.edu) and Saurabh Arora (sa08751@uga.edu), as a part of project of thinc lab at Computer Science
# Dept at the university of Georgia. 
########################################################################################################

import argparse

import torch.backends.cudnn as cudnn

from utils import google_utils
from utils.datasets import *
from utils.utils import *


import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Float32MultiArray
from collections import OrderedDict
from sawyer_irl_project.srv import yoloOutputs,yoloOutputsResponse

global pubCentBlem,pubCentUnBlem,pubLocOrderedPreds,pubCentLocOrd
#publishing float array of centroids on 2 separate topics: centroids_blemished, centroids_unblemished
# pubCentLocOrd = rospy.Publisher('location_ordered_centroids', Float32MultiArray, queue_size=2)
# pubCentBlem = rospy.Publisher('centroids_blemished', Float32MultiArray, queue_size=2)
# pubCentUnBlem = rospy.Publisher('centroids_unblemished', Float32MultiArray, queue_size=2)
# pubLocOrderedPreds = rospy.Publisher('location_ordered_predictions', Int32MultiArray, queue_size=2)
# for streaming
pubYoloPredictionsFlat = rospy.Publisher('yolo_predictions_flattened', Float32MultiArray, queue_size=2)
pubLocOrderedBoxesPreds = rospy.Publisher('location_ordered_boxes_predictions', Float32MultiArray, queue_size=2)

msgdata_arrCentxyPredOrd = []

def handle_yoloOutputsService(req):
    return yoloOutputsResponse(True) #Float32MultiArray(data=msgdata_arrCentxyPredOrd))

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    # if os.path.exists(out):
    #     shutil.rmtree(out)  # delete output folder
    # os.makedirs(out)  # make new output folder
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
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    #img0 = input_image.astype('float32')
    #img0 = cv2.resize(img0, (640,480), interpolation = cv2.INTER_AREA)
    #img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    #img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    #img = np.ascontiguousarray(img)



################################################################################################## no of frames
    for i in range(100): # path, img, im0s, vid_cap in dataset: #
        vid_cap = None
        path = 'img'+str(i)+'.jpg'

        msg = rospy.wait_for_message("/kinect2/hd/image_color",numpy_msg(Image))
        # msg = rospy.wait_for_message("/rgb/image_raw",numpy_msg(Image))
        print('msg.encoding', msg.encoding, 'msg.height', msg.height, 'msg.width', msg.width)
        im0s = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1).astype('float32')
        # for Azure
        # im0s = cv2.cvtColor(im0s, cv2.COLOR_BGRA2BGR)
        cv2.imwrite('sample.jpg', im0s);print('im0s',im0s); 
        img = letterbox(im0s, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        #img = torch.from_numpy(img).to(device)
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

        # needed specific to each image
        Loc2arrCent = {}
        arrCentBlem = []
        arrCentUnBlem = []
        Loc2Cls = {}
        Loc2arrCentPreds = {}

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh

            bounding_boxes = {}
            print("len(det) ",len(det))
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
                for *xyxy, conf, cls in det:

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

                        if centy > 300 and centy < 800:
                            # FOR CAMERA LOOKING FROM SIDE,
                            # if center is on the belt
                            print(centy)
                            if names[int(cls)]=='blemished':
                                # Loc2Cls[centy] = 0
                                Loc2arrCentPreds[centy] = (centx,centy,tlx,tly,brx,bry,0) 
                            else:
                                Loc2arrCentPreds[centy] = (centx,centy,tlx,tly,brx,bry,1) 
                                # Loc2Cls[centy] = 1

                            # Loc2arrCent[centy] = (centx,centy,tlx,tly,brx,bry) 
                            

                            # if names[int(cls)]=='blemished' and centx not in arrCentBlem:
                            #     arrCentBlem.append(centx)
                            #     arrCentBlem.append(centy)
                            # else:
                            #     if centy not in arrCentUnBlem:
                            #         arrCentUnBlem.append(centx)
                            #         arrCentUnBlem.append(centy)


            # Print time (inference + NMS)
            print('%s Inference. (%.3fs)' % (s, t2 - t1))
            # im0 = im0[miny:maxy, minx:maxx]


            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                arr_img = np.array(im0)
                flat_arr = np.ravel(arr_img).tolist()
                # pubYoloPredictionsFlat.publish(Float32MultiArray(data=flat_arr))
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
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

        # publish centroids for current image 
        # print("publishing arrCentBlem ",arrCentBlem )
        # print("publishing arrCentUnBlem ",arrCentUnBlem )
        # pubCentBlem.publish(Float32MultiArray(data=arrCentBlem))
        # pubCentUnBlem.publish(Float32MultiArray(data=arrCentUnBlem))
        
        # make array LocOrderedPreds 
        LocOrderedPreds = list(OrderedDict(sorted(Loc2Cls.items(), key=lambda t: t[0])).values())
        # print("length LocOrderedPreds ",len(LocOrderedPreds)) 
        # pubLocOrderedPreds.publish(Int32MultiArray(data=LocOrderedPreds))
        arrCentxyOrd = list(OrderedDict(sorted(Loc2arrCent.items(), key=lambda t: t[0])).values())
        # print("len(arrCentxyOrd) ",len(arrCentxyOrd)) 
        arrCentLocOrd = []
        for i in range(len(arrCentxyOrd)):
            arrCentLocOrd.append(arrCentxyOrd[i][0])
            arrCentLocOrd.append(arrCentxyOrd[i][1])
            arrCentLocOrd.append(arrCentxyOrd[i][2])
            arrCentLocOrd.append(arrCentxyOrd[i][3])
            arrCentLocOrd.append(arrCentxyOrd[i][4])
            arrCentLocOrd.append(arrCentxyOrd[i][5])
        # print("publishing arrCentLocOrd ",arrCentLocOrd) 
        # pubCentLocOrd.publish(Float32MultiArray(data=arrCentLocOrd))
        
        print("len Loc2arrCentPreds",len(Loc2arrCentPreds))
        arrTupsCentxyPredOrd = list(OrderedDict(sorted(Loc2arrCentPreds.items(), key=lambda t: t[0])).values())
        print("number of elements in arrTupsCentxyPredOrd ",len(arrTupsCentxyPredOrd))
        global msgdata_arrCentxyPredOrd
        msgdata_arrCentxyPredOrd = []
        for i in range(len(arrTupsCentxyPredOrd)):
            msgdata_arrCentxyPredOrd.append(arrTupsCentxyPredOrd[i][0])
            msgdata_arrCentxyPredOrd.append(arrTupsCentxyPredOrd[i][1])
            msgdata_arrCentxyPredOrd.append(arrTupsCentxyPredOrd[i][2])
            msgdata_arrCentxyPredOrd.append(arrTupsCentxyPredOrd[i][3])
            msgdata_arrCentxyPredOrd.append(arrTupsCentxyPredOrd[i][4])
            msgdata_arrCentxyPredOrd.append(arrTupsCentxyPredOrd[i][5])
            msgdata_arrCentxyPredOrd.append(arrTupsCentxyPredOrd[i][6])

        pubLocOrderedBoxesPreds.publish(Float32MultiArray(data=msgdata_arrCentxyPredOrd))


    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

class Options:
    def __init__(self):
      self.weights = 'src/beginner_tutorials/scripts/yolov5/weights/best.pt'
      self.source = 'src/beginner_tutorials/scripts/yolov5/inference/images'
      self.output = 'src/beginner_tutorials/scripts/yolov5/inference/output'
      self.img_size = 640
      self.conf_thres = 0.75 #0.4
      self.iou_thres = 0.5
      self.fourcc = 'mp4v'
      self.device = ''
      self.view_img = False
      self.agnostic_nms = False
      self.augment = False      
      self.save_txt = False
      self.classes = None

opt = Options()
if __name__ == '__main__':
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

    rospy.init_node('detect_updated', anonymous=True)
    rate = rospy.Rate(1) # 10hz
    s = rospy.Service('yoloOutputsService', yoloOutputs, handle_yoloOutputsService)
    while not rospy.is_shutdown():
        with torch.no_grad():
            detect()
        rate.sleep()

        # Update all models
        # for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
        #    detect()
        #    create_pretrained(opt.weights, opt.weights)



#opt = Options()


#def callback(data):
#    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    
#def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously. /kinect2/hd/image_color

    #rospy.Subscriber("chatter", String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    # rospy.spin()








