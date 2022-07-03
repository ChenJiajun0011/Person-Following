from curses import A_ATTRIBUTES
import re
import pyrealsense2 as rs
import cv2

import sys
import numpy as np
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32MultiArray, String
from std_srvs.srv import Trigger

sys.path.insert(0, './yolov5')
from pid_controller import get_controls, stop_controls, get_average_distance
from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import math
import torch
import rospy
import time

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective

WIDTH = 640
FPS = 30
HEIGHT = 480
i_error_linear = 0
i_error_angular = 0
d_error_linear = 0
d_error_a = 0
A = []
twist = Twist()

rospy.init_node('webcam_display', anonymous=True)
pub = rospy.Publisher('motion_control/cmd_vel', Twist, queue_size=1)

config = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

pipeline = rs.pipeline()
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

cfg = get_config()
cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
attempt_download("deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7", repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                    nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

device = select_device("cpu")  # "0" for gpu
half = device.type != "cpu"  # half precision only supported on CUDA

model = attempt_load('yolov5/weights/yolov5s.pt', map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size
names = model.module.names if hasattr(model, 'module') else model.names  # get class names
if half:
    model.half()

def mat_inter(box1, box2):
    # 判断两个矩形是否相交
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False

def solve_coincide(box1, box2):
    # box=(xA,yA,xB,yB)
    # 计算两个矩形框的重合度
    if mat_inter(box1, box2) == True:
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        col = min(x02, x12) - max(x01, x11)
        row = min(y02, y12) - max(y01, y11)
        intersection = col * row
        area1 = (x02 - x01) * (y02 - y01)
        area2 = (x12 - x11) * (y12 - y11)
        coincide = intersection / (area1 + area2 - intersection)
        return coincide
    else:
        return 0
# 
def Yolo_Deepsort():
    Target_id = []
    id_copy = []
    global key_press
    global TargetBox
    key_press = 0
    TargetBox = []
    global bridge
    bridge = CvBridge()
    try:
        rospy.on_shutdown(spot_shutdown_hook)
        while not rospy.is_shutdown():
            e = time.time()
            global color_image
            global images
            global result_image
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame or not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            # Depth画像
            depth_color_frame = rs.colorizer().colorize(depth_frame)
            depth_color_image = np.asanyarray(depth_color_frame.get_data())
            img0 = color_image.copy()
            (H, W) = depth_color_image.shape[:2]
            screen_center = int(W / 2)
            img = letterbox(img0, 640, 32, True)[0]
            # Convert
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0

            frame_idx = 0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img, augment=False)[0]
            pred = non_max_suppression(
                pred, 0.4, 0.5, agnostic=False)

            for i, det in enumerate(pred):  # detections per image
                im0 = img0
                # print string
                annotator = Annotator(im0, line_width=2, pil=not ascii)

                anotationList = []
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]
                    # pass detections to deepsort
                    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    # print('ListLength:', len(outputs))
                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)):
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]

                            c = int(cls)  # integer class
                            if names[c] == "person":
                                label = f'{id} {names[c]} {conf:.2f}'
                                annotator.box_label(bboxes, label, color=colors(c, True))

                                bbox_left = output[0]
                                bbox_top = output[1]
                                bbox_w = output[2] - output[0]
                                bbox_h = output[3] - output[1]

                                center_x = math.floor(bbox_left + (bbox_w / 2))
                                center_y = math.floor(bbox_top + (bbox_h / 2))
 
                                # get depth of the box center
                                depth = round(depth_frame.get_distance(center_x, center_y), 3)
                                # anotationList is the list of person
                                anotationList.append(
                                    [frame_idx, id, c, names[c], bbox_left, bbox_top, bbox_w, bbox_h, center_x,
                                     center_y,
                                     depth])

            result_image = annotator.result()

            if len(anotationList) > 0: # if there are people
                global p1, p2
                for anotation in anotationList:
                    print(anotation)
                    cv2.putText(result_image, str(anotation[10]), (int(anotation[8]), int(anotation[9])),
                                cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 3, cv2.LINE_AA)
                    p1 = (anotation[4], anotation[5])
                    p2 = (anotation[4] + anotation[6], anotation[5] + anotation[7])
                    cv2.rectangle(images, p1, p2, (255, 0, 0), 4)

                # # Select Target and save the target id, then will skip this part
                # key_press detect if user press "框选目标" on master computer
                if len(Target_id) == 0:
                    # User did press "框选目标"
                        # id_copy用来保存目标ID或检查之前选的目标ID是否重新出现
                        if id_copy:
                            print("id_copy", id_copy)
                            for ID in anotationList:
                                if ID[1] == id_copy[0]:
                                    Target_id = id_copy
                                    break
                            if len(Target_id) == 0:
                                Target_id = TargetSelect(anotationList)
                                id_copy = Target_id
                        else:
                            Target_id = TargetSelect(anotationList)
                            id_copy = Target_id

                if Target_id is not None:
                    try:# if Target_id is not None:       
                        # Match the saved target id in all person box in a new frame to find out the target in this new frame
                        for FindID in anotationList:
                            if FindID[1] == Target_id[0]:
                                TargetCenter = FindID
                                cv2.putText(result_image, str('Target'), (int(TargetCenter[4]), int(TargetCenter[5] + 60)),
                                            cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 8, cv2.LINE_AA)
                            else:  # if there are people but no matching target
                                print("Target Lost...STOP")
                                Target_id = []
                                TargetBox = []
                                twist = Twist()
                                # Target_id = []
                                twist.linear.x = 0
                                twist.linear.y = 0
                                twist.angular.z = 0
                                pub.publish(twist)
                    except:
                        pass

                if Target_id:
                    tar_x, tar_z = TargetCenter[8], TargetCenter[10]
                    twist, error = get_controls(tar_x, tar_z, 1/3, 0, 0.2, -1/500, 0, 0, i_error_linear, i_error_angular, d_error_linear, d_error_a)
                    print(f'linear x: {twist.linear.x}, angular z: {twist.angular.z}')
                    i_error_l, i_error_a, d_error_l, d_error_l = error
                    try:
                        pub.publish(twist)
                    except CvBridgeError as e:
                        twist = stop_controls()
                        pub.publish(twist)
                        print ('e')
 
                    # Motion(TargetCenter,screen_center)
                    TargetCenter = []
                # else:
                #     send out stop twist to spot
            
            else:  # if there is no person, clear Targer_id
                print("Target Lost, saved ID, STOP", Target_id)
                twist = Twist()
                TargetBox = []
                # Target_id = []
                twist.linear.x = 0
                twist.linear.y = 0
                twist.angular.z = 0
                pub.publish(twist)
                # should send stop Twist to spot++++++++++++++

            images = np.hstack((result_image, depth_color_image))
            print('Target_id:.', Target_id)
            webcamImagePub(images)   # If there is target, publish following image

            s = time.time()
            print("time for current frame:", round(s - e, 3),"\n")
            # print("TargetBox from amplify: ", TargetBox)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

def TargetSelect(anotationList):
    global inter_area, data_to_send
    global key_press
    global TargetBox
    intersectionList = []
    Target_id = []
    data = Int32MultiArray()
    data.data = []
    # TargetBoxdata = Int32MultiArray()
    # TargetBoxdata.data = []
    img_pub = rospy.Publisher('/Realsense/TargetSelect', Image, queue_size=1,latch=True)
    # img_pub = rospy.Publisher('/Realsense/TargetSelect', Image, queue_size=1)
    msg = bridge.cv2_to_imgmsg(result_image, encoding="bgr8")
    img_pub.publish(msg)    # pub image for rosmaster to select
    print("Select target in 10 second please.")
    try:  # try because if user do not select target in amplify_hz, data = nothing
        data = rospy.wait_for_message('/Realsense/Targetbox', Int32MultiArray, timeout=10) 
        print("bbbbbbbbbbbWait or not?!?!?!?!?!")
    except:
        pass

    # if TargetBoxdata.data:
    if data.data:
        print("data.data",data)
        [min_x, min_y, width, height] = data.data
        TargetBox = data.data

        print("TargetBox from amplify: ", TargetBox)
        for inter_area in anotationList:
            box1 = [min_x, min_y, min_x + width, min_y + height]
            box2 = [inter_area[4], inter_area[5], inter_area[4] + inter_area[6],
                    inter_area[5] + inter_area[7]]
            # calculate the intersection between every person box and selected area
            intersection1 = solve_coincide(box1, box2)
            intersectionList.append(intersection1)
        print('intersectionList:', intersectionList)
        intersectionMax = np.argmax(intersectionList)

        # Target id selected
        Target_id = [anotationList[intersectionMax][1]]
        print('Target_id:', Target_id)
        TargetBox = []
        key_press = 0
        # clear data.data after used
        data.data = []
        # TargetBoxdata.data = []
    else:
        TargetBox = []
        Target_id = []
    return Target_id

def spot_shutdown_hook():
    print("Calling spot to sit down")
    # spot_sit_topic = "spot/sit"
    # call_trigger_service(spot_sit_topic)

def call_trigger_service(topic):
    """Calls Service which is of type std_srvs/Trigger.
    Args:
        topic (str): topic of service.
    """
    rospy.wait_for_service(topic)
    try:
        claim = rospy.ServiceProxy(topic, Trigger)
        resp_claim = claim()
        print(f"Service call successful with response: {resp_claim}")
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")

def webcamImagePub(frame):
    img_pub = rospy.Publisher('Realsense/FollowingImg', Image, queue_size=2)
    bridge = CvBridge()
    msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
    img_pub.publish(msg)

def TargetImagePub(frame):
    img_pub = rospy.Publisher('Realsense/TargetSelect', Image, queue_size=1)
    bridge = CvBridge()
    msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
    img_pub.publish(msg)

if __name__ == '__main__':
    test = Yolo_Deepsort()
    print(test)