from curses import A_ATTRIBUTES
from difflib import Match
import re
import pyrealsense2 as rs
import cv2

import sys
import numpy as np
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32MultiArray
from std_srvs.srv import Trigger

sys.path.insert(0, './yolov5')
from PyQt5 import QtCore, QtGui, QtWidgets
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

def Yolo_Deepsort():
    Target_id = []
    id_copy = []
    global bridge
    bridge = CvBridge()
    try:
        rospy.on_shutdown(spot_shutdown_hook)
        while not rospy.is_shutdown():
            e = time.time()
            global color_image, images
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

            if len(anotationList) > 0: # 大于零表示图中有人出现
                global p1, p2
                for anotation in anotationList:
                    print(anotation)
                    cv2.putText(result_image, str(anotation[10]), (int(anotation[8]), int(anotation[9])),
                                cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 3, cv2.LINE_AA)
                    p1 = (anotation[4], anotation[5])
                    p2 = (anotation[4] + anotation[6], anotation[5] + anotation[7])
                    # 将所有人框画出来
                    cv2.rectangle(images, p1, p2, (255, 0, 0), 4)

                # # Select Target and save the target id, then will skip this part
                # 第一次开始选人,以后要再进入,就要等到Target_id被清零
                if len(Target_id) == 0:  
                    if id_copy:  # 检查有无曾经保存过的目标ID
                        MatchID = False
                        print("id_copy", id_copy)
                        # 在所有人框中查找目标人框，即查看目标ID有无在图中出现
                        for ID in anotationList:
                            if ID[1] == id_copy[0]:
                                Target_id = id_copy
                                # ID出现则表示目标人物没有丢失依然在图中，机器人保持对其跟踪
                                MatchID = True
                                break
                        # if len(Target_id) == 0: # 曾经的id不在，重新选目标
                        if MatchID == False: # 曾经的id不在图中,即是目标丢失了需要重新选目标
                            Target_id = TargetSelect(anotationList)
                            id_copy = Target_id
                    else: # 第一次直接运行这里，上面跳过
                        Target_id = TargetSelect(anotationList)
                        id_copy = Target_id

                if Target_id is not None: # 如果有Target_id,贴标
                    try:   # try因为anotationList可能为零即图中无人    
                        # Match the saved target id in all person box in a new frame to find out the target in this new frame
                        for FindID in anotationList:
                            # 查找图中是否有目标ID在其中
                            if FindID[1] == id_copy[0]:
                                # 有则提取出位置信息,
                                TargetCenter = FindID
                                print("Target Match!")
                                Matching = True
                                cv2.putText(result_image, str('Target'), (int(TargetCenter[4]), int(TargetCenter[5] + 60)),
                                            cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 8, cv2.LINE_AA)
                                break # Target match, qiut for loop
                            # Target lost: if there are people but no matching target
                            else:  
                                # 图中有人，但是没有目标人（没有目标ID）
                                print("Do not match!")
                                # 如果这里加入Target_id =[]，一旦有一个人框不匹配都清空了Target_id
                                # 那在两人情况下，每帧至少有一个人框进入else Do not match而清空Target_id，导致下帧就进入205行选人函数
                                # Target_id = [] # 这里不要清空Target_id
                                Matching = False
                                # 同理，也不能发送停止指令，不然每检测到一个不匹配的人框，它都停下来，直到match到人框
                                # twist = stop_controls()
                                # pub.publish(twist)

                        # 让for循环完所有人框再看是不是match到目标
                        # 确定没match到就是图中有人但没有目标人，再清空target_id发送停止运动指令
                        if Matching == False:
                            Target_id = [] # 这里清空Target_id让下一帧进入205行可能选新目标
                            twist = stop_controls()
                            pub.publish(twist)
                            print("Robot stop!")
                    # try except pass 遇到错误时直接中断整个for循环再执行except
                    except:
                        pass

                if Target_id: #提取深度，运动控制
                    if Matching:
                        tar_x, tar_z = TargetCenter[8], TargetCenter[10]
                        twist, error = get_controls(tar_x, tar_z, 1/3, 0, 0.2, -1/500, 0, 0, i_error_linear, i_error_angular, d_error_linear, d_error_a)
                        print(f'linear x: {twist.linear.x}, angular z: {twist.angular.z}')
                        i_error_l, i_error_a, d_error_l, d_error_l = error
                        try:
                            pub.publish(twist)
                        except CvBridgeError as e:
                            twist = stop_controls() # 
                            pub.publish(twist)
                            print ('e')
    
                        # Motion(TargetCenter,screen_center)
                        TargetCenter = []
                    else:
                        twist = stop_controls() # 停止机器人运动
                        pub.publish(twist)
                else:
                    twist = stop_controls()
                    pub.publish(twist)            
            else:  # if there is no person, clear Targer_id
                print("Target Lost")
                Target_id = []
                twist = stop_controls()
                pub.publish(twist)
            # images = np.hstack((result_image, depth_color_image))
            images = result_image
            print('Target_id:.........', Target_id)
            webcamImagePub(images)   # If there is target, publish following image

            s = time.time()
            print("time for current frame:", round(s - e, 3),"\n")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

# 此函数用于发送图片话题给amp程序弹窗选目标，
# 此函数订阅amp程序发送的/Realsense/Targetbox话题去获得用户选择的目标框即data.data
# 然后对比data.data和图中人框重合度来确定用户选了哪个为目标框，返回这个目标框的ID回主程序去开始跟踪
def TargetSelect(anotationList):
    global inter_area, data_to_send
    intersectionList = []
    Target_id = []
    # data = []

    global inter_area, data_to_send
    global TargetBox
    intersectionList = []
    Target_id = []
    data = Int32MultiArray()
    data.data = []     # clear data.data everytime
    TargetBoxdata = Int32MultiArray()
    TargetBoxdata.data = []
    # pub the latest camera view
    img_pub = rospy.Publisher('/Realsense/TargetSelect', Image, queue_size=1,latch=True)
    # img_pub = rospy.Publisher('/Realsense/TargetSelect', Image, queue_size=1)
    msg = bridge.cv2_to_imgmsg(result_image, encoding="bgr8")
    img_pub.publish(msg)    # pub image for rosmaster to select
    print("Select target in 10 second please.")
    try:  # try because if user do not select target in amplify_hz, data = nothing
        data = rospy.wait_for_message('/Realsense/Targetbox', Int32MultiArray, timeout=10) 
        # if [] is published to targetbox, data = []???
        print("bbbbbbbbbbbWait or not?!?!?!?!?!")
    except:
        pass

    if data.data:
        print("data.data",data.data)
        [min_x, min_y, width, height] = data.data
        TargetBox = data.data
        print("TargetBox from amplify: ", TargetBox)  # To see if TargetBox change
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
    return Target_id
    # else:
    #     return num_of_sec

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