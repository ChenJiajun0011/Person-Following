#!/usr/bin/env python
#!coding=utf-8

import rospy
import cv2
import time
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray

def on_mouse(event, x, y, flags, param):
    global Target_img, point1, point2, A, var1
    img2 = Target_img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 5)
        cv2.imshow('RealSense', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 5)
        cv2.imshow('RealSense', img2)
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), 5)
        cv2.imshow('RealSense', img2)
        A = []
        min_x = min(point1[0], point2[0])
        A.append(min_x)
        min_y = min(point1[1], point2[1])
        A.append(min_y)
        width = abs(point1[0] - point2[0])
        A.append(width)
        height = abs(point1[1] - point2[1])
        A.append(height)
        var1 = A  # if selected they will be the same
        print('Selected area:', A, '\nPress Enter to start, Press Esc to stop')

def TargetCallback(data):
    global Target_img, A
    Target_img = bridge.imgmsg_to_cv2(data, "bgr8")
    cv2.namedWindow('RealSense')
    cv2.setMouseCallback('RealSense', on_mouse)
    cv2.imshow('RealSense', Target_img)
    cv2.waitKey(7000)
    data_to_send = Int32MultiArray()
    try:
        data_to_send.data = A
        pub_TargetBox.publish(data_to_send)
        A = []  # clear A will publish empty to Targetbox topic
    except:
        pass
    
    cv2.destroyAllWindows()

def MotionCallback(data):
    global last_input
    last_input = data

if __name__ == '__main__':
    last_input = Twist()
    bridge = CvBridge()
    var1 = []
    rospy.init_node("rospy_rate_test")

    pub_spot = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size=10)
    # pub_spot = rospy.Publisher('/spot/cmd_vel', Twist, queue_size=10)
    pub_TargetBox = rospy.Publisher('/Realsense/Targetbox', Int32MultiArray, queue_size=1,latch=True)
    
    # subscribe to output window to select target
    rospy.Subscriber('/Realsense/TargetSelect', Image, TargetCallback)

    # subscribe to the motion control topic and repeat each speed command in 50 Hz
    rospy.Subscriber('/motion_control/cmd_vel',Twist, MotionCallback)
    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        pub_spot.publish(last_input)
        rate.sleep()