#!/usr/bin/env python
#!coding=utf-8

import sys
import rospy
import cv2
import time
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
import numpy as np
from threading import Thread
from PyQt5.QtWidgets import QMainWindow,QApplication
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

begin = 0  #定义一个控制位变量，用于控制视频录制的开始与结束
vi_name = './video'  #保存路径
count = 0  #用于给视频命名
image = np.array([]) #用于共享图片信息

class CameraRosNode(QThread):

    raw_image_signal = pyqtSignal() #创建 signal信号， 当ros接收到消息后 触发子界面数据更新

    def __init__(self):
        QThread.__init__(self)  #创建线程
        print("Begin")
        self.raw_image_opencv = np.array([])
        global vi_name, count
        #对视频设置的编码解码的方式MPEG-4编码
        self.fource=cv2.VideoWriter_fourcc(*'MJPG')
        #保存的位置，以及编码解码方式，帧率，视频帧大小
        video_name = '%s/%s%s.mp4' % (vi_name,'test',count)
        self.resulte=cv2.VideoWriter(video_name,self.fource,25.0,(640,480))
        #rospy.init_node("FOllowingImg")#创建ros节点
        rospy.Subscriber('/Realsense/FollowingImg', Image, self.callback_raw_image)#此处将话题名称改为自己订阅的图像话题名称即可

    #回调函数
    def callback_raw_image(self, data):
        global begin,  image
        bridge = CvBridge()
        self.raw_image_opencv = bridge.imgmsg_to_cv2(data, "bgr8")
        image = self.raw_image_opencv
        self.raw_image = cv2.resize(self.raw_image_opencv, (640, 480))
        self.resulte.write(self.raw_image)
        #print(type(image))
        #self.captureNextFrame()
        #下面这项可作RGB三色显示使用
        #self.raw_image_opencv = cv2.cvtColor(self.raw_image_opencv, cv2.COLOR_BGR2RGB)
        print("已订阅到PC图像话题")
        self.raw_image_signal.emit()

    def captureNextFrame(self):
        self.raw_image = cv2.resize(self.raw_image_opencv, (640, 480))
        self.resulte.write(self.raw_image)

    def record(self):  #定义record函数，用于视频录制
        global vi_name, count
        #对视频设置的编码解码的方式MPEG-4编码
        self.fource=cv2.VideoWriter_fourcc(*'MJPG')
        #保存的位置，以及编码解码方式，帧率，视频帧大小
        video_name = '%s/%s%s.mp4' % (vi_name,'test',count)
        self.resulte=cv2.VideoWriter(video_name,self.fource,25.0,(640,480))

    def run(self):
        rospy.spin()    #订阅数据

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
        #开始视频录制
        cameraRosNode = CameraRosNode()
        #ameraRosNode.record()  
        #cameraRosNode.start()      


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
    #启动PC端订阅小车图像话题线程

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
