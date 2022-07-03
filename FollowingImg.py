import rospy
import cv2
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

def Followcallback(data):
    cv_img = bridge.imgmsg_to_cv2(data, "bgr8")
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow("RealSense", cv_img)
    cv2.waitKey(1)

if __name__ == '__main__':
    bridge = CvBridge()
    rospy.init_node("FOllowingImg")
    rospy.Subscriber('/Realsense/FollowingImg',Image, Followcallback)
    rospy.spin()
