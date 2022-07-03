#!/usr/bin/env python3

import rospy
from std_srvs.srv import Trigger
from spot_msgs.srv import SetLocomotion, SetLocomotionRequest

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

def set_locomotion_mode():
    """Calls spot service to set locomotion mode to amble.
    """
    topic = "spot/locomotion_mode"
    rospy.wait_for_service(topic)
    try:
        loco = rospy.ServiceProxy(topic, SetLocomotion)
        loco_req = SetLocomotionRequest(2)
        loco_resp = loco(loco_req)
        print(f"Service call successful with response: {loco_resp}")
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")

if __name__=="__main__":
    print("Starting spot startup process")
    spot_claim_topic = "spot/claim"
    spot_power_on_topic = "spot/power_on"
    spot_stand_topic = "spot/stand"
    call_trigger_service(spot_claim_topic)
    call_trigger_service(spot_power_on_topic)
    call_trigger_service(spot_stand_topic)
    set_locomotion_mode()