#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from std_msgs.msg import Int32
import os
import sys

def publish_animals(animal_list):
    rospy.init_node("animal_rescue_publisher", anonymous=True)
    rospy.sleep(1)
    rospy.loginfo("animal_list: %s", animal_list)
    animal_publisher = rospy.Publisher('/animal_unique', String, queue_size=10)
    
    num_rescues_pub = rospy.Publisher('/num_rescues', Int32, queue_size=10) 
    num_rescues_msg = Int32()
    num_rescues_msg.data = len(animal_list)
    #num_rescues_pub.publish(num_rescues_msg)

    rate = rospy.Rate(1) # 1 Hz
    while(True):
        num_rescues_pub.publish(num_rescues_msg)
        for animal in animal_list:
            animal_msg = String()
            animal_msg.data = animal
            rospy.loginfo("animal_msg: %s", animal_msg) 
            animal_publisher.publish(animal_msg)
        rate.sleep()

if __name__ == '__main__':
    publish_animals(sys.argv[1:])
    rospy.spin()
