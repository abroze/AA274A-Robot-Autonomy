#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose2D, PoseStamped, Point
from std_msgs.msg import String
import tf

# if using gmapping, you will have a map frame. otherwise it will be odom frame
mapping = True

class GoalPoseCommander:

    def __init__(self):
        rospy.init_node('goal_pose_commander', anonymous=True)
        # initialize variables
        self.x_g = None
        self.y_g = None
        self.theta_g = None
        self.goal_pose_received = False
        self.trans_listener = tf.TransformListener()
        self.start_time = rospy.get_rostime()

        # waypoints for exploration
        self.waypoints = [[3.4,1.6,3.14], [3.4,0.4,3.14], [2.3,0.4,3.14], [0.3,0.4,1.57], [2.3,0.4,0], [2.4,1.6,1.57], [2.2,2.8,0], [3.4,2.9,0], [3.4,1.6,3.14]] 
        self.current_waypoint = 1
        # command pose for controller
        self.nav_goal_publisher = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.rviz_goal_callback)

        rospy.Subscriber('/arrived', String, self.arrived_callback)


    def rviz_goal_callback(self, msg):
        """ callback for a pose goal sent through rviz """
        rospy.loginfo("rviz command received!")
        try:
            origin_frame = "/map" if mapping else "/odom"
            rospy.loginfo("getting frame")
            nav_pose_origin = self.trans_listener.transformPose(origin_frame, msg)
            self.x_g = nav_pose_origin.pose.position.x
            self.y_g = nav_pose_origin.pose.position.y
            quaternion = (
                    nav_pose_origin.pose.orientation.x,
                    nav_pose_origin.pose.orientation.y,
                    nav_pose_origin.pose.orientation.z,
                    nav_pose_origin.pose.orientation.w)
            euler = tf.transformations.euler_from_quaternion(quaternion)
            self.theta_g = euler[2]
            self.start_time = rospy.get_rostime()

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

    def arrived_callback(self, msg):
        if self.current_waypoint < len(self.waypoints):

            pose_g_msg = Pose2D()
            self.x_g = self.waypoints[self.current_waypoint][0]
            self.y_g = self.waypoints[self.current_waypoint][1]
            self.theta_g = self.waypoints[self.current_waypoint][2]
            self.current_waypoint += 1

            pose_g_msg.x = self.x_g
            pose_g_msg.y = self.y_g
            pose_g_msg.theta = self.theta_g
            self.nav_goal_publisher.publish(pose_g_msg)



    def publish_goal_pose(self):
        """ sends the current desired pose to the navigator """
        pose_g_msg = Pose2D()
        self.x_g = self.waypoints[0][0]
        self.y_g = self.waypoints[0][1]
        self.theta_g = self.waypoints[0][2]
        pose_g_msg.x = self.x_g
        pose_g_msg.y = self.y_g
        pose_g_msg.theta = self.theta_g 

        rospy.loginfo("Publishing first goalpose")
        self.nav_goal_publisher.publish(pose_g_msg)


        """ for i in range(len(self.waypoints)):
            self.x_g = self.waypoints[i][0]
            self.y_g = self.waypoints[i][1]
            self.theta_g = self.waypoints[i][2]

            #if self.x_g is not None:
                pose_g_msg = Pose2D()
                pose_g_msg.x = self.x_g
                pose_g_msg.y = self.y_g
                pose_g_msg.theta = self.theta_g
                self.nav_goal_publisher.publish(pose_g_msg)
                if i < 1:
                    rospy.sleep(10)
                else:
                    rospy.sleep(20)
        """

    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            t = rospy.get_rostime()
            rospy.loginfo((t-self.start_time).to_sec())
            if (t - self.start_time).to_sec() > 60.0 and self.x_g == None:
                self.publish_goal_pose()
            rate.sleep()


if __name__ == '__main__':
    sup = GoalPoseCommander()
    try:
        sup.loop()
    except rospy.ROSInterruptException:
        pass        
    
