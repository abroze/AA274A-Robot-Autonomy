#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from std_msgs.msg import String
from std_msgs.msg import Int32
from asl_turtlebot.msg import DetectedObject
from visualization_msgs.msg import Marker
import tf
import numpy as np
from numpy import linalg
from utils.utils import wrapToPi
from utils.grids import StochOccupancyGrid2D
from planners import AStar, compute_smoothed_traj
import scipy.interpolate
import matplotlib.pyplot as plt
from controllers import PoseController, TrajectoryTracker, HeadingController
from enum import Enum
import random

from dynamic_reconfigure.server import Server
from asl_turtlebot.cfg import NavigatorConfig

# state machine modes, not all implemented
class Mode(Enum):
    IDLE = 0
    ALIGN = 1
    TRACK = 2
    PARK = 3
    TELEOP = 4
    PICKUP = 5
    STOP = 6


class Navigator:
    """
    This node handles point to point turtlebot motion, avoiding obstacles.
    It is the sole node that should publish to cmd_vel
    """

    def __init__(self):
        rospy.init_node("turtlebot_navigator", anonymous=True)
        self.mode = Mode.IDLE

        # initial state
        self.x_start = None
        self.y_start = None
        self.theta_start = None

        # current state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # detected animals
        self.detected_animals = {}

        # goal state
        self.x_g = 3.4
        self.y_g = 0.4
        self.theta_g = 3.14

        self.th_init = 0.0

        # map parameters
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0
        self.map_origin = [0, 0]
        self.map_probs = []
        self.occupancy = None
        self.occupancy_updated = False

        # plan parameters
        self.plan_resolution = 0.1
        self.plan_horizon = 4

        # time when we started following the plan
        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = 0
        self.plan_start = [0.0, 0.0]

        # Robot limits
        self.v_max = 0.3  # maximum velocity
        self.om_max = 1  # maximum angular velocity

        self.v_des = 0.2  # desired cruising velocity
        self.theta_start_thresh = 0.05  # threshold in theta to start moving forward when path-following
        self.start_pos_thresh = (
            0.2  # threshold to be far enough into the plan to recompute it
        )

        # threshold at which navigator switches from trajectory to pose control
        self.near_thresh = 0.2
        self.at_thresh = 0.05
        self.at_thresh_theta = 0.2

        # trajectory smoothing
        self.spline_alpha = 0
        self.spline_deg = 3  # cubic spline
        self.traj_dt = 0.1

        # trajectory tracking controller parameters
        self.kpx = 5
        self.kpy = 5
        self.kdx = 5
        self.kdy = 5

        # rescue parameters
        self.min_pickup_time = 3 #sec
        self.pickup_start = None
        self.rescue_phase = False
        self.num_rescue_targets = 0 
        self.rescue_targets = {} # animals to rescue (key = animal, value = position)
        self.rescue_target = None # current animal to rescue

        # stop sign params
        self.stop_sign_start = None
        self.stop_time = 5
        self.last_stop_sign = rospy.get_rostime()

        # teleop parameters
        self.V_teleop = 0.0
        self.om_teleop = 0.0

        # heading controller parameters
        self.kp_th = 1.0

        self.traj_controller = TrajectoryTracker(
            self.kpx, self.kpy, self.kdx, self.kdy, self.v_max, self.om_max
        )
        self.pose_controller = PoseController(
            0.0, 0.0, 0.0, self.v_max, self.om_max
        )
        self.heading_controller = HeadingController(self.kp_th, self.om_max)

        self.nav_planned_path_pub = rospy.Publisher(
            "/planned_path", Path, queue_size=10
        )
        self.nav_smoothed_path_pub = rospy.Publisher(
            "/cmd_smoothed_path", Path, queue_size=10
        )
        self.nav_smoothed_path_rej_pub = rospy.Publisher(
            "/cmd_smoothed_path_rejected", Path, queue_size=10
        )
        self.nav_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        self.animal_noise_pub = rospy.Publisher('/animal_noise', String, queue_size=10)

        self.marker_pub = rospy.Publisher('/marker_topic', Marker, queue_size=10)

        self.arrived_pub = rospy.Publisher('/arrived', String, queue_size=10)

        self.trans_listener = tf.TransformListener()

        self.cfg_srv = Server(NavigatorConfig, self.dyn_cfg_callback)

        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber("/map_metadata", MapMetaData, self.map_md_callback)
        rospy.Subscriber("/cmd_nav", Pose2D, self.cmd_nav_callback)
        rospy.Subscriber("/cmd_vel_teleop", Twist, self.cmd_vel_teleop_callback)
        rospy.Subscriber("/animal_unique", String, self.animal_rescue_callback)
        rospy.Subscriber('/detector/stop_sign', DetectedObject, self.stop_sign_callback)
        rospy.Subscriber('/num_rescues', Int32, self.num_rescues_callback)

        # Detector subscribers
        #rospy.Subscriber("/detector/bird", DetectedObject, self.bird_callback)
        rospy.Subscriber("/detector/cat", DetectedObject, self.cat_callback)
        rospy.Subscriber("/detector/dog", DetectedObject, self.dog_callback)
        #rospy.Subscriber("/detector/horse", DetectedObject, self.horse_callback)
        #rospy.Subscriber("/detector/sheep", DetectedObject, self.sheep_callback)
        rospy.Subscriber("/detector/cow", DetectedObject, self.cow_callback)
        rospy.Subscriber("/detector/elephant", DetectedObject, self.elephant_callback)
        #rospy.Subscriber("/detector/bear", DetectedObject, self.bear_callback)
        rospy.Subscriber("/detector/zebra", DetectedObject, self.zebra_callback)
        rospy.Subscriber("/detector/giraffe", DetectedObject, self.giraffe_callback)

        msg = String()
        msg.data = "arrived"
        self.arrived_pub.publish(msg)

        rospy.sleep(10)
        self.replan()
        print("finished init")

    def dyn_cfg_callback(self, config, level):
        rospy.loginfo(
            "Reconfigure Request: k1:{k1}, k2:{k2}, k3:{k3}".format(**config)
        )
        self.pose_controller.k1 = config["k1"]
        self.pose_controller.k2 = config["k2"]
        self.pose_controller.k3 = config["k3"]
        return config

    def cmd_nav_callback(self, data):
        """
        loads in goal if different from current goal, and replans
        """
        rospy.loginfo("New goal received")
        if (
            data.x != self.x_g
            or data.y != self.y_g
            or data.theta != self.theta_g
        ):
            rospy.logdebug(f"New command nav received:\n{data}")
            self.x_g = data.x
            self.y_g = data.y
            self.theta_g = data.theta
            self.replan()


    def cmd_vel_teleop_callback(self, data):
        self.V_teleop = data.linear.x
        self.om_teleop = data.angular.z
        self.mode = Mode.TELEOP


    def num_rescues_callback(self, msg):
        self.num_rescue_targets = msg.data

    def animal_rescue_callback(self, msg): # msg.data contains animal name (string)
        """
        populates rescue_targets dictionary with animals published from command line
        """
        if not self.rescue_phase: 
            animal_location = self.detected_animals[msg.data]
            self.rescue_targets[msg.data] = (animal_location['x'], animal_location['y'], animal_location['th'])
            rospy.loginfo("ADDING %s to RESCUE TARGETS", msg.data)
            rospy.loginfo("Number of rescue targets: %s", self.num_rescue_targets)
            rospy.loginfo("Length of rescue_targets: %s", len(self.rescue_targets))
            if len(self.rescue_targets) == self.num_rescue_targets:
                self.init_rescue()


    def stop_sign_callback(self, msg):
        if rospy.get_rostime() - self.last_stop_sign > rospy.Duration.from_sec(60):
            self.init_stop_sign()


    def map_md_callback(self, msg):
        """
        receives maps meta data and stores it
        """
        self.map_width = msg.width
        self.map_height = msg.height
        self.map_resolution = msg.resolution
        self.map_origin = (msg.origin.position.x, msg.origin.position.y)

    def map_callback(self, msg):
        """
        receives new map info and updates the map
        """
        self.map_probs = msg.data
        # if we've received the map metadata and have a way to update it:
        if (
            self.map_width > 0
            and self.map_height > 0
            and len(self.map_probs) > 0
        ):
            self.occupancy = StochOccupancyGrid2D(
                self.map_resolution,
                self.map_width,
                self.map_height,
                self.map_origin[0],
                self.map_origin[1],
                7,
                self.map_probs,
            )
            if self.x_g is not None:
                # if we have a goal to plan to, replan
                rospy.loginfo("replanning because of new map")
                self.replan()  # new map, need to replan

    def shutdown_callback(self):
        """
        publishes zero velocities upon rospy shutdown
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.nav_vel_pub.publish(cmd_vel)

    def detected_animal(self, animal, object_msg, marker_id):
        if not self.rescue_phase:
            if self.occupancy.is_free((self.x, self.y)):
                if animal not in self.detected_animals or self.detected_animals[animal]['conf'] < object_msg.confidence:
                    self.detected_animals[animal] = {'x': self.x, 'y': self.y, 'th': self.theta, 'conf': object_msg.confidence}
                    rospy.loginfo("Detected a %s. Currently detected animals: %s", animal, self.detected_animals.keys()) 
                    self.publish_marker(marker_id)

    def publish_marker(self, marker_id):
        marker = Marker()
        marker.header.frame_id = "base_footprint"
        #marker.frame_locked = False
        marker.header.stamp = rospy.Time()
        marker.id = marker_id
        rospy.loginfo("MARKER_ID: %d", marker.id)
        marker.type = 2 # sphere
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 0.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.marker_pub.publish(marker)
        rospy.loginfo('Published marker!')

    def bird_callback(self, object_msg):
       self.detected_animal('bird', object_msg)

    def cat_callback(self, object_msg):
        self.detected_animal('cat', object_msg, 0)
        # publish meow
        meow_msg = String()
        meow_msg.data = "MEOW"
        self.animal_noise_pub.publish(meow_msg)

    def dog_callback(self, object_msg):
        self.detected_animal('dog', object_msg, 1)
        # publish woof
        woof_msg = String()
        woof_msg.data = "WOOF"
        self.animal_noise_pub.publish(woof_msg)

    def horse_callback(self, object_msg):
        self.detected_animal('horse', object_msg)

    def sheep_callback(self, object_msg):
        self.detected_animal('sheep', object_msg)

    def cow_callback(self, object_msg):
        self.detected_animal('cow', object_msg, 2)

    def elephant_callback(self, object_msg):
        self.detected_animal('elephant', object_msg, 3)

    def bear_callback(self, object_msg):
        self.detected_animal('bear', object_msg)

    def zebra_callback(self, object_msg):
        self.detected_animal('zebra', object_msg, 4)

    def giraffe_callback(self, object_msg):
        self.detected_animal('giraffe', object_msg, 5)


    def near_goal(self):
        """
        returns whether the robot is close enough in position to the goal to
        start using the pose controller
        """
        return (
            linalg.norm(np.array([self.x - self.x_g, self.y - self.y_g]))
            < self.near_thresh
        )

    def at_goal(self):
        """
        returns whether the robot has reached the goal position with enough
        accuracy to return to idle state
        """
        return (
            linalg.norm(np.array([self.x - self.x_g, self.y - self.y_g]))
            < self.at_thresh
            and abs(wrapToPi(self.theta - self.theta_g)) < self.at_thresh_theta
        )

    def aligned(self):
        """
        returns whether robot is aligned with starting direction of path
        (enough to switch to tracking controller)
        """
        return (
            abs(wrapToPi(self.theta - self.th_init)) < self.theta_start_thresh
        )

    def close_to_plan_start(self):
        return (
            abs(self.x - self.plan_start[0]) < self.start_pos_thresh
            and abs(self.y - self.plan_start[1]) < self.start_pos_thresh
        )

    def snap_to_grid(self, x):
        return (
            self.plan_resolution * round(x[0] / self.plan_resolution),
            self.plan_resolution * round(x[1] / self.plan_resolution),
        )

    def init_rescue(self):
        self.rescue_phase = True
        rospy.loginfo("RESCUE PHASE")
        self.TSP()
        rospy.loginfo("Rescue targets sorted: %s", self.rescue_targets)
        self.rescue_target = self.sorted_rescue_targets[0]
        #self.rescue_target = random.choice(list(self.rescue_targets.keys())) # choose first rescue target randomly
        rospy.loginfo("NEXT RESCUE TARGET: %s", self.rescue_target)
        self.x_g, self.y_g, self.theta_g = self.rescue_targets[self.rescue_target]
        self.replan()

    def init_stop_sign(self):
        self.stop_sign_start = rospy.get_rostime()
        self.last_stop_sign = rospy.get_rostime()
        self.switch_mode(Mode.STOP)

    def has_stopped(self):
        """ checks if robot stops at stop sign for 3 sec """

        return rospy.get_rostime() - self.stop_sign_start > rospy.Duration.from_sec(self.stop_time)

    def TSP(self):
        rospy.loginfo("Running TSP")

        coordinates = {0: (self.x_start, self.y_start, self.theta_start)}
        num_to_animal = {}
        num = 1
        for key, value in self.rescue_targets.items():
            num_to_animal[num] = key
            coordinates[num] = (value[0], value[1], value[2])
            num += 1

        rospy.loginfo("Coordinates: %s", coordinates)
        rospy.loginfo("Num to animal: %s", num_to_animal)


        if len(self.rescue_targets) == 1 or len(self.rescue_targets) == 2:
            sorted_rescue_targets = list(self.rescue_targets.keys())
            self.sorted_rescue_targets = sorted_rescue_targets
            return
        elif len(self.rescue_targets) == 3:
            combinations = [[0,1,2,3,0], [0,1,3,2,0], [0,2,1,3,0]]
        elif len(self.rescue_targets) == 4:
            combinations = [[0,1,2,3,4,0],
                            [0,1,2,4,3,0],
                            [0,1,3,2,4,0],
                            [0,1,3,4,2,0],
                            [0,1,4,2,3,0],
                            [0,1,4,3,2,0],
                            [0,2,1,3,4,0],
                            [0,2,1,4,3,0],
                            [0,2,3,1,4,0],
                            [0,2,4,1,3,0],
                            [0,3,1,2,4,0]]

        min_length = np.inf
        min_combination = combinations[0]
        for combination in combinations:
            length_planned_path = 0
            for i in range(len(combination)-1):
                x_init = self.snap_to_grid((coordinates[combination[i]][0], coordinates[combination[i]][1]))
                x_goal = self.snap_to_grid((coordinates[combination[i+1]][0], coordinates[combination[i+1]][1]))
                problem = AStar(self.snap_to_grid((0, 0)), self.snap_to_grid((self.plan_horizon, self.plan_horizon)), x_init, x_goal, self.occupancy, self.plan_resolution)

                success = problem.solve()
                if not success:
                    rospy.loginfo("A* failed in TSP")
                    length_planned_path += np.inf
                else:
                    rospy.loginfo("A* succeded in TSP")
                    planned_path = problem.path
                    length_planned_path += len(planned_path)

            rospy.loginfo("Length of path: %s", length_planned_path)
            if length_planned_path < min_length:
                min_length = length_planned_path
                min_combination = combination

        rospy.loginfo("Shortest combination: %s", min_combination)


        sorted_rescue_targets = []
        for j in range(1,len(min_combination)-1):
            animal_num = min_combination[j]
            sorted_rescue_targets.append(num_to_animal[animal_num])

        self.sorted_rescue_targets = sorted_rescue_targets


    def has_picked_up(self):
        """ checks if animal has been picked up (robot has stopped for 3-5s) """
        if self.mode == Mode.PICKUP and rospy.get_rostime() - self.pickup_start >= rospy.Duration.from_sec(self.min_pickup_time):
            rospy.loginfo("STOPPED > 3 SEC")
            return True
        else:
            return False

    def switch_mode(self, new_mode):
        rospy.loginfo("Switching from %s -> %s", self.mode, new_mode)
        self.mode = new_mode

    def publish_planned_path(self, path, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = "map"
        for state in path:
            pose_st = PoseStamped()
            pose_st.pose.position.x = state[0]
            pose_st.pose.position.y = state[1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = "map"
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_smoothed_path(self, traj, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = "map"
        for i in range(traj.shape[0]):
            pose_st = PoseStamped()
            pose_st.pose.position.x = traj[i, 0]
            pose_st.pose.position.y = traj[i, 1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = "map"
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_control(self):
        """
        Runs appropriate controller depending on the mode. Assumes all controllers
        are all properly set up / with the correct goals loaded
        """
        t = self.get_current_plan_time()

        if self.mode == Mode.PARK:
            V, om = self.pose_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        elif self.mode == Mode.TRACK:
            V, om = self.traj_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        elif self.mode == Mode.ALIGN:
            V, om = self.heading_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        elif self.mode == Mode.TELEOP:
            V = self.V_teleop
            om = self.om_teleop 
        else:
            V = 0.0
            om = 0.0

        cmd_vel = Twist()
        cmd_vel.linear.x = V
        cmd_vel.angular.z = om
        self.nav_vel_pub.publish(cmd_vel)

    def get_current_plan_time(self):
        t = (rospy.get_rostime() - self.current_plan_start_time).to_sec()
        return max(0.0, t)  # clip negative time to 0

    def replan(self):
        """
        loads goal into pose controller
        runs planner based on current pose
        if plan long enough to track:
            smooths resulting traj, loads it into traj_controller
            sets self.current_plan_start_time
            sets mode to ALIGN
        else:
            sets mode to PARK
        """
        # Make sure we have a map
        if not self.occupancy:
            rospy.loginfo(
                "Navigator: replanning canceled, waiting for occupancy map."
            )
            self.switch_mode(Mode.IDLE)
            return

        # Attempt to plan a path
        state_min = self.snap_to_grid((0, 0))
        state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
        x_init = self.snap_to_grid((self.x, self.y))
        self.plan_start = x_init
        x_goal = self.snap_to_grid((self.x_g, self.y_g))
        problem = AStar(
            state_min,
            state_max,
            x_init,
            x_goal,
            self.occupancy,
            self.plan_resolution,
        )

        rospy.loginfo("Navigator: computing navigation plan")
        success = problem.solve()
        if not success:
            rospy.loginfo("Planning failed")
            #problem.plot_tree()
            return

        rospy.loginfo("Planning Succeeded")

        planned_path = problem.path

        # Check whether path is too short
        if len(planned_path) < 4:
            rospy.loginfo("Path too short to track")
            self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
            self.switch_mode(Mode.PARK)
            return

        # Smooth and generate a trajectory
        t_new, traj_new = compute_smoothed_traj(
            planned_path, self.v_des, self.spline_deg, self.spline_alpha, self.traj_dt
        )

        # If currently tracking a trajectory, check whether new trajectory will take more time to follow
        if self.mode == Mode.TRACK:
            t_remaining_curr = (
                self.current_plan_duration - self.get_current_plan_time()
            )

            # Estimate duration of new trajectory
            th_init_new = traj_new[0, 2]
            th_err = wrapToPi(th_init_new - self.theta)
            t_init_align = abs(th_err / self.om_max)
            t_remaining_new = t_init_align + t_new[-1]

            if t_remaining_new > t_remaining_curr:
                rospy.loginfo(
                    "New plan rejected (longer duration than current plan)"
                )
                self.publish_smoothed_path(
                    traj_new, self.nav_smoothed_path_rej_pub
                )
                return

        # Otherwise follow the new plan
        self.publish_planned_path(planned_path, self.nav_planned_path_pub)
        self.publish_smoothed_path(traj_new, self.nav_smoothed_path_pub)

        self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
        self.traj_controller.load_traj(t_new, traj_new)

        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = t_new[-1]

        self.th_init = traj_new[0, 2]
        self.heading_controller.load_goal(self.th_init)

        if not self.aligned():
            rospy.loginfo("Not aligned with start direction")
            self.switch_mode(Mode.ALIGN)
            return

        rospy.loginfo("Ready to track")
        self.switch_mode(Mode.TRACK)

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # try to get state information to update self.x, self.y, self.theta
            try:
                (translation, rotation) = self.trans_listener.lookupTransform(
                    "/map", "/base_footprint", rospy.Time(0)
                )
                euler = tf.transformations.euler_from_quaternion(rotation)
                if self.x_start is None:
                    self.x_start = translation[0]
                    self.y_start = translation[1]
                    self.theta_start = euler[2]
                self.x = translation[0]
                self.y = translation[1]
                self.theta = euler[2]
            except (
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ) as e:
                self.current_plan = []
                rospy.loginfo("Navigator: waiting for state info")
                if self.mode != Mode.TELEOP:
                    self.switch_mode(Mode.IDLE)
                print(e)
                pass

            # STATE MACHINE LOGIC
            # some transitions handled by callbacks
            if self.mode == Mode.IDLE:
                pass
            elif self.mode == Mode.ALIGN:
                if self.aligned():
                    self.current_plan_start_time = rospy.get_rostime()
                    self.switch_mode(Mode.TRACK)
            elif self.mode == Mode.TRACK:
                if self.near_goal():
                    self.switch_mode(Mode.PARK)
                elif not self.close_to_plan_start():
                    rospy.loginfo("replanning because far from start")
                    self.replan()
                elif (
                    rospy.get_rostime() - self.current_plan_start_time
                ).to_sec() > self.current_plan_duration:
                    rospy.loginfo("replanning because out of time")
                    self.replan()  # we aren't near the goal but we thought we should have been, so replan
            elif self.mode == Mode.PARK:
                if self.at_goal():
                    # forget about goal:
                    self.x_g = None
                    self.y_g = None
                    self.theta_g = None
                    if not self.rescue_phase:
                        msg = String()
                        msg.data = "arrived"
                        self.arrived_pub.publish(msg)
                    if self.rescue_phase:
                        rospy.loginfo("STARTING PICKUP")
                        self.pickup_start = rospy.get_rostime()
                        self.switch_mode(Mode.PICKUP)
                    else:
                        # forget about goal:
                        self.switch_mode(Mode.IDLE)
            elif self.mode == Mode.PICKUP:
                # pick up animal
                #if not self.at_goal():
                    #self.replan()
                    #rospy.loginfo("Stuck trying to find path")
                if self.has_picked_up(): # robot has stopped for 3-5 sec
                    rospy.loginfo("PICKUP COMPLETE")
                    self.sorted_rescue_targets.remove(self.rescue_target)
                    if self.sorted_rescue_targets: # still more animals to rescue
                        self.rescue_target = self.sorted_rescue_targets[0]
                        rospy.loginfo("NEXT PICKUP TARGET: %s", self.rescue_target)
                        self.x_g, self.y_g, self.theta_g = self.rescue_targets[self.rescue_target]
                    else:
                        # return to start
                        rospy.loginfo("DONE RESCUING")
                        self.x_g = self.x_start
                        self.y_g = self.y_start
                        self.theta_g = self.theta_start
                    self.replan()
            elif self.mode == Mode.STOP:
                # stop at stop sign
                if self.has_stopped():
                    self.replan()

            self.publish_control()
            rate.sleep()


if __name__ == "__main__":
    nav = Navigator()
    rospy.on_shutdown(nav.shutdown_callback)
    nav.run()
