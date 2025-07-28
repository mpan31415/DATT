#!/usr/bin/env python3

import rospy
import numpy as np

from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from rospy.timer import TimerEvent


class TrajTrackingSchedulerNode:

    def __init__(self, quad_name="kolibri", ref_traj_name="line"):

        # policy ready flag subscriber
        self.policy_ready_sub = rospy.Subscriber("policy_ready", Bool, self.policy_ready_callback)

        # start pose publisher
        self.start_pos_pub = rospy.Publisher(quad_name + '/agiros_pilot/go_to_pose', PoseStamped, queue_size=10)
        self.start_pos_msg = PoseStamped()

        # policy activation publisher
        self.policy_activation_pub = rospy.Publisher('/activate_policy', Bool, queue_size=1)

        self.policy_ready = False
        self.use_policy_flag = False

        # define start position
        if ref_traj_name == "circle":
            self.start_pos = np.array([0.0, 0.0, 1.0])
        elif ref_traj_name == "fig8":
            self.start_pos = np.array([1.5, 0.0, 1.0])
        else:
            self.start_pos = np.array([0.0, 0.0, 0.0])

        # define time parameters
        self.init_sleep_time = 10.0
        self.reach_start_pos_time = 3.0                                 # time to reach start position
        self.policy_rollout_time = 60.0                                 # run policy for the length of the trajectory
        self.wait_time = 1.0
        rospy.Timer(rospy.Duration(0.01), self.publish_policy_activation)

        # main function
        self.run()


    def publish_start_pos(self):
        pos = self.start_pos
        # position
        self.start_pos_msg.pose.position.x = pos[0]
        self.start_pos_msg.pose.position.y = pos[1]
        self.start_pos_msg.pose.position.z = pos[2]
        # orientation (facing x axis)
        self.start_pos_msg.pose.orientation.x = 0.0
        self.start_pos_msg.pose.orientation.y = 0.0
        self.start_pos_msg.pose.orientation.z = 0.0
        self.start_pos_msg.pose.orientation.w = 1.0
        # publish message
        self.start_pos_pub.publish(self.start_pos_msg)
        print(f"Published start pos: {pos}")

    def policy_ready_callback(self, msg: Bool):
        self.policy_ready = msg.data
        if not self.policy_ready and msg.data:
            rospy.loginfo("Policy is ready to be activated.")

    def publish_policy_activation(self, event: TimerEvent):
        self.policy_activation_pub.publish(Bool(self.use_policy_flag))


    def run(self):

        rospy.sleep(self.init_sleep_time)

        self.publish_start_pos()
        
        rospy.sleep(self.reach_start_pos_time)

        # activate policy
        self.use_policy_flag = True

        # run policy for a fixed time
        rospy.sleep(self.policy_rollout_time)

        # deactivate policy
        self.use_policy_flag = False

        # finish
        rospy.loginfo("Finished! Stopping policy")



if __name__ == "__main__":

    rospy.init_node('traj_tracking_scheduler_node', anonymous=True)

    # get launch params
    quad_name = rospy.get_param("~quad_name")
    ref_traj_name = rospy.get_param("~ref_traj_name", "circle")

    rospy.loginfo(f"quad_name: {quad_name}")
    rospy.loginfo(f"ref_traj_name: {ref_traj_name}")
    
    node = TrajTrackingSchedulerNode(quad_name=quad_name, ref_traj_name=ref_traj_name)
    rospy.spin()
