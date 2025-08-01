#!/usr/bin/env python3

import rospy
import numpy as np

from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped
from rospy.timer import TimerEvent

from utils.hovering_targets import generate_hovering_targets, HoveringTargets


class HoveringSchedulerNode:

    def __init__(self, quad_name="kolibri", target_pos=np.array([0.0, 0.0, 1.0])):

        # policy ready flag subscriber
        self.policy_ready_sub = rospy.Subscriber("policy_ready", Bool, self.policy_ready_callback)

        # target pose publisher
        self.target_pose_pub = rospy.Publisher(quad_name + '/agiros_pilot/go_to_pose', PoseStamped, queue_size=10)
        self.target_pose_msg = PoseStamped()

        # policy activation publisher
        self.policy_activation_pub = rospy.Publisher('/activate_policy', Bool, queue_size=1)

        # target progress publisher
        self.target_progress_pub = rospy.Publisher('/target_progress', String, queue_size=1)

        # data collection variables
        OFFSET = target_pos - np.array([0.0, 0.0, 1.0])         # offset to the target position
        targets_type = HoveringTargets.GRID_2x2x2_1m
        self.target_poses = generate_hovering_targets(targets_type, offset=OFFSET)
        print("Generated a total of ", len(self.target_poses), " poses")

        self.current_pose_index = 0
        self.all_poses_done = False

        self.policy_ready = False
        self.use_policy_flag = False

        # times
        self.reach_target_time = 3
        self.policy_time = 10
        self.wait_time = 1
        target_pose_pub_period = self.reach_target_time + self.policy_time + self.wait_time
        rospy.Timer(rospy.Duration(0.01), self.publish_policy_activation)
        rospy.Timer(rospy.Duration(target_pose_pub_period), self.publish_target_pose)

        print("Waiting for policy to be ready...")

    def policy_ready_callback(self, msg: Bool):
        self.policy_ready = msg.data
        if not self.policy_ready and msg.data:
            rospy.loginfo("Policy is ready to be activated.")

    def publish_policy_activation(self, event: TimerEvent):
        self.policy_activation_pub.publish(Bool(self.use_policy_flag))
    
    def publish_target_progress(self, pose_idx, state):
        msg_string = f"pose {pose_idx} {state}"
        self.target_progress_pub.publish(String(msg_string))


    def publish_target_pose(self, event: TimerEvent):

        if not self.policy_ready or self.all_poses_done:
            return

        # publish target pose
        pose = self.target_poses[self.current_pose_index]
        self.target_pose_msg.pose.position.x = pose[0]
        self.target_pose_msg.pose.position.y = pose[1]
        self.target_pose_msg.pose.position.z = pose[2]
        self.target_pose_pub.publish(self.target_pose_msg)
        print(f"Published target pose #{self.current_pose_index+1}/{len(self.target_poses)}: {pose}")
        
        rospy.sleep(self.reach_target_time)

        # activate policy
        self.use_policy_flag = True
        self.publish_target_progress(self.current_pose_index, "start")

        rospy.sleep(self.policy_time)

        # deactivate policy
        self.use_policy_flag = False
        self.publish_target_progress(self.current_pose_index, "end")

        # increment target pose index
        self.current_pose_index += 1
        if self.current_pose_index == len(self.target_poses):
            rospy.loginfo("All poses published, stopping policy activation")
            self.all_poses_done = True



if __name__ == "__main__":

    rospy.init_node('hovering_scheduler_node', anonymous=True)

    # get launch params
    quad_name = rospy.get_param("~quad_name")
    target_pos = rospy.get_param("~target_pos", "0.0, 0.0, 1.0")
    target_pos = [float(x) for x in target_pos.split(",")]

    rospy.loginfo(f"quad_name: {quad_name}")
    rospy.loginfo(f"target_pos: {target_pos}")

    node = HoveringSchedulerNode(quad_name=quad_name, target_pos=target_pos)
    rospy.spin()
