#!/usr/bin/env python3

import rospy
from rospy.timer import TimerEvent
from std_msgs.msg import Bool

from utils.conversion import *

from DATT.configuration import (
    kolibri_hover,
    kolibri_hover_adaptive,
    kolibri_tracking,
    kolibri_tracking_adaptive,
)
from DATT.controllers import cntrl_config_presets
from DATT.controllers.datt_controller import DATTController

from time import time


class OnlineStateHoverRolloutNode:

    def __init__(self, quad_name="kolibri", target_pos=np.array([0.0, 0.0, 1.0])):

        self.dt = 0.02  # seconds
        self.target_pos = target_pos

        # TODO: add adaptive option
        
        # subscribers
        self.state_sub = rospy.Subscriber(f"{quad_name}/agiros_pilot/state", QuadState, self.state_callback, queue_size=100, tcp_nodelay=True)
        self.activate_sub = rospy.Subscriber("activate_policy", Bool, self.activate_callback, queue_size=10, tcp_nodelay=True)

        # policy activation flag
        self.activated = False

        # publishers
        self.control_pub = rospy.Publisher(f"{quad_name}/agiros_pilot/feedthrough_command", Command, queue_size=100, tcp_nodelay=True)

        # for tracking the quadrotor state
        self.state = StateStruct()

        # configs for creating the controller
        self.datt_config = kolibri_hover.config
        self.control_config = cntrl_config_presets.kolibri_hover_config

        # create controller class
        self.controller = DATTController(self.datt_config, self.control_config)
        print("\033[93m[rollout]\033[0m Created DATT controller!")
        # warmup the controller
        warmup_inputs = {
            't' : 0.1, 
            'state' : self.state, 
        }
        self.controller.response(**warmup_inputs)
        print("\033[93m[rollout]\033[0m Warmed up DATT controller!")

        # timers
        print("\033[93m[rollout]\033[0m Starting timers ...")
        rospy.Timer(rospy.Duration(self.dt), self.command)

        # sleep function
        tic = time()
        counter = 0
        while time() - tic < 10.0:
            rospy.sleep(1.0)
            counter += 1
            print("counter: ", counter)
        self.activated = True


    def command(self, event: TimerEvent):

        if not self.activated:
            return
        
        # get observation from current state
        obs = self.get_obs_fn(self.state)

        # get action from policy
        norm_thrust, bodyrates = self.controller.response(**obs)

        # send command
        self.send_command_msg(norm_thrust, bodyrates)


    def get_obs_fn(self, state):
        obs = {
            't': state.t,
            'state': state,
        }
        return obs


    def send_command_msg(self, thrust, bodyrates):
        msg = commands_to_msg(thrust, bodyrates)
        if self.activated:
            self.control_pub.publish(msg)


    def state_callback(self, msg):

        pos, quat, vel, omega = quad_state_from_msg(msg)      # note: quat is in w, x, y, z order

        # subtract the hovering target position, since policy is trained to hover at zero position
        pos -= self.target_pos

        # update state struct
        self.state.pos = pos
        self.state.rot = R.from_quat(np.array([quat[1], quat[2], quat[3], quat[0]]))   # convert to x, y, z, w order
        self.state.vel = vel
        self.state.ang = omega

        # increment time only if activated
        if self.activated:
            self.state.t += self.dt


    def activate_callback(self, msg: Bool):
        use_policy = msg.data
        if use_policy and not self.activated:
            self.activated = True
            # IMPORTANT: reset time to zero
            self.state.t = 0.0
            print("\033[93m[rollout]\033[0m ------ Activated policy ------")
        if not use_policy and self.activated:
            self.activated = False
            print("\033[93m[rollout]\033[0m ------ Deactivated policy ------")



##############################################################################
def main():

    rospy.init_node("datt_state_hover_rollout_node", anonymous=True)

    # get launch params
    quad_name = rospy.get_param("~quad_name", "kolibri")
    target_pos = rospy.get_param("~target_pos", "0.0, 0.0, 1.0")
    target_pos = [float(x) for x in target_pos.split(",")]

    rospy.loginfo(f"quad_name: {quad_name}")
    rospy.loginfo(f"target_pos: {target_pos}")

    # create node class
    node = OnlineStateHoverRolloutNode(quad_name=quad_name, target_pos=target_pos)
    rospy.spin()


if __name__ == "__main__":
    main()
