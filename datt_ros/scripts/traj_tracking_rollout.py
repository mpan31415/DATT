#!/usr/bin/env python3

import rospy
from rospy.timer import TimerEvent
from std_msgs.msg import Bool

from utils.conversion import *

from DATT.configuration import (
    kolibri_tracking,
    kolibri_tracking_adaptive,
    kolibri_mppi,
    kolibri_pid,
)
from DATT.controllers import cntrl_config_presets
from DATT.controllers.datt_controller import DATTController
from DATT.controllers.mppi_controller import MPPIController
from DATT.controllers.pid_controller import PIDController
from DATT.refs import TrajectoryRef


class TrajTrackingRolloutNode:

    def __init__(self, quad_name="kolibri", ref_traj_name="circle", controller_type="pid"):

        self.dt = 0.02  # seconds
        ref_traj_name = "my_" + ref_traj_name + "_ref"

        # set position offset
        if ref_traj_name == "my_circle_ref":
            self.offset_pos = np.array([0.0, 0.0, 1.0])
        elif ref_traj_name == "my_fig8_ref":
            self.offset_pos = np.array([1.5, 0.0, 1.0])
        else:
            self.offset_pos = np.array([0.0, 0.0, 0.0])
        
        # subscribers
        self.state_sub = rospy.Subscriber(f"{quad_name}/agiros_pilot/state", QuadState, self.state_callback, queue_size=100, tcp_nodelay=True)
        self.activate_sub = rospy.Subscriber("activate_policy", Bool, self.activate_callback, queue_size=10, tcp_nodelay=True)

        # publishers
        self.control_pub = rospy.Publisher(f"{quad_name}/agiros_pilot/feedthrough_command", Command, queue_size=100, tcp_nodelay=True)
        self.policy_ready_pub = rospy.Publisher("policy_ready", Bool, queue_size=10)

        # flags
        self.ready = False
        self.activated = False

        # for tracking the quadrotor state
        self.state = StateStruct()

        # configs for creating the controller
        if controller_type == "datt":
            self.env_config = kolibri_tracking.config
            self.control_config = cntrl_config_presets.kolibri_tracking_config
            print("\033[93m[rollout]\033[0m Using standard DATT tracking config!")
        elif controller_type == "datt_adaptive":
            self.env_config = kolibri_tracking_adaptive.config
            self.control_config = cntrl_config_presets.kolibri_tracking_adaptive_config
            print("\033[93m[rollout]\033[0m Using adaptive DATT tracking config!")
        elif controller_type == "mpc":
            self.env_config = kolibri_mppi.config
            self.control_config = cntrl_config_presets.mppi_config
            print("\033[93m[rollout]\033[0m Using L1 MPC tracking config!")
        elif controller_type == "pid":
            self.env_config = kolibri_pid.config
            self.control_config = cntrl_config_presets.pid_config
            print("\033[93m[rollout]\033[0m Using PID tracking config!")
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")
            
        # create reference trajectory function
        ref_traj_obj = TrajectoryRef.get_by_value(ref_traj_name)
        ref_traj_func = ref_traj_obj.ref(self.env_config.ref_config)

        # create controller class
        if controller_type == "datt" or controller_type == "datt_adaptive":
            self.controller = DATTController(self.env_config, self.control_config)
        elif controller_type == "mpc":
            self.controller = MPPIController(self.env_config, self.control_config)
        elif controller_type == "pid":
            self.controller = PIDController(self.env_config, self.control_config)
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")
        
        print("\033[93m[rollout]\033[0m Created controller!")
        # set the reference trajectory function
        self.controller.ref_func = ref_traj_func
        self.controller.start_pos = np.zeros(3)

        # warmup the controller
        warmup_inputs = {
            't' : 0.1, 
            'state' : self.state, 
        }
        self.controller.response(**warmup_inputs)
        print("\033[93m[rollout]\033[0m Warmed up controller!")

        # initial countdown
        self.time_start = rospy.Time.now()
        self.time_till_ready = rospy.Duration(3)
        self.last_countdown = 11

        # timers
        print("\033[93m[rollout]\033[0m Starting timers ...")
        rospy.Timer(rospy.Duration(self.dt), self.command)
        rospy.Timer(rospy.Duration(self.dt), self.publish_policy_ready)
        rospy.Timer(rospy.Duration(0.002), self.print_countdown)
        rospy.Timer(rospy.Duration(self.dt), self.increment_time)


    def publish_policy_ready(self, event: TimerEvent):
        self.policy_ready_pub.publish(Bool(self.ready))

    def print_countdown(self, _event):
        now = rospy.Time.now().to_sec()
        left = (self.time_start + self.time_till_ready).to_sec() - now

        if int(left) + 1 < self.last_countdown and not self.ready:
            time = int(left) + 1
            print(f"Ready in {time} seconds")
            self.last_countdown = left

    def increment_time(self, _event: TimerEvent):
        if self.activated:
            self.state.t += self.dt
        else:
            self.state.t = 0.0


    def command(self, event: TimerEvent):
        
        # get observation from current state
        obs = self.get_obs_fn(self.state)

        # get action from policy
        norm_thrust, bodyrates = self.controller.response(**obs)

        # send command
        self.send_command_msg(norm_thrust, bodyrates)

        if (
            not self.ready
            and rospy.Time.now() - self.time_start > self.time_till_ready
        ):
            self.ready = True
            rospy.loginfo("Ready to send commands")
            print("Ready to send commands")


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

        # subtract the traj offset pos, since policy is trained to start tracking from zero
        pos -= self.offset_pos

        # update state struct
        self.state.pos = pos
        self.state.rot = R.from_quat(np.array([quat[1], quat[2], quat[3], quat[0]]))   # convert to x, y, z, w order
        self.state.vel = vel
        self.state.ang = omega


    def activate_callback(self, msg: Bool):
        use_policy = msg.data
        if use_policy and not self.activated:
            self.activated = True
            print("\033[93m[rollout]\033[0m ------ Activated policy ------")
        if not use_policy and self.activated:
            self.activated = False
            print("\033[93m[rollout]\033[0m ------ Deactivated policy ------")



##############################################################################
def main():

    rospy.init_node("datt_traj_tracking_rollout_node", anonymous=True)

    # get launch params
    quad_name = rospy.get_param("~quad_name", "kolibri")
    ref_traj_name = rospy.get_param("~ref_traj_name", "circle")
    controller_type = rospy.get_param("~controller_type", "pid")

    rospy.loginfo(f"quad_name: {quad_name}")
    rospy.loginfo(f"ref_traj_name: {ref_traj_name}")
    rospy.loginfo(f"controller_type: {controller_type}")

    # create node class
    node = TrajTrackingRolloutNode(quad_name=quad_name, ref_traj_name=ref_traj_name, controller_type=controller_type)
    rospy.spin()


if __name__ == "__main__":
    main()
