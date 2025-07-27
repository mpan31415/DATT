#!/usr/bin/env python3

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.05"

import base64
from rospy.timer import TimerEvent
from std_msgs.msg import Bool, String

from flax import serialization

from flightning.envs import HoveringStateEnv
from flightning.envs.hovering_state_env import EnvState
from flightning.utils.math import normalize
from flightning_ros.scripts.utils.conversion import *
from flightning_ros.scripts.utils.loading import (
    get_state_hover_policy_info,
    load_policy_fn_and_params,
)
from flightning.objects import Quadrotor


# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# use only cpu for jax
# jax.config.update("jax_platforms", "cpu")

KEY = jax.random.key(0)
CPU_DEVICE = jax.devices("cpu")[0]


class OnlineStateHoverRolloutNode:

    def __init__(self, env: HoveringStateEnv, policy_fn, params_template, real_quad="kolibri"):

        # assign env, and reset state variable using env init function
        self.env = env
        self.state: EnvState = self.env.reset(KEY)[0]

        # policy function and loaded policy template
        self.policy_fn = policy_fn
        self.params_template = params_template
        
        # subscribers
        self.state_sub = rospy.Subscriber(f"{real_quad}/agiros_pilot/state", QuadState, self.state_callback, queue_size=100, tcp_nodelay=True)
        self.policy_params_sub = rospy.Subscriber("/policy_params", String, self.policy_params_callback, queue_size=1, tcp_nodelay=True)
        self.activate_sub = rospy.Subscriber("activate_policy", Bool, self.activate_callback, queue_size=10, tcp_nodelay=True)

        # publishers
        self.control_pub = rospy.Publisher(f"{real_quad}/agiros_pilot/feedthrough_command", Command, queue_size=100, tcp_nodelay=True)
        self.policy_ready_pub = rospy.Publisher("policy_ready", Bool, queue_size=10, tcp_nodelay=True)

        # other variables
        self.policy_params = None

        # flag variables
        self.ready = False
        self.activated = False

        # timers
        print("\033[93m[rollout]\033[0m Starting timers ...")
        rospy.Timer(rospy.Duration(env.dt), self.command)
        rospy.Timer(rospy.Duration(env.dt), self.publish_policy_ready)

    
    def policy_params_callback(self, msg: String):
        received_bytes = base64.b64decode(msg.data)
        self.policy_params = serialization.from_bytes(self.params_template, received_bytes)
        # rospy.loginfo("[tracking rollout] Received new policy parameters!")


    def publish_policy_ready(self, event: TimerEvent):
        self.policy_ready_pub.publish(Bool(self.ready))


    def command(self, event: TimerEvent):

        if self.policy_params is None:
            return
        
        if not self.ready:
            self.ready = True
            print("\033[93m[rollout]\033[0m Initial policy params received, Ready to send commands!")
        
        # get observation from current state
        obs = self.get_obs_fn(self.state)

        # get action from policy
        obs_cpu = jax.device_put(obs, CPU_DEVICE)
        params_cpu = jax.device_put(self.policy_params, CPU_DEVICE)
        action = self.policy_fn(params_cpu, obs_cpu)

        # clip action and send command
        action = self.clip_action(action)
        thrust, bodyrates = action[0], action[1:]
        self.send_command_msg(thrust, bodyrates)

        # update last actions in state
        self.state = self.update_last_actions(self.state, action)


    @partial(jax.jit, static_argnums=0)
    def update_last_actions(self, state, action):
        last_actions = jnp.roll(state.last_actions, shift=-1, axis=0)
        last_actions = last_actions.at[-1].set(action)
        state.replace(last_actions=last_actions)
        return state


    @partial(jax.jit, static_argnums=0)
    def clip_action(self, action):
        return jnp.clip(action, self.env.action_space.low, self.env.action_space.high)


    @partial(jax.jit, static_argnums=0)
    def get_obs_fn(self, state):
        obs = self.env._get_obs(state)
        obs = normalize(obs, self.env.observation_space.low, self.env.observation_space.high)
        return obs


    def send_command_msg(self, thrust, bodyrates):
        thrust = thrust / self.env.quadrotor._mass
        msg = norm_thrust_rates_to_msg(thrust, bodyrates)

        if self.ready and self.activated:
            self.control_pub.publish(msg)


    def state_callback(self, msg: QuadState):
        self.state = update_state_with_hist_from_msg(self.state, msg)


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

    rospy.init_node("online_state_hover_rollout_node", anonymous=True)

    # get launch params
    sim_quad = rospy.get_param("~sim_quad", "kolibri")
    real_quad = rospy.get_param("~real_quad", "kolibri")
    target_pos = rospy.get_param("~target_pos", "0.0, 0.0, 1.0")
    target_pos = [float(x) for x in target_pos.split(",")]
    num_residual_models = rospy.get_param("~num_residual_models", 1)

    rospy.loginfo(f"sim_quad: {sim_quad}")
    rospy.loginfo(f"real_quad: {real_quad}")
    rospy.loginfo(f"target_pos: {target_pos}")
    rospy.loginfo(f"num_residual_models: {num_residual_models}")

    # fake env
    dt = 0.02                     # seconds
    quad_obj = Quadrotor.from_name(sim_quad)
    env = HoveringStateEnv(
        dt=dt,
        delay=0.04,
        quad_obj=quad_obj,
        hover_target=target_pos,
    )
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    # create policy function and params template
    policy_params_path, policy_hidden_dims = get_state_hover_policy_info(sim_quad)
    policy_fn, params_template = load_policy_fn_and_params(obs_dim, action_dim, hidden_dims=policy_hidden_dims, 
                                                           mass=env.quadrotor._mass, cpu_device=CPU_DEVICE)

    # create node class
    node = OnlineStateHoverRolloutNode(env, policy_fn, params_template, real_quad=real_quad)
    rospy.spin()


if __name__ == "__main__":
    main()
