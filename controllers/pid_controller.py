import numpy as np
from scipy.spatial.transform import Rotation as R
from DATT.quadsim.control import Controller
from DATT.quadsim.models import RBModel
from DATT.controllers.cntrl_config import PIDConfig
from DATT.quadsim.rigid_body import State_struct
from DATT.configuration.configuration import AllConfig

from time import time


class PIDController(Controller):
  def __init__(self, config : AllConfig, cntrl_config : PIDConfig):
    super().__init__()
    self.pid_config = cntrl_config
    self.config = config

    self.pos_err_int = np.zeros(3)
    self.v_prev = np.zeros(3)
    self.prev_t = None
    self.start_pos = np.zeros(3)

    self.show_policy_time = self.pid_config.show_policy_time


  def response(self, **response_inputs ):

    tic = time()
    
    t = response_inputs.get('t')
    state : State_struct = response_inputs.get('state')
    # ref_dict : dict = response_inputs.get('ref')

    ref_state = self.ref_func.get_state_struct(t)

    # print('offset : ', self.ref_func.offset_pos)

    if self.prev_t != None:
      dt = t - self.prev_t
    else:
      dt = self.config.sim_config.dt()

    # PID
    pos = state.pos - self.start_pos
    vel = state.vel
    rot = state.rot
    p_err = pos - ref_state.pos
    v_err = vel - ref_state.vel

   # Updating error for integral term.
    self.pos_err_int += p_err * dt

    acc_des = (np.array([0, 0, self.config.sim_config.g()]) 
              - self.pid_config.kp_pos * (p_err) 
              - self.pid_config.kd_pos * (v_err) 
              - self.pid_config.ki_pos * self.pos_err_int )
    

    u_des = rot.as_matrix().T.dot(acc_des)

    acc_des = np.linalg.norm(u_des)

    rot_err = np.cross(u_des / acc_des, np.array([0, 0, 1]))

    eulers = rot.as_euler("ZYX")
    yaw = eulers[0]
    omega_des = - self.pid_config.kp_rot * rot_err
    omega_des[2] += - self.pid_config.yaw_gain * (yaw - 0.0)
      
    self.v_prev = state.vel
    self.prev_t = t

    if self.show_policy_time:
      toc = time()
      print(f"PIDController response time: {toc - tic:.4f} seconds")

    # action = np.r_[acc_des, omega_des]
    return acc_des, omega_des