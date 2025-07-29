import numpy as np
# from DATT.learning.train_policy import DroneTask
from DATT.configuration.configuration import *
# from DATT.learning.train_policy import DroneTask, RLAlgo, SAVED_POLICY_DIR, import_config, CONFIG_DIR
from DATT.learning.configs import *
from DATT.learning.tasks import DroneTask

class MPPIConfig:
    lam = 0.05 # temparature
    H = 40 # horizon
    N = 4096 # number of samples
    K_delay = 1
    sim_K_delay  = 1

    run_L1 = True   # run L1 adaptation
    show_wind_terms = False
    show_policy_time = False

    sample_std = [0.25, 0.7, 0.7, 0.7] # standard deviation for sampling = [thrust (unit = hovering thrust), omega (unit = rad/s)]
    gamma_mean = 1.0 # learning rate
    gamma_Sigma = 0. # learning rate
    omega_gain = 40. # gain of the low-level controller
    discount = 0.99 # discount factor in MPPI
    a_min = [0., -5., -5., -2.] # bounds of sampling action = [thrust, omega (unit = rad/s)]
    a_max = [0., 5., 5., 2.]

    # reward functions
    alpha_p = 5.0
    alpha_w = 0.0
    alpha_a = 0.0
    alpha_R = 3.0
    alpha_v = 0.0
    alpha_z  = 0.0
    alpha_yaw  = 0.0

    noise_measurement_std = np.zeros(10)
    noise_measurement_std[:3] = 0.005
    noise_measurement_std[3:6] = 0.005
    noise_measurement_std[6:10] = 0.01

class PIDConfig:
    kp_pos = 6.0
    kd_pos = 4.0
    ki_pos = 1.2 # 0 for sim
    kp_rot =   150.0/16
    yaw_gain = 220.0/16
    kp_ang =   16

    show_policy_time = False

class DATTConfig:
    task = DroneTask.HOVER
    policy_name = "datt_hover"
    # config_filename = "default_hover.py"

    adaptive = False 
    adaptation_type = None # l1/naive/rma
    adaptive_policy_name = None # policy name if rma

    # config : AllConfig = import_config(config_filename)

    # def load_config(self, ):
    #     self.config = import_config(self.config_filename)


