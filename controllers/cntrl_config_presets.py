from DATT.learning.tasks import DroneTask
from DATT.controllers.cntrl_config import PIDConfig, MPPIConfig, DATTConfig

# DATT hover
datt_hover_config = DATTConfig()

# Simple DATT w/ feedforward, without any adaptation
datt_config = DATTConfig()
datt_config.policy_name = 'datt'
datt_config.task = DroneTask.TRAJFBFF


# DATT w/ feedforward and L1 adaptation
datt_adaptive_L1_config = DATTConfig()
datt_adaptive_L1_config.policy_name = 'datt_wind_adaptive'
datt_adaptive_L1_config.task = DroneTask.TRAJFBFF
datt_adaptive_L1_config.adaptive = True
datt_adaptive_L1_config.adaptation_type = 'l1'
datt_adaptive_L1_config.adaptive_policy_name = None

# DATT w/ feedforward and RMA adaptation
datt_adaptive_RMA_config = DATTConfig()
datt_adaptive_RMA_config.policy_name = 'datt_wind_adaptive'
datt_adaptive_RMA_config.task = DroneTask.TRAJFBFF
datt_adaptive_RMA_config.adaptive = True
datt_adaptive_RMA_config.adaptation_type = 'rma'
datt_adaptive_RMA_config.adaptive_policy_name = 'wind_RMA'


################### FOR KOLIBRI TASKS ###################

# Kolibri hover
kolibri_hover_config = DATTConfig()
kolibri_hover_config.policy_name = 'kolibri_hover'
kolibri_hover_config.task = DroneTask.HOVER

# Kolibri adaptive hover
kolibri_hover_adaptive_config = DATTConfig()
kolibri_hover_adaptive_config.policy_name = 'kolibri_hover_adaptive10'
kolibri_hover_adaptive_config.task = DroneTask.HOVER
kolibri_hover_adaptive_config.adaptive = True
kolibri_hover_adaptive_config.adaptation_type = 'l1'
kolibri_hover_adaptive_config.adaptive_policy_name = None

# Kolibri tracking
kolibri_tracking_config = DATTConfig()
kolibri_tracking_config.policy_name = 'kolibri_tracking_circle'
# kolibri_tracking_config.policy_name = 'kolibri_tracking_fig8'
kolibri_tracking_config.task = DroneTask.TRAJFBFF

# Kolibri adaptive tracking
kolibri_tracking_adaptive_config = DATTConfig()
# kolibri_tracking_adaptive_config.policy_name = 'kolibri_tracking_circle_adaptive10_highfid_50envs'
kolibri_tracking_adaptive_config.policy_name = 'kolibri_tracking_fig8_adaptive10_highfid_50envs'
# kolibri_tracking_adaptive_config.policy_name = 'kolibri_tracking_fig8_adaptive10_highfid_50envs_new'
kolibri_tracking_adaptive_config.task = DroneTask.TRAJFBFF
kolibri_tracking_adaptive_config.adaptive = True
kolibri_tracking_adaptive_config.adaptation_type = 'l1'
kolibri_tracking_adaptive_config.adaptive_policy_name = None



################### OTHER CONTROLLERS ###################

mppi_config = MPPIConfig()
mppi_config.H = 40
mppi_config.N = 1024
mppi_config.run_L1 = False
### misc ###
mppi_config.show_wind_terms = False
mppi_config.show_policy_time = False


pid_config = PIDConfig()
# for tracking (higher PID gains)
pid_config.kp_pos = 24.0
pid_config.kd_pos = 8.0
pid_config.ki_pos = 2.4
### misc ###
pid_config.show_policy_time = False
