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
kolibri_hover_config.policy_name = 'kolibri_hover_500000_steps'
kolibri_hover_config.task = DroneTask.HOVER

# Kolibri adaptive hover
kolibri_hover_adaptive_config = DATTConfig()
kolibri_hover_adaptive_config.policy_name = 'kolibri_hover_500000_steps'
kolibri_hover_adaptive_config.task = DroneTask.HOVER
kolibri_hover_adaptive_config.adaptive = True
kolibri_hover_adaptive_config.adaptation_type = 'l1'
kolibri_hover_adaptive_config.adaptive_policy_name = None

# Kolibri tracking
kolibri_tracking_config = DATTConfig()
kolibri_tracking_config.policy_name = 'kolibri_tracking_circle_20000000_steps'
kolibri_tracking_config.task = DroneTask.TRAJFBFF

# Kolibri adaptive tracking
kolibri_tracking_adaptive_config = DATTConfig()
kolibri_tracking_adaptive_config.policy_name = 'kolibri_tracking_circle_20000000_steps'
kolibri_tracking_adaptive_config.task = DroneTask.TRAJFBFF
kolibri_tracking_adaptive_config.adaptive = True
kolibri_tracking_adaptive_config.adaptation_type = 'l1'
kolibri_tracking_adaptive_config.adaptive_policy_name = None



################### OTHER CONTROLLERS ###################

pid_config = PIDConfig()


mppi_config = MPPIConfig()