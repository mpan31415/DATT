import numpy as np

from DATT.configuration.configuration import *

drone_config = DroneConfiguration()

wind_config = WindConfiguration()

init_config = InitializationConfiguration(
    pos = ConfigValue[np.ndarray](
        default=np.array([0.0, 0.0, 0.0]), 
        randomize=True,
        min=np.array([-0.5, -0.5, -0.5]),
        max=np.array([ 0.5,  0.5,  0.5])
    ),
    vel = ConfigValue[np.ndarray](
        default=np.array([0.0, 0.0, 0.0]), 
        randomize=False
    ),
    # Represented as Euler ZYX in degrees
    rot = ConfigValue[np.ndarray](
        default=np.array([0.0, 0.0, 0.0]),
        randomize=False
    ),
    ang = ConfigValue[np.ndarray](
        default=np.array([0.0, 0.0, 0.0]),
        randomize=False
    )
)

sim_config = SimConfiguration(
    k=ConfigValue[float](default=0.4, randomize=False),
)

adapt_config = AdaptationConfiguration()

train_config = TrainingConfiguration()

policy_config = PolicyConfiguration()

ref_config = RefConfiguration()

config = AllConfig(drone_config, wind_config, init_config, sim_config, adapt_config, train_config, policy_config, ref_config)
