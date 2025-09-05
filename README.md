# DATT: Deep Adaptive Trajectory Tracking for Quadrotor Control

This repo is forked from the original [DATT repo](https://github.com/KevinHuang8/DATT), and contains both the original training and sim eval pipeline scripts, and most importantly, the added `datt_ros` package for rolling out the controllers in Agilicious simulation. 

## 1. Setup

#### Create conda environment:
```
mamba create -n datt python=3.10
mamba activate datt
```

#### Extra steps to solve compatibility issues:
```
# 1. edit requirements.txt and comment out the following 2 lines:
gym==0.21.0
torch==1.13.0+cu117

# 2. manually install openai gym (0.21.0)
pip install git+https://github.com/openai/gym.git@9180d12e1b66e7e2a1a622614f787a6ec147ac40

# 3. manually install torch (1.13.0) with CUDA (11.7) support
pip install torch==1.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```

#### Then, install the remaining dependencies:
```
pip install -r requirements.txt
```

#### (IMPORTANT) Configure the PYTHONPATH
The repo requires the parent folder exist on `PYTHONPATH`.

The recommended setup is to create a folder named "python" (e.g. in your home folder) and then clone `DATT` in `~/python`.

```
mkdir ~/python && cd ~/python
git clone https://github.com/mpan31415/DATT.git
```

Next, in your `.bashrc`, add `${HOME}/python` to `PYTHONPATH`.
e.g. add the following line.
```
export PYTHONPATH="${HOME}/python":"${PYTHONPATH}"
```


## 2. Defining Task & Configuration

Training a policy requires specifying a *task* and a *configuration*. The task describes the environment and reward, while the configuration defines various environmental parameters, such as drone mass, wind, etc., and whether/how they are randomly sampled.

### Tasks

See tasks in `.learning/tasks/`. Each class is superclass of `BaseEnv`, which has the gym env API. In practice, the primary thing that should change between different drone tasks are the action space and reward function.

Standard trajectory tracking should have `trajectory_fbff` passed in.

NOTE: When adding a new task, you must modify the `DroneTask` enum in `train_policy.py` to add the new task along with its corresponding environment, for it to get recognized. 

### Configuration

`./configuration/configuration.py` defines all the modifiable parameters, as well as their default values. To define a configuration, create a new `.py` file that instantiates a `AllConfig` object named `config`, which modifies the config values for parameters that are different from the default values. See config profiles in `./configuration/` for examples.

**NOTE: `configuration.py` should not be modified (it just defines the configurable parameters). To create a configuration, a new file needs to be created.**

Configurable parameters that can be randomly sampled during training can be set to a `ConfigValue` (see `configuration.py`). Each `ConfigValue` takes in the default value of the parameter, and whether or not that parameter should be randomized during training. If a param should be randomized, you need to also specify the min and max possible range of randomization for that parameter.

Each parameter is part of some parameter group, which shares a `Sampler`, which specifies how parameters in that group should be randomly sampled if they are specified to be randomized. By default, the sampling function is just uniform sampling, but the sampling function can take in anything, like the reward or timestep, which can be used to specify a learning curriculum, etc. To add more info to the sampling function input, or to change *when* in training a parameter is resampled from the default, however, you must modify the task/environment.


## 3. Training a Policy

Run `train_policy.py` from the command line. It takes the following arguments:

- `-n` `--name` : the name of the policy. All log/data files will have this name as the prefix. If you pass in a name that already exists (a policy exists if a file with the same name appears in `./saved_policies/`), then you will continue training that policy with new data. If not provided, autogenerates depending on the other parameters.
- `-t` `--task` : the name of the task; must be defined in the `DroneTask` enum in `train_policy.py`. Essentially, this specifies the environment. Defaults to hovering
- `-c` `--config` : The configuration file (a `.py` file), which must instantiate an `AllConfig` object named `config`. 
- `-ts` `--timesteps` : The number of timesteps to train for, defaults to 1 million. The model also saves checkpoints every 500,000 steps.
- `-d` `--log-dir` : The directory to save training logs (tensorboard) to. Defaults to `./logs/{policy_name}_logs`

**NOTE: must run `train_policy.py` from the `./learning/` directory for save directories to line up correctly.**

Go to the `learning` folder and run:

1. To train a trajectory tracking policy, run:
```
# no adaptation
python train_policy.py -n kolibri_tracking -c kolibri_tracking.py -t trajectory_fbff --ref my_circle_ref -ts 25000000 --checkpoint True

# L1 adaptation
python train_policy.py -n kolibri_tracking_adaptive -c kolibri_tracking_adaptive.py -t trajectory_fbff --ref my_circle_ref -ts 25000000 --checkpoint True
```

2. To train a hovering policy, run:
```
# no adaptation
python train_policy.py -n kolibri_hover -c kolibri_hover.py -t hover -ts 500000 --checkpoint True

# L1 adaptation
python train_policy.py -n kolibri_hover_adaptive -c kolibri_hover_adaptive.py -t hover -ts 500000 --checkpoint True
```

To visualize tensorboard logs, run:
```
cd /path/to/tensorboard/logs
tensorboard --logdir=. --port=6006
```


## 4. Evaluating a policy

Run `eval_policy.py` with the policy name, task, algorithm the policy was trained on, and the number of evaluation steps.

This script currently just prints out the mean/std rewards over randomized episodes, and visualizes rollouts of the policy in sim.

For example, run:

```
# tracking policy
python eval_policy.py -n policy -c DATT_config.py -t trajectory_fbff --ref random_zigzag -s 500 --viz True

# hovering policy
python eval_policy.py -n kolibri_hover_500000_steps -c kolibri_datt_hover.py -t hover --ref hover -s 500 --viz True
```

## 5. Running the Simulator

As stated in the paper we introduce our architecture DATT and compare it with PID and MPPI as baselines. 

A sample sim run can look like : 

```bash
python3 main.py --cntrl <controller name> --cntrl_config <controller config preset> --env_config <env config file> --ref <ref>
```

- `cntrl` : name of controller i.e `pid` / `mppi` / `datt` [Default : `datt`]
- `cntrl_config` : Default controller configurations are in `./controllers/cntrl_config.py`. In this field, you make a preset of the default configurations and add them to `./controllers/cntrl_config_presets.py`. You add the preset name from the preset file in this field.
- `env_config` is the same as config during training
- `ref` : reference trajectory to be tracked by the controller
- `seed` : seed of a particular trajectory family you want to use.

### PID
```bash
main.py --cntrl pid --cntrl_config pid_config --env_config datt.py --ref random_zigzag
```

### MPPI
```bash
python3 main.py --cntrl mppi --cntrl_config mppi_config --env_config datt.py --ref random_zigzag
```

### DATT

We are providing pre-trained models for DATT for different tasks : 

| Task                                 | Configuration file | Model |
| -------------                        | -------------      |-----------------|
| Hover                                | [datt_hover.py](configuration/datt_hover.py)     |[datt_hover](learning/saved_policies/datt_hover.zip)|
| Trajectory tracking (No adaptation)  | [datt.py](configuration/datt.py) |       [datt](learning/saved_policies/datt.zip)|
| Trajectory tracking with adaptation  | [datt_wind_adaptive.py](configuration/datt_wind_adaptive.py)       |[datt_wind_adaptive](learning/saved_policies/datt_wind_adaptive.zip)|



```bash
# python3 main.py --cntrl <controller> --cntrl_config <cntrl_config_preset> --env_config <env_config>.py --ref <task>

# kolibri hover (DATT without adaptation)
python3 main.py --cntrl datt --cntrl_config kolibri_hover_config --env_config kolibri_hover.py --ref hover
# kolibri hover (DATT with L1 adaptation)
python3 main.py --cntrl datt --cntrl_config kolibri_hover_adaptive_config --env_config kolibri_hover_adaptive.py --ref hover
# kolibri hover (L1-MPC)
python3 main.py --cntrl mppi --cntrl_config mppi_config --env_config kolibri_mppi.py --ref hover
# kolibri hover (DF-PID)
python3 main.py --cntrl pid --cntrl_config pid_config --env_config kolibri_pid.py --ref hover

################################################################################################

# kolibri tracking (DATT without adaptation)
python3 main.py --cntrl datt --cntrl_config kolibri_tracking_config --env_config kolibri_tracking.py --ref my_circle_ref
# kolibri tracking (DATT with L1 adaptation)
python3 main.py --cntrl datt --cntrl_config kolibri_tracking_adaptive_config --env_config kolibri_tracking_adaptive.py --ref my_circle_ref
# kolibri tracking (L1-MPC)
python3 main.py --cntrl mppi --cntrl_config mppi_config --env_config kolibri_mppi.py --ref my_circle_ref
# kolibri tracking (DF-PID)
python3 main.py --cntrl pid --cntrl_config pid_config --env_config kolibri_pid.py --ref my_circle_ref

################################################################################################

# hover
python3 main.py --cntrl datt --cntrl_config datt_hover_config --env_config datt_hover.py --ref hover
# trajectory tracking without adaptation
python3 main.py --cntrl datt --cntrl_config datt_config --env_config datt.py --ref random_zigzag --seed 2023
# trajectory tracking with adaptation with L1 adaptation
python3 main.py --cntrl datt --cntrl_config datt_adaptive_L1_config --env_config datt_wind_adaptive.py --ref random_zigzag --seed 2023
# trajectory tracking with adaptation with RMA adaptation
python3 main.py --cntrl datt --cntrl_config datt_adaptive_RMA_config --env_config datt_wind_adaptive.py --ref random_zigzag --seed 2023
```