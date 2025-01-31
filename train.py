import argparse
import os
import pickle
import shutil

from env import RunnerEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from curriculum import get_reward_scales


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 64,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "save_interval": 500,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "model_path": "path_to_humanoid_model.xml",
        "num_actions": 29,
        "default_joint_angles": {
            "left_hip_pitch_joint": 0.1,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.1,
            "left_ankle_pitch_joint": -0.1,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.1,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.1,
            "right_ankle_pitch_joint": -0.1,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0
        },
        "dof_names": [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
        ],
        # PD
        "kp": 100.0,
        "kd": 5.0,
        "force_range": 80.0,
        # termination
        "termination_base_height": 0.4,
        "termination_if_roll_greater_than": 60,  # degree
        "termination_if_pitch_greater_than": 60,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.77],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 1.0,
        "simulate_action_latency": True,
        "clip_actions": 10.0,
    }
    obs_cfg = {
        "num_obs": 96,
        "obs_scales": {
            "lin_vel": 1.0,
            "ang_vel": 0.5,
            "dof_pos": 1.0,
            "dof_vel": 0.1,
        },
    }
    reward_cfg = {
        "reward_scales": {
            "survival_time": 8.0,
            "base_height": 5.0,  # Encourage maintaining height
            "stability": 3.0,  # Strongly discourage angular velocity
            "energy_efficiency": 2.0,  # Penalize excessive energy usage
            "forward_velocity": 12.0,   # Encourage forward movement
            "tracking_lin_vel": 8.0,  # Encourage tracking linear velocity
            "foot_contact": 0.00001,  # Penalize foot contact
            "smooth_motion": 0.1,  # Negative weight to discourage rapid changes
            "straight_walking": 10.0,  # Encourage walking in a straight line
            "large_strides": 10.0,
            "torso_upright": 0.1,
            "crotch_control": 8.0,
        },
        "base_height_target": 0.77,
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-1.0, 1.0],
        "lin_vel_y_range": [-1.0, 1.0],
        "ang_vel_range": [-3.14, 3.14],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="runner")
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=1000)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    current_dir = os.path.dirname(__file__)
    log_dir = os.path.join(current_dir, f"logs/{args.exp_name}")

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    reward_cfg["reward_scales"] = get_reward_scales()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = RunnerEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="mps")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python train.py --max_iterations 10000
"""
