import argparse
import os
import pickle

from env import RunnerEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def get_train_cfg(exp_name, max_iterations, resume_path=None):
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
            "resume": True,
            "resume_path": resume_path,
            "run_name": "",
            "save_interval": 500,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }
    return train_cfg_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="runner")
    parser.add_argument("--resume_ckpt", type=int, default=100, help="Checkpoint to resume from")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Number of additional training iterations")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    current_dir = os.path.dirname(__file__)
    log_dir = os.path.join(current_dir, f"../logs/{args.exp_name}")
    resume_path = os.path.join(log_dir, f"model_{args.resume_ckpt}.pt")

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations, resume_path)

    updated_log_dir = os.path.join(current_dir, f"../logs/resume_{args.exp_name}")
    os.makedirs(updated_log_dir, exist_ok=True)
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{updated_log_dir}/cfgs.pkl", "wb"),
    )

    env = RunnerEnv(
        num_envs=train_cfg["runner"]["num_steps_per_env"],
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )

    runner = OnPolicyRunner(env, train_cfg, updated_log_dir, device="mps")
    runner.load(resume_path)

    # Train
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# Resume training with updated reward functions
python src/resume.py -e runner --resume_ckpt 4500 --max_iterations 10000
"""
