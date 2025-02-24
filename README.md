# Runner: Reinforcement Learning for Humanoid Running without Motion Capture

![Runner_video_1-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/fe6cf06d-c815-4304-9d53-171005711324)

## Abstract

This repository contains my individual research on using deep reinforcement learning (RL) to train a humanoid robot—a Unitree G1—to run without relying on any motion capture data. By leveraging a custom environment built on the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) simulator and training with Proximal Policy Optimization (PPO), I demonstrate that well-crafted reward functions and observation spaces can produce robust, human-like running gaits in simulation.

## Introduction

Humanoid locomotion is a rich research area: while many works focus on walking, my goal here is to tackle the more challenging problem of running in simulation. Traditional control strategies often rely on precisely modeled dynamics or motion capture data. In this project, I use pure reinforcement learning—without any motion capture—to let the algorithm learn high-level running behaviors directly from trial-and-error.

### Key Objectives

- Human-Like Running: Aim for higher speeds and flight phases characteristic of running.
- No Motion Capture: Rely solely on physics simulation and reward shaping to discover the gait.
- Stability & Efficiency: Balance is critical, but so is minimizing energy usage, which often leads to more natural running.

## Methods

### Environment Design

I built a custom environment, RunnerEnv, using the Genesis library. The environment:

1. **Loads the Unitree G1 Model**: Based on the MJCF model from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1).
2. **Applies PD Control**: The control inputs from the policy are converted into position targets via PD controllers for each of the robot’s 29 joints.
3. **Uses a Variety of Rewards**: Includes forward velocity, base height stability, energy efficiency, and others that encourage running-like motions.

#### Observations

Each environment step emits a high-dimensional vector encompassing:

- Base linear and angular velocities (in the robot’s frame).
- Gravity projection and orientation information.
- Commands for desired velocity (to encourage accelerating to running speeds).
- Joint positions, velocities, and the previous actions (to penalize jerky motions).

#### Actions

I use action offsets around a nominal pose. Each offset is mapped to a position target for the humanoid’s joints, with optional one-step latency to approximate real robotic systems.

### Reward Shaping

The reward function is crucial for achieving running behavior. Key components:

- **Forward Velocity**: Strongly incentivizes achieving higher speeds.
- **Base Height & Stability**: Minimizes excessive pitch/roll and ensures the robot remains upright.
- **Energy Penalties**: Penalizes large, wasteful joint torques to foster efficient motion.
- **Optional Terms**: Encouraging symmetrical leg swings, limiting hip roll/yaw deviations, etc.

I iterated on these reward components to balance raw speed with realistic human-like posture.

### Policy Optimization

I train with **Proximal Policy Optimization (PPO)** from the [rsl_rl](https://github.com/leggedrobotics/rsl_rl) library. The policy and value networks each have hidden layers of sizes [512, 256, 128], using ELU activation functions. Core hyperparameters:

- **Clip Parameter**: 0.2
- **Entropy Coefficient**: 0.01
- **Mini-Batches**: 4
- **Learning Rate**: 0.001
- **Discount Factor (γ)**: 0.99

These settings let the policy improve steadily while avoiding destructively large updates.

## Results

After training the policy for tens of thousands of iterations in a parallel simulation setting (up to 2048 environments at once), I observed:

- **Human-Like Running Gaits**: The agent spontaneously learned a gait with flight phases, significantly faster than a typical walking speed.
- **Stable Locomotion**: The agent reliably remained upright for the full episode duration, even under modest perturbations.
- **Efficient Motions**: The combination of forward speed rewards and energy penalties led to smooth, efficient leg cycling.

By combining multiple reward terms, the resulting motion is reminiscent of a forward-leaning run, rather than a stepping walk. Video demos (or recorded logs) indicate the policy’s strong running behavior.

## Usage

1. Install dependencies

```bash
conda create -n runner python=3.10
conda activate runner
conda install -c conda-forge numpy
conda install pytorch -c pytorch
pip install genesis-world
pip install -r requirements.txt
```

2. Run Training

```bash
python src/train.py -e runner_project --max_iterations 20000
```

This command will save logs and models in `logs/runner_project`.

3. Continue/Resume Training

If you want to continue from a checkpoint:

```bash
python resume.py -e runner_project --resume_ckpt 2000 --max_iterations 10000
```

4. Evaluation

```bash
python eval.py -e runner_project --ckpt 5000 -v
```

The `-v` option starts an interactive viewer to see the policy in action.

## Conclusion

Through individual effort, I have developed a reinforcement learning system capable of training a Unitree G1 humanoid to run in simulation, entirely without motion capture data. By leveraging carefully shaped rewards, robust PPO training, and the flexibility of the Genesis simulator, the resulting policy exhibits stable, human-like running. This project highlights that carefully engineered RL pipelines can discover sophisticated locomotion skills from scratch.

## References

- Schulman, J. et al. “Proximal Policy Optimization Algorithms.” arXiv:1707.06347, 2017.
- Peng, X.B. et al. “DeepLoco: Dynamic Locomotion Skills Using Hierarchical Deep Reinforcement Learning.” ACM TOG 36(4), 2017.
- [Unitree G1 Model in MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1)
- [Genesis Documentation](https://github.com/Genesis-Embodied-AI/Genesis)
- [rsl_rl Documentation](https://github.com/leggedrobotics/rsl_rl)
