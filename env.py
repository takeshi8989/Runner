import os
import torch
import math
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def run_sim(env, policy):
    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)


class RunnerEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="mps"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequence on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 960),
                max_FPS=int(0.5 / self.dt),
                camera_pos=(-3.0, 0.0, 2.0),
                camera_lookat=(2.0, -1.5, 1.0),
                camera_fov=60,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=False,
                world_frame_size=1.0,
                n_rendered_envs=1
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.plane = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        self.plane.geoms[0].set_friction(4.0)

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        current_dir = os.path.dirname(__file__)
        robot_path = os.path.join(current_dir, "model/g1.xml")
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file=robot_path)
        )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # force range
        self.robot.set_dofs_force_range(
            lower=np.array([-self.env_cfg["force_range"]] * self.num_actions),
            upper=np.array([self.env_cfg["force_range"]] * self.num_actions),
            dofs_idx_local=self.motor_dofs,
        )

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -10], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging

    def setup_sim(self, policy):
        gs.tools.run_in_another_thread(fn=run_sim, args=(self, policy))
        self.scene.viewer.start()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.base_pos[:, 2] < self.env_cfg.get("termination_base_height", 0.4)

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_forward_velocity(self):
        # Reward forward velocity (x-axis)
        forward_velocity = self.base_lin_vel[:, 0]
        return forward_velocity

    def _reward_energy_efficiency(self):
        # Penalize excessive energy usage
        energy_efficiency = torch.sum(torch.square(self.actions), dim=1)
        return -energy_efficiency

    def _reward_joint_limits(self):
        # Penalize extreme joint positions
        joint_limits = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return -joint_limits

    def _reward_stability(self):
        # Reward for keeping roll and pitch angles small
        roll_pitch_error = torch.square(self.base_euler[:, 0]) + torch.square(self.base_euler[:, 1])
        return torch.exp(-roll_pitch_error)

    def _reward_base_height(self):
        # Reward for maintaining target base height
        return -torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_survival_time(self):
        # Constant positive reward for staying alive
        return torch.ones(self.num_envs, device=self.device)

    def _reward_knee_straightness(self):
        # Target angle for straight knees and tolerance for small bending
        target_knee_angle = 0.0
        tolerance = 0.2  # Allowable bending in radians
        knee_joints = [
            self.env_cfg["dof_names"].index("left_knee_joint"),
            self.env_cfg["dof_names"].index("right_knee_joint"),
        ]
        knee_angles = self.dof_pos[:, knee_joints]
        penalty = torch.sum(
            torch.square(torch.clamp(knee_angles - target_knee_angle, min=-tolerance, max=tolerance)),
            dim=1
        )
        return -penalty

    def _reward_foot_contact(self):
        # Use ankle roll links as foot contact points
        left_foot_link_name = "left_ankle_roll_link"
        right_foot_link_name = "right_ankle_roll_link"

        # Retrieve link entities
        left_foot_entity = self.robot.get_link(left_foot_link_name)
        right_foot_entity = self.robot.get_link(right_foot_link_name)

        # Get contact information for the left and right foot entities
        left_foot_contacts = self.robot.get_contacts(with_entity=left_foot_entity)
        right_foot_contacts = self.robot.get_contacts(with_entity=right_foot_entity)

        # Convert valid_mask from NumPy to PyTorch tensor
        left_foot_in_contact = torch.tensor(left_foot_contacts["valid_mask"], device=self.device).any(dim=1).float()
        right_foot_in_contact = torch.tensor(right_foot_contacts["valid_mask"], device=self.device).any(dim=1).float()

        # Reward if either foot is in contact
        return (left_foot_in_contact + right_foot_in_contact).clamp(max=1.0)

    def _reward_smooth_motion(self):
        # Penalize large action changes (to avoid jerky movements)
        return -torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_tracking_lin_vel(self):
        # Penalize deviations from the target velocity (reduce jerky forward movement)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return -lin_vel_error  # Penalize deviations

    def _reward_straight_walking(self):
        reward_x_pos = self.base_pos[:, 0]
        reward_y_pos = -torch.abs(self.base_pos[:, 1])
        return reward_x_pos + reward_y_pos

    def _reward_large_strides(self):
        # Define the hip pitch joints for left and right sides
        left_hip_joint = "left_hip_pitch_joint"
        right_hip_joint = "right_hip_pitch_joint"

        # Get the indices of the hip joints
        left_hip_index = self.env_cfg["dof_names"].index(left_hip_joint)
        right_hip_index = self.env_cfg["dof_names"].index(right_hip_joint)

        # Calculate the absolute difference between left and right hip joint actions
        hip_joint_difference = torch.abs(self.actions[:, left_hip_index] - self.actions[:, right_hip_index])

        # Return the reward proportional to the hip joint differences
        return hip_joint_difference

    def _reward_torso_upright(self):
        # Penalize pitch and roll angles of the torso
        torso_pitch_roll = torch.square(self.base_euler[:, :2])  # [pitch, roll]
        return -torch.sum(torso_pitch_roll, dim=1)

    def _reward_crotch_control(self):
        # Get the positions (angles) of the left and right hip roll joints
        left_hip_roll_angle = self.dof_pos[:, self.env_cfg["dof_names"].index("left_hip_roll_joint")]
        right_hip_roll_angle = self.dof_pos[:, self.env_cfg["dof_names"].index("right_hip_roll_joint")]

        # Square the deviations to penalize large angles more
        hip_roll_deviation = torch.square(left_hip_roll_angle) + torch.square(right_hip_roll_angle)

        # Return a negative reward proportional to the total deviation
        return -hip_roll_deviation

    def _reward_arm_swing(self):
        # Reward symmetrical arm swings
        left_arm = self.actions[:, self.env_cfg["dof_names"].index("left_shoulder_pitch_joint")]
        right_arm = self.actions[:, self.env_cfg["dof_names"].index("right_shoulder_pitch_joint")]
        return -torch.abs(left_arm + right_arm)  # Penalize asymmetrical arm swings
