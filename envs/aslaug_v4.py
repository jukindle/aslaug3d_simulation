from gym import spaces
import numpy as np
import pybullet as pb
import random
from . import aslaug_base


# Aslaug environment with automatic domain randomization
class AslaugEnv(aslaug_base.AslaugBaseEnv):

    # Overvation: [sp_pos_ee[3], mb_vel_r[3], link_pos[3*n_joints],
    #              joint_pos[3], joint_vel[3]]
    # Action: [mb_d_vel_r[3], joint_d_vel[3]]
    # State: [mb_pos_w[3], mb_vel_w[3], joint_pos[3], joint_vel[3]]
    def __init__(self, folder_name="", gui=False, free_cam=False,
                 recording=False, params=None, randomized_env=True):
        # Common params
        version = "v4"
        self.folder_name = folder_name
        self.soft_reset = False
        self.recording = recording
        self.success_counter = 0
        self.episode_counter = 0
        self.randomized_env = randomized_env

        # Initialize super class
        super().__init__(version, params, gui=gui, init_seed=None,
                         free_cam=free_cam)

        # Initialize score counter for ADR
        self.env_score = EnvScore(self.p["adr"]["batch_size"])

        for el in self.p["adr"]["adaptions"]:
            param = el["param"]
            self.set_param(param, el["start"])

    def setup_action_observation_spaces(self):
        # Define action space
        # [delta_mb_vel_r[3], delta_joint_vel[n_joints]]
        accel_lims_mb = self.p["base"]["acc_mag"]
        acc_lim_joints = (self.n_joints * [self.p["joints"]["acc_mag"]])
        highs_a = (self.p["world"]["tau"]
                   * np.concatenate((accel_lims_mb, acc_lim_joints)))
        lows_a = -highs_a
        n_d = self.p["world"]["action_discretization"]
        self.action_space = spaces.MultiDiscrete(lows_a.shape[0] * [n_d])
        self.actions = np.linspace(lows_a, highs_a, n_d)
        # Define observation space
        # Overvation: [sp_pos_ee[6], mb_vel_r[3], link_pos[6*n_links+1],
        #              joint_pos[n_joints], joint_vel[n_joints],
        #              scan[n_scans]]
        high_sp = np.array([self.p["world"]["size"]] * 2 + [1.5] + 3 * [np.pi])
        low_sp = -high_sp
        high_mb = np.array(self.p["base"]["vel_mag"])
        low_mb = -high_mb

        high_lp = []
        low_lp = []
        for v in self.p["joints"]["link_mag"]:
            high_lp += [v, v, v, np.pi, np.pi, np.pi]
            low_lp += [-v, -v, -0.3, -np.pi, -np.pi, -np.pi]
        high_lp = np.array(high_lp)
        low_lp = np.array(low_lp)
        high_j_p = self.joint_limits[self.actuator_selection, 1]
        low_j_p = self.joint_limits[self.actuator_selection, 0]
        high_j_v = np.array([self.p["joints"]["vel_mag"]] * self.n_joints)
        low_j_v = -high_j_v
        rng = self.p["sensors"]["lidar"]["range"]
        n_lid = sum([self.p["sensors"]["lidar"]["link_id1"] is not None,
                     self.p["sensors"]["lidar"]["link_id2"] is not None])
        high_scan_s = rng * np.ones(self.p["sensors"]["lidar"]["n_scans"])
        high_scan = np.repeat(high_scan_s, n_lid)
        low_scan = 0.1 * high_scan
        high_o = np.concatenate((high_sp, high_mb, high_lp, high_j_p,
                                 high_j_v, high_scan))
        low_o = np.concatenate((low_sp, low_mb, low_lp, low_j_p,
                                low_j_v, low_scan))

        self.obs_slicing = [0]
        for e in (high_sp, high_mb, high_lp, high_j_p, high_j_v) \
                + n_lid * (high_scan_s,):
            self.obs_slicing.append(self.obs_slicing[-1] + e.shape[0])
        self.observation_space = spaces.Box(low_o, high_o)

    def calculate_reward(self):
        # Introducte reward variable
        reward = 0.0
        done = False
        info = {}

        # Penalize if joint limit reached and end episode
        if self.check_joint_limits_reached():
            reward += self.p["reward"]["rew_joint_limits"]
            info["done_reason"] = "joint_limits_reached"
            done = True

        # Penalize collisions and end episode
        if self.check_collision():
            reward += self.p["reward"]["rew_collision"]
            info["done_reason"] = "collision"
            done = True

        # Reward for base to setpoint orientation
        r_ang_sp = self.calculate_base_sp_angle()
        reward += ((np.abs(self.last_r_ang_sp) - np.abs(r_ang_sp))
                   * self.p["reward"]["fac_base_sp_ang"]
                   / np.pi)
        self.last_r_ang_sp = r_ang_sp
        # Penalize velocity in move base rotation
        # mb_ang_vel = self.get_base_vels()[2]
        # reward += np.abs(mb_ang_vel)*self.tau*self.p["reward"]["fac_ang_vel"]

        # Check for timeout
        if self.step_no >= self.timeout_steps:
            reward += self.p["reward"]["rew_timeout"]
            info["done_reason"] = "timeout"
            done = True

        # Calculate goal distance
        eucl_dis, eucl_ang = self.calculate_goal_distance()

        # Calculate intermediate reward
        delta_eucl_dis = self.last_eucl_dis - eucl_dis
        delta_eucl_ang = self.last_eucl_ang - eucl_ang
        reward += (self.scl_eucl_dis
                   * self.p["reward"]["fac_goal_dis_lin"] * delta_eucl_dis)
        reward += (self.scl_eucl_ang
                   * self.p["reward"]["fac_goal_dis_ang"] * delta_eucl_ang)
        self.last_eucl_dis, self.last_eucl_ang = eucl_dis, eucl_ang

        # Check if goal reached
        if eucl_dis <= self.p["setpoint"]["tol_lin_mag"] and \
                eucl_ang <= self.p["setpoint"]["tol_ang_mag"]:

            if self.sp_hold_time >= self.p["setpoint"]["hold_time"]:
                if self.p["setpoint"]["continious_mode"]:
                    self.soft_reset = True
                    self.sp_hold_time = 0.0
                    self.step_no = 0
                    self.integrated_hold_reward = 0.0
                if not self.recording:
                    done = True
                    info["done_reason"] = "success"
                else:
                    self.success_counter += 1
                    self.reset()

                reward += self.p["reward"]["rew_goal_reached"]

            self.sp_hold_time += self.tau
            dis_f = 1.0 - eucl_dis / self.p["setpoint"]["tol_lin_mag"]
            rew_hold = (self.tau * self.p["reward"]["fac_sp_hold"]
                        + self.tau
                        * self.p["reward"]["fac_sp_hold_near"] * dis_f)
            rew_hold = rew_hold / self.p["setpoint"]["hold_time"]
            self.integrated_hold_reward += rew_hold
            reward += rew_hold
        else:
            reward -= self.integrated_hold_reward
            self.integrated_hold_reward = 0.0
            self.sp_hold_time = 0.0

        return reward, done, info

    def calculate_observation(self):
        sp_pose_ee = self.get_ee_sp_transform()
        link_pose_r = self.get_link_states(self.link_mapping)
        j_pos, j_vel = self.get_joint_states(self.actuator_selection)
        mb_vel_w = self.get_base_vels()
        scan_ret = self.get_lidar_scan()
        scan = np.concatenate([x for x in scan_ret if x])
        obs = np.concatenate((sp_pose_ee, mb_vel_w, link_pose_r.flatten(),
                              j_pos, j_vel, scan))
        return obs

    def reset(self, init_state=None, init_setpoint_state=None,
              init_obstacle_grid=None, init_obstacle_locations=None):

        if self.done_info is not None and \
                self.done_info["done_reason"] == "success":
            queue_full = self.env_score.add(1)
        else:
            queue_full = self.env_score.add(0)
        self.done_info = None

        if queue_full:
            score = self.env_score.get_avg_score()

            if score >= self.p["adr"]["success_threshold"]:
                d_sign = 1
            elif score < self.p["adr"]["fail_threshold"]:
                d_sign = -1
            else:
                d_sign = 0
            did_modification = False
            if d_sign != 0:
                for el in self.p["adr"]["adaptions"]:
                    param = el["param"]
                    val_cur = self.get_param(param)
                    delta = (el["end"]-el["start"])/el["steps"]
                    new_val = val_cur + delta * d_sign
                    if el["start"] <= new_val <= el["end"] or \
                            el["end"] <= new_val <= el["start"]:
                        self.set_param(param, new_val)
                        did_modification = True
                        print("Adapting {} to {}".format(param, new_val))
            if did_modification:
                self.env_score.reset()


        # Reset internal parameters
        self.episode_counter += 1
        self.step_no = 0
        self.integrated_hold_reward = 0.0

        # Reset setpoint only if requested
        if self.soft_reset:
            self.reset_setpoint(max_dis=self.p["world"]["spawn_range_x"])
            return self.calculate_observation()
        else:
            self.sp_history = []

        # Reset internal state
        self.state = {"base_vel": np.array([0.0, 0.0, 0.0]),
                      "joint_vel": np.array(7 * [0.0])}

        # Reset environment
        for i in range(len(self.joint_mapping)):
            pb.resetJointState(self.robotId, self.joint_mapping[i],
                               self.p["joints"]["init_states"][i],
                               0.0, self.clientId)

        self.possible_sp_pos = self.randomize_environment()

        # Reset robot base
        pb.resetBaseVelocity(self.robotId, [0, 0, 0], [0, 0, 0], self.clientId)

        # Reset setpoint
        sp_pos = self.reset_setpoint()

        # Reset robot arm
        collides = True
        while collides:
            x_min = np.max((5.5, sp_pos[0] - self.p["world"]["spawn_range_x"]))
            x_max = np.min((20.5, sp_pos[0] + self.p["world"]["spawn_range_x"]))
            x_coord = self.np_random.uniform(x_min, x_max)

            robot_pos = (x_coord, 1.5, 0.08)
            robot_init_yaw = self.np_random.uniform(-np.pi, np.pi)
            robot_ori = pb.getQuaternionFromEuler([np.pi / 2, 0, robot_init_yaw])
            pb.resetBasePositionAndOrientation(self.robotId, robot_pos, robot_ori,
                                               self.clientId)

            j0 = self.np_random.uniform(self.joint_limits[0, 0],
                                        self.joint_limits[0, 1])
            j3 = self.np_random.uniform(self.joint_limits[3, 0],
                                        self.joint_limits[3, 1])
            pb.resetJointState(self.robotId, self.joint_mapping[0],
                               j0,
                               0.0, self.clientId)
            pb.resetJointState(self.robotId, self.joint_mapping[3],
                               j3,
                               0.0, self.clientId)
            pb.stepSimulation(self.clientId)
            collides = self.check_collision()

        # Calculate observation and return
        obs = self.calculate_observation()
        self.last_yaw = None
        return obs

    def reset_setpoint(self, max_dis=None):
        # Spawn random setpoint
        sp_pos = random.sample(self.possible_sp_pos, 1)[0]
        self.move_sp(sp_pos)

        # Initialize reward state variables
        eucl_dis, eucl_ang = self.calculate_goal_distance()
        if max_dis is not None:
            for i in range(200):
                if eucl_dis <= max_dis:
                    break
                else:
                    sp_pos = random.sample(self.possible_sp_pos, 1)[0]
                    self.move_sp(sp_pos)
                    eucl_dis, eucl_ang = self.calculate_goal_distance()

        self.last_eucl_dis, self.last_eucl_ang = eucl_dis, eucl_ang
        self.scl_eucl_dis = 1 / self.last_eucl_dis
        self.scl_eucl_ang = 1 / self.last_eucl_ang
        self.last_r_ang_sp = self.calculate_base_sp_angle()
        self.scl_r_ang_sp = 1 / self.last_r_ang_sp
        self.sp_hold_time = 0.0

        self.soft_reset = False
        self.sp_history.append(sp_pos.tolist())
        return sp_pos

    def spawn_robot(self):
        # Spawn robot
        robot_pos = [0, 0, 0.02]
        robot_ori = pb.getQuaternionFromEuler([0, 0, 0])
        model_path = 'urdf/robot/aslaug.urdf'
        return pb.loadURDF(model_path, robot_pos, robot_ori,
                           useFixedBase=True,
                           physicsClientId=self.clientId)

    def spawn_setpoint(self):
        # Spawn setpoint
        mug_pos = [5, 2, 0.0]
        mug_ori = pb.getQuaternionFromEuler([0, 0, 0])
        model_path = 'urdf/beer_rothaus/beer_rothaus.urdf'
        spId = pb.loadURDF(model_path, mug_pos, mug_ori,
                           useFixedBase=True,
                           physicsClientId=self.clientId)

        # Spawn setpoint marker
        mug_pos = [5, 3, 0.0]
        mug_ori = pb.getQuaternionFromEuler([0, 0, 0])
        self.markerId = pb.loadURDF("sphere2red.urdf", mug_pos, mug_ori,
                                    globalScaling=0.2, useFixedBase=True,
                                    physicsClientId=self.clientId)
        return spId

    def spawn_additional_objects(self):
        ids = []
        # Spawn ground plane
        ids.append(pb.loadURDF('urdf/lee/lee-j-corridor.urdf',
                               useFixedBase=True,
                               physicsClientId=self.clientId))
        self.random_objects = self.p["world"]["random_objects"]
        for key in self.random_objects:
            file = key["file"]
            pose2d = key["pose2d"]
            id = pb.loadURDF("urdf/lee/random_objects/{}.urdf".format(file),
                             useFixedBase=True,
                             physicsClientId=self.clientId)
            key["id"] = id
            ids.append(id)
            self.move_object(id, pose2d)
        return ids

    def calculate_goal_distance(self):
        sp_pose_ee = self.get_ee_sp_transform()
        eucl_dis = np.linalg.norm(sp_pose_ee[1:3])  # Ignore x coord
        eucl_ang = np.linalg.norm(sp_pose_ee[3:6])
        return eucl_dis, eucl_ang

    def calculate_base_sp_angle(self):
        base_pose_sp = self.get_base_sp_transform()
        return base_pose_sp[5]

    def spawn_bookcases(self, n, easy=False):
        '''
        Prepares the simulation by spawning n bookcases.

        Args:
            n (int): Number of bookcases.
        Returns:
            list: List of bookcase IDs.
        '''
        model_path = 'urdf/kallax/kallax_large.urdf'

        kallax1 = pb.loadURDF(model_path, [1.47 / 2 + 7.9, 0, 0],
                              useFixedBase=True, physicsClientId=self.clientId)
        kallax2 = pb.loadURDF(model_path, [1.47 / 2 + 18.034, 0, 0],
                              useFixedBase=True, physicsClientId=self.clientId)
        self.bookcaseIds = [kallax1, kallax2]

    def move_object(self, id, pose2d):
        pos = [pose2d[0], pose2d[1], 0.0]
        ori = [0.0, 0.0] + [pose2d[2]]
        ori_quat = pb.getQuaternionFromEuler(ori)
        pb.resetBasePositionAndOrientation(id, pos, ori_quat,
                                           self.clientId)

        Rt = self.rotation_matrix(pose2d[2]).T
        return np.array(pos), Rt

    def move_bookcase(self, bookcaseId, pose2d, sp_layers=[0, 1, 2, 3]):
        '''
        Function which moves a bookcase to a new position and returns a list of
        possible setpoint locations w.r.t. the new position.

        Args:
            bookcaseId (int): ID of bookcase.
            pose2d (numpy.array): 2D pose to which bookcase should be moved to.
            sp_layers (list): Selection specifying in what layers the setpoint
                might be spawned. 0 means lowest and 3 top layer.
        Returns:
            list: 3D positions of possible setpoint locations w.r.t. pose2d.
        '''
        pos, Rt = self.move_object(bookcaseId, pose2d)

        # Calculate possible setpoint positions
        sp_pos = []
        for l in sp_layers:
            z = 0.037 + (0.33 + 0.025) * l
            y = 0.195 + 0.15
            sp_pos.append(pos + Rt.dot(np.array([+0.1775, y, z])))
            sp_pos.append(pos + Rt.dot(np.array([-0.1775, y, z])))
            sp_pos.append(pos + Rt.dot(np.array([+0.5325, y, z])))
            sp_pos.append(pos + Rt.dot(np.array([-0.5325, y, z])))

        return sp_pos

    def randomize_environment(self):
        # Randomize bookcases
        possible_sp_pos = []
        pos = [1.47 / 2 + 7.9, 0, 0]
        possible_sp_pos += self.move_bookcase(self.bookcaseIds[0], pos,
                                              sp_layers=[1])
        pos = [1.47 / 2 + 18.034, 0, 0]
        possible_sp_pos += self.move_bookcase(self.bookcaseIds[1], pos,
                                              sp_layers=[1])

        # Randomize static object positions in environment
        if self.randomized_env:
            for key in self.random_objects:
                id = key["id"]
                prob = key["prob"]
                pose2d = key["pose2d"]
                low = key["d_low"]
                high = key["d_high"]
                visible = self.np_random.rand() <= prob
                if visible:
                    delta = self.np_random.uniform(low, high)
                    pose = np.array(pose2d) + delta
                else:
                    pose = [-20, -20, 0]
                self.move_object(id, pose)

        return possible_sp_pos


class EnvScore:
    def __init__(self, batch_size=100):
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.score_buffer = np.full(self.batch_size, np.nan)

    def add(self, val):
        self.score_buffer = np.roll(self.score_buffer, 1)
        self.score_buffer[0] = val
        return self.is_full()

    def is_full(self):
        return not np.isnan(self.score_buffer).any()

    def get_avg_score(self):
        nansum = np.nansum(self.score_buffer)
        numnonnan = np.count_nonzero(~np.isnan(self.score_buffer))
        return nansum / numnonnan
