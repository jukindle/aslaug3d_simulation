from gym import spaces
import numpy as np
import pybullet as pb
import random
from . import aslaug_base


# Aslaug environment with automatic domain randomization and sensor noise.
class AslaugEnv(aslaug_base.AslaugBaseEnv):

    def __init__(self, folder_name="", gui=False, free_cam=False,
                 recording=False, params=None, randomized_env=True):
        # Common params
        version = "v6"
        self.folder_name = folder_name
        self.soft_reset = False
        self.recording = recording
        self.success_counter = 0
        self.episode_counter = 0
        self.cum_rew = 0.0
        self.randomized_env = randomized_env

        # Initialize super class
        super().__init__(version, params, gui=gui, init_seed=None,
                         free_cam=free_cam)

        # Initialize score counter for ADR and adaption variables
        self.env_score = EnvScore(self.p["adr"]["batch_size"])
        for el in self.p["adr"]["adaptions"]:
            param = el["param"]
            self.set_param(param, el["start"])

    def setup_action_observation_spaces(self):
        self.calibrate_lidar()
        # Define action space
        accel_lims_mb = self.p["base"]["acc_mag"]
        acc_lim_joints = (self.n_joints * [self.p["joints"]["acc_mag"]])
        highs_a = (self.p["world"]["tau"]
                   * np.concatenate((accel_lims_mb, acc_lim_joints)))
        lows_a = -highs_a
        n_d = self.p["world"]["action_discretization"]
        if n_d > 0:
            self.action_space = spaces.MultiDiscrete(lows_a.shape[0] * [n_d])
            self.actions = np.linspace(lows_a, highs_a, n_d)
        else:
            self.action_space = spaces.Box(lows_a, highs_a)

        # Define observation space
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

        self.observation_space = spaces.Box(low_o, high_o)

        # Store slicing points in observation
        self.obs_slicing = [0]
        for e in (high_sp, high_mb, high_lp, high_j_p, high_j_v) \
                + n_lid * (high_scan_s,):
            self.obs_slicing.append(self.obs_slicing[-1] + e.shape[0])

    def calculate_reward(self):
        # Introducte reward variable
        reward = 0.0
        done = False
        info = {}

        # Reward: Joint limit reached
        if self.check_joint_limits_reached():
            reward += self.p["reward"]["rew_joint_limits"]
            info["done_reason"] = "joint_limits_reached"
            done = True

        # Reward: Collision
        if self.check_collision():
            reward += self.p["reward"]["rew_collision"]
            info["done_reason"] = "collision"
            done = True

        # Reward: Safety margin
        scan_ret = self.get_lidar_scan()
        scan_cal = np.concatenate([x for x in self.scan_calib if x is not None])
        scan = np.concatenate([x for x in scan_ret if x is not None])
        min_val = np.min(scan-scan_cal)
        start_dis = self.p["reward"]["dis_lidar"]
        rew_lidar = self.p["reward"]["rew_lidar_p_s"]
        if min_val <= start_dis:
            reward += rew_lidar*self.p["world"]["tau"]*(1-min_val/start_dis)
        # Reward: Base-to-setpoint orientation
        r_ang_sp = self.calculate_base_sp_angle()
        reward += ((np.abs(self.last_r_ang_sp) - np.abs(r_ang_sp))
                   * self.p["reward"]["fac_base_sp_ang"]
                   / np.pi)
        self.last_r_ang_sp = r_ang_sp

        # Reward: Timeout
        if self.step_no >= self.timeout_steps:
            reward += self.p["reward"]["rew_timeout"]  # /self.timeout_steps
            info["done_reason"] = "timeout"
            done = True

        # Reward: Goal distance
        eucl_dis, eucl_ang = self.calculate_goal_distance()
        delta_eucl_dis = self.last_eucl_dis - eucl_dis
        delta_eucl_ang = self.last_eucl_ang - eucl_ang
        reward += (self.scl_eucl_dis
                   * self.p["reward"]["fac_goal_dis_lin"] * delta_eucl_dis)
        reward += (self.scl_eucl_ang
                   * self.p["reward"]["fac_goal_dis_ang"] * delta_eucl_ang)
        self.last_eucl_dis, self.last_eucl_ang = eucl_dis, eucl_ang

        # Reward: Goal-hold
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
        self.cum_rew += reward
        return reward, done, info

    def calculate_observation(self):
        # Observation: Setpoint
        sp_pose_ee = self.get_ee_sp_transform()

        # Add noise to setpoint
        mean_lin = self.p["sensors"]["setpoint_meas"]["noise"]["mean_lin"]
        std_lin = self.p["sensors"]["setpoint_meas"]["noise"]["std_lin"]
        mean_ang = self.p["sensors"]["setpoint_meas"]["noise"]["mean_ang"]
        std_ang = self.p["sensors"]["setpoint_meas"]["noise"]["std_ang"]
        sp_pose_ee[0:3] += self.np_random.normal(mean_lin, std_lin, size=3)
        sp_pose_ee[3:6] += self.np_random.normal(mean_ang, std_ang, size=3)

        link_pose_r = self.get_link_states(self.link_mapping)
        j_pos, j_vel = self.get_joint_states(self.actuator_selection)

        # Observation: Base velocities
        mb_vel_w = self.get_base_vels()

        # Add noise to base velocities
        mean_lin = self.p["sensors"]["odometry"]["noise"]["mean_lin"]
        std_lin = self.p["sensors"]["odometry"]["noise"]["std_lin"]
        mean_ang = self.p["sensors"]["odometry"]["noise"]["mean_ang"]
        std_ang = self.p["sensors"]["odometry"]["noise"]["std_ang"]
        mb_vel_w[0:2] += self.np_random.normal(mean_lin, std_lin, size=2)
        sp_pose_ee[2:3] += self.np_random.normal(mean_ang, std_ang, size=1)

        # Observation: Lidar
        scan_ret = self.get_lidar_scan()
        scan = np.concatenate([x for x in scan_ret if x is not None])

        # Add noise to lidar sensors
        mean = self.p["sensors"]["lidar"]["noise"]["mean"]
        std = self.p["sensors"]["lidar"]["noise"]["std"]
        p_err = self.p["sensors"]["lidar"]["noise"]["p_err"]
        noise_scan = self.np_random.normal(mean, std, size=len(scan))
        mask_scan = self.np_random.uniform(size=len(scan))
        scan[mask_scan <= p_err] = self.p["sensors"]["lidar"]["range"]
        scan += noise_scan
        scan = np.clip(scan, 0, self.p["sensors"]["lidar"]["range"])

        obs = np.concatenate((sp_pose_ee, mb_vel_w, link_pose_r.flatten(),
                              j_pos, j_vel, scan))
        return obs

    def get_success_rate(self):
        return self.env_score.get_avg_score()

    def reset(self, init_state=None, init_setpoint_state=None,
              init_obstacle_grid=None, init_obstacle_locations=None):

        if self.done_info is not None:
            success = self.done_info["done_reason"] == "success"
            self.env_score.add(success)
            self.done_info = None

        # Reset internal parameters
        self.valid_buffer_scan = False
        self.episode_counter += 1
        self.step_no = 0
        self.integrated_hold_reward = 0.0
        self.cum_rew = 0.0

        # Reset setpoint only if requested
        if self.np_random.uniform() > self.p["world"]["prob_proceed"]:
            self.soft_reset = False
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
        i = 0
        while collides:
            i += 1
            if i >= 150:
                self.randomize_environment(force_new_env=True)
                i = 0
            cl = self.corridor_length
            x_min = np.max((0.0, sp_pos[0] - self.p["world"]["spawn_range_x"]))
            x_max = np.min((cl, sp_pos[0] + self.p["world"]["spawn_range_x"]))
            x_coord = self.np_random.uniform(x_min, x_max)

            robot_pos = (x_coord, self.corridor_width/2, 0.08)
            robot_init_yaw = self.np_random.uniform(-np.pi, np.pi)
            robot_ori = pb.getQuaternionFromEuler([np.pi / 2, 0, robot_init_yaw])
            pb.resetBasePositionAndOrientation(self.robotId, robot_pos, robot_ori,
                                               self.clientId)
            # Sample for all actuated joints
            for i in range(len(self.actuator_selection)):
                if self.actuator_selection[i]:
                    j = self.np_random.uniform(self.joint_limits[i, 0],
                                               self.joint_limits[i, 1])
                    pb.resetJointState(self.robotId, self.joint_mapping[i],
                                       j, 0.0, self.clientId)

            pb.stepSimulation(self.clientId)
            self.valid_buffer_scan = False
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
        robot_pos = [0, 0, 10]
        robot_ori = pb.getQuaternionFromEuler([0, 0, 0])
        model_path = 'urdf/robot/aslaug.urdf'
        robot_id = pb.loadURDF(model_path, robot_pos, robot_ori,
                               useFixedBase=True,
                               physicsClientId=self.clientId)

        return robot_id

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
        corr_l = self.np_random.uniform(*self.p["world"]["corridor_length"])
        corr_w = self.np_random.uniform(*self.p["world"]["corridor_width"])
        wall_w = self.np_random.uniform(*self.p["world"]["wall_width"])
        wall_h = self.np_random.uniform(*self.p["world"]["wall_height"])
        self.corridor_width = corr_w
        self.corridor_length = corr_l

        # Spawn walls, row 1
        pos = np.zeros(3)
        while pos[0] < corr_l:
            wall_l_i = self.np_random.uniform(*self.p["world"]["wall_length"])
            door_l_i = self.np_random.uniform(*self.p["world"]["door_length"])

            halfExtents = [wall_l_i/2, wall_w/2, wall_h/2]
            colBoxId = pb.createCollisionShape(pb.GEOM_BOX,
                                               halfExtents=halfExtents)
            visBoxId = pb.createCollisionShape(pb.GEOM_BOX,
                                               halfExtents=halfExtents)
            pos_i = pos + np.array(halfExtents*np.array((1, -1, 1)))
            id = pb.createMultiBody(0, colBoxId, visBoxId, pos_i)
            ids.append(id)
            pos += np.array((wall_l_i + door_l_i, 0, 0))
            pb.setCollisionFilterPair(self.robotId, id, -1, -1, True,
                                      self.clientId)

        # Spawn walls, row 2
        pos += np.array((0, corr_w, 0))
        while pos[0] > 0:
            wall_l_i = self.np_random.uniform(*self.p["world"]["wall_length"])
            door_l_i = self.np_random.uniform(*self.p["world"]["door_length"])

            halfExtents = [wall_l_i/2, wall_w/2, wall_h/2]
            colBoxId = pb.createCollisionShape(pb.GEOM_BOX,
                                               halfExtents=halfExtents)
            visBoxId = pb.createCollisionShape(pb.GEOM_BOX,
                                               halfExtents=halfExtents)
            pos_i = pos + np.array(halfExtents*np.array((-1, 1, 1)))
            id = pb.createMultiBody(0, colBoxId, visBoxId, pos_i)
            ids.append(id)
            pos -= np.array((wall_l_i+door_l_i, 0, 0))
            pb.setCollisionFilterPair(self.robotId, id, -1, -1, True,
                                      self.clientId)
        # Spawn shelves, row 1
        pos = np.zeros(3)
        while pos[0] < corr_l:
            shlf_l_i = self.np_random.uniform(*self.p["world"]["shelf_length"])
            shlf_w_i = self.np_random.uniform(*self.p["world"]["shelf_width"])
            shlf_h_i = self.np_random.uniform(*self.p["world"]["shelf_height"])
            sgap_l_i = self.np_random.uniform(*self.p["world"]["shelf_gap"])

            halfExtents = [shlf_l_i/2, shlf_w_i/2, shlf_h_i/2]
            colBoxId = pb.createCollisionShape(pb.GEOM_BOX,
                                               halfExtents=halfExtents)
            visBoxId = pb.createCollisionShape(pb.GEOM_BOX,
                                               halfExtents=halfExtents)
            pos_i = pos + np.array(halfExtents*np.array((1, 1, 1)))
            if abs(pos_i[0] - 4.735) >= 0.735 + halfExtents[0] and \
                    abs(pos_i[0] - 12.735) >= 0.735 + halfExtents[0]:
                id = pb.createMultiBody(0, colBoxId, visBoxId, pos_i)
                ids.append(id)
            pos += np.array((shlf_l_i + sgap_l_i, 0, 0))
            pb.setCollisionFilterPair(self.robotId, id, -1, -1, True,
                                      self.clientId)

        # Spawn shelves, row 2
        pos += np.array((0, corr_w, 0))
        while pos[0] > 0:
            shlf_l_i = self.np_random.uniform(*self.p["world"]["shelf_length"])
            shlf_w_i = self.np_random.uniform(*self.p["world"]["shelf_width"])
            shlf_h_i = self.np_random.uniform(*self.p["world"]["shelf_height"])
            sgap_l_i = self.np_random.uniform(*self.p["world"]["shelf_gap"])

            halfExtents = [shlf_l_i/2, shlf_w_i/2, shlf_h_i/2]
            colBoxId = pb.createCollisionShape(pb.GEOM_BOX,
                                               halfExtents=halfExtents)
            visBoxId = pb.createCollisionShape(pb.GEOM_BOX,
                                               halfExtents=halfExtents)
            pos_i = pos + np.array(halfExtents*np.array((-1, -1, 1)))
            id = pb.createMultiBody(0, colBoxId, visBoxId, pos_i)
            ids.append(id)
            pos -= np.array((shlf_l_i+sgap_l_i, 0, 0))
            pb.setCollisionFilterPair(self.robotId, id, -1, -1, True,
                                      self.clientId)

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

        kallax1 = pb.loadURDF(model_path, [1.47 / 2 + 4.0, 0, 0],
                              useFixedBase=True, physicsClientId=self.clientId)
        kallax2 = pb.loadURDF(model_path, [1.47 / 2 + 12.0, 0, 0],
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

    def randomize_environment(self, force_new_env=False):
        if force_new_env or \
                self.np_random.uniform() <= self.p["world"]["prob_new_env"]:
            for id in self.additionalIds:
                pb.removeBody(id, physicsClientId=self.clientId)
            self.additionalIds = self.spawn_additional_objects()
        # Randomize bookcases
        layers = self.p["setpoint"]["layers"]
        possible_sp_pos = []
        pos = [1.47 / 2 + 4, 0, 0]
        possible_sp_pos += self.move_bookcase(self.bookcaseIds[0], pos,
                                              sp_layers=layers)
        pos = [1.47 / 2 + 12, 0, 0]
        possible_sp_pos += self.move_bookcase(self.bookcaseIds[1], pos,
                                              sp_layers=layers)

        return possible_sp_pos

    def calibrate_lidar(self):
        robot_pos = [0, 0, 10]
        robot_ori = pb.getQuaternionFromEuler([0, 0, 0])
        model_path = 'urdf/calibration/ridgeback_lidar_calib.urdf'
        calib_id = pb.loadURDF(model_path, robot_pos, robot_ori,
                               useFixedBase=True,
                               physicsClientId=self.clientId)
        robot_pos = (0, 0, 10)
        robot_ori = pb.getQuaternionFromEuler([np.pi/2, 0, np.pi/2])
        pb.resetBasePositionAndOrientation(self.robotId, robot_pos, robot_ori,
                                           self.clientId)
        pb.stepSimulation(self.clientId)
        scan_ret = self.get_lidar_scan()
        self.scan_calib = scan_ret
        pb.removeBody(calib_id, self.clientId)


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
