from gym import spaces
import numpy as np
import pybullet as pb
import random
from . import aslaug_base
import cv2
from scipy.signal import convolve2d


# Aslaug environment with automatic domain randomization, sensor noise,
# harmonic potential field path, fast HPT, adapted maximum velocity of base
# and improved GUI
class AslaugEnv(aslaug_base.AslaugBaseEnv):

    def __init__(self, folder_name="", gui=False, free_cam=False,
                 recording=False, params=None, randomized_env=True):
        # Common params
        version = "v12"
        self.folder_name = folder_name
        self.soft_reset = False
        self.recording = recording
        self.success_counter = 0
        self.episode_counter = 0
        self.cum_rew = 0.0
        self.randomized_env = randomized_env
        self.scan_calib = None

        self.collision_links = ["top_link", "chassis_link", "panda_link1",
                                "panda_link2", "panda_link3", "panda_link4",
                                "panda_link5", "panda_link6", "panda_link7",
                                "panda_hand", "panda_leftfinger",
                                "panda_rightfinger"]

        # Initialize super class
        super().__init__(version, params, gui=gui, init_seed=None,
                         free_cam=free_cam)

        # Adjust joint limit of joint 4 to avoid self collision in 2D config
        self.joint_limits[3, 0] = -2.7

        # Initialize score counter for ADR and adaption variables
        self.env_score = EnvScore(self.p["adr"]["batch_size"])
        for ele in self.p["adr"]["adaptions"][::-1]:
            for el in ele:
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
            n_da = n_d + self.p["world"]["use_stop_action"]
            self.action_space = spaces.MultiDiscrete(lows_a.shape[0] * [n_da])
            self.actions = np.linspace(lows_a, highs_a, n_d)
        else:
            if self.p["world"]["use_stop_action"]:
                lows_a = np.append(lows_a, [0, 0])
                highs_a = np.append(highs_a, [1, 1])
            self.action_space = spaces.Box(lows_a, highs_a)

        # Define observation space
        high_sp = np.array([self.p["world"]["size"]] * 2 + [np.pi])
        low_sp = -high_sp
        high_mb = np.array([self.p['base']["vel_mag_lin"]]*2 + [self.p['base']['vel_mag_ang']])
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
        rew_lidar = self.p["reward"]["rew_lidar_p_m"]
        if min_val <= start_dis:
            vel_norm = np.linalg.norm(self.get_base_vels()[:2])
            reward += rew_lidar*self.p["world"]["tau"]*(1-min_val/start_dis)*vel_norm
        # Reward: Base-to-setpoint orientation
        r_ang_sp = self.calculate_base_sp_angle()
        reward += ((np.abs(self.last_r_ang_sp) - np.abs(r_ang_sp))
                   * self.p["reward"]["fac_base_sp_ang"]
                   / np.pi)
        self.last_r_ang_sp = r_ang_sp

        # Reward: Timeout
        reward += self.p["reward"]["rew_timeout"] / self.timeout_steps
        if self.step_no >= self.timeout_steps:
            info["done_reason"] = "timeout"
            done = True

        # Reward: Goal distance
        eucl_dis, eucl_ang = self.calculate_goal_distance()
        delta_eucl_dis = self.last_eucl_dis - eucl_dis
        delta_eucl_ang = self.last_eucl_ang - eucl_ang
        # if delta_eucl_dis > 0:
        reward += (self.scl_eucl_dis
                   * self.p["reward"]["fac_goal_dis_lin"] * delta_eucl_dis)
        self.last_eucl_dis = eucl_dis
        if delta_eucl_ang > 0:
            reward += (self.scl_eucl_ang
                       * self.p["reward"]["fac_goal_dis_ang"] * delta_eucl_ang)
            self.last_eucl_ang = eucl_ang

        # Reward from optimal path
        dis_to_p, rem_dis = self.get_path_stats()

        delta_dtp = self.last_dis_to_path - dis_to_p
        delta_rem_dis = self.last_remaining_dis - rem_dis

        fac_dtp = self.p["reward"]["rew_path_dis_p_m"]
        fac_rem_dis = self.p["reward"]["rew_path_total"]

        reward += fac_dtp*delta_dtp
        reward += fac_rem_dis*delta_rem_dis/self.total_path_length

        self.last_dis_to_path = dis_to_p
        self.last_remaining_dis = rem_dis

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
            dis_f = (1.0 - eucl_dis / self.p["setpoint"]["tol_lin_mag"])**2
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
        std_lin = self.p["sensors"]["setpoint_meas"]["std_lin"]
        std_ang = self.p["sensors"]["setpoint_meas"]["std_ang"]
        sp_pose_ee[0:3] *= self.np_random.normal(1, std_lin, size=3)
        sp_pose_ee[3:6] *= self.np_random.normal(1, std_ang, size=3)
        sp_pose_ee = np.array((sp_pose_ee[0], sp_pose_ee[1], sp_pose_ee[5]))
        link_pose_r = self.get_link_states(self.link_mapping)
        j_pos, j_vel = self.get_joint_states(self.actuator_selection)

        # Observation: Base velocities
        mb_vel_w = self.get_base_vels()

        # Add noise to base velocities
        std_lin = self.p["sensors"]["odometry"]["std_lin"]
        std_ang = self.p["sensors"]["odometry"]["std_ang"]
        mb_vel_w[0:2] *= self.np_random.normal(1, std_lin, size=2)
        mb_vel_w[2:3] *= self.np_random.normal(1, std_ang, size=1)

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
            sp_pos = self.reset_setpoint(max_dis=self.p["world"]["spawn_range_x"])
            self.generate_occmap_path()
            return self.calculate_observation()
        else:
            self.sp_history = []

        # Reset internal state
        self.state = {"base_vel": np.array([0.0, 0.0, 0.0]),
                      "joint_vel": np.array(7 * [0.0])}

        # Reset environment
        up = self.p['joints']['static_act_noise_mag']
        self.fixed_joint_states = (np.array(self.p["joints"]["init_states"])
                                   + self.np_random.uniform(-up, up))
        for i in range(len(self.joint_mapping)):
            pb.resetJointState(self.robotId, self.joint_mapping[i],
                               self.fixed_joint_states[i],
                               0.0, self.clientId)

        self.possible_sp_pos = self.randomize_environment()

        # Reset robot base
        pb.resetBaseVelocity(self.robotId, [0, 0, 0], [0, 0, 0], self.clientId)

        # Reset setpoint
        sp_pos = self.reset_setpoint()

        # Reset robot arm
        collides = True
        n_tries = 0
        pb.stepSimulation(self.clientId)
        while collides:
            n_tries += 1
            if n_tries >= 150:
                self.randomize_environment(force_new_env=True)
                n_tries = 0
            cl = self.corridor_length

            x_min = np.max((0.0, sp_pos[0] - self.p["world"]["spawn_range_x"]))
            x_max = np.min((cl, sp_pos[0] + self.p["world"]["spawn_range_x"]))
            x_coord = self.np_random.uniform(x_min, x_max)

            robot_pos = (x_coord, self.corridor_width/2, 0.08)
            robot_init_yaw = self.np_random.uniform(-np.pi, np.pi)
            robot_ori = pb.getQuaternionFromEuler([0, 0,
                                                   robot_init_yaw])
            pb.resetBasePositionAndOrientation(self.robotId, robot_pos,
                                               robot_ori, self.clientId)
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

        self.robot_init_pos = robot_pos
        self.sp_init_pos = sp_pos
        # Reset setpoint variables
        self.occmap.set_sp(sp_pos)
        self.generate_occmap_path()
        self.reset_setpoint_normalization()

        # Initialize human poses
        for human in self.humans:
            h_s_x = self.np_random.uniform(self.sp_init_pos[0]-7.5, self.sp_init_pos[0]+7.5)
            h_s_y = self.np_random.uniform(-0.5, self.corridor_width+0.5)
            h_e_x = self.np_random.uniform(self.sp_init_pos[0]-7.5, self.sp_init_pos[0]+7.5)
            h_e_y = self.np_random.uniform(-0.5, self.corridor_width+0.5)
            human.set_start_end([h_s_x, h_s_y], [h_e_x, h_e_y])
            human.setEnabled(self.np_random.uniform() <= self.p['world']['p_spawn_human'])
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

        self.reset_setpoint_normalization()

        self.soft_reset = False
        self.sp_history.append(sp_pos.tolist())
        self.occmap.set_sp(sp_pos)
        return sp_pos

    def reset_setpoint_normalization(self):
        eucl_dis, eucl_ang = self.calculate_goal_distance()
        self.last_eucl_dis, self.last_eucl_ang = eucl_dis, eucl_ang
        if eucl_dis == 0:
            self.scl_eucl_dis = 0
        else:
            self.scl_eucl_dis = 1 / (self.last_eucl_dis+1e-9)
        if eucl_ang == 0:
            self.scl_eucl_ang = 0
        else:
            self.scl_eucl_ang = 1 / (self.last_eucl_ang+1e-9)
        self.last_r_ang_sp = self.calculate_base_sp_angle()
        if self.last_r_ang_sp == 0:
            self.scl_r_ang_sp = 0
        else:
            self.scl_r_ang_sp = 1 / self.last_r_ang_sp
        self.sp_hold_time = 0.0

    def spawn_robot(self):
        # Spawn robot
        robot_pos = [0, 0, 10]
        robot_ori = pb.getQuaternionFromEuler([0, 0, 0])
        model_path = 'urdf/robot/mopa/mopa.urdf'
        robot_id = pb.loadURDF(model_path, robot_pos, robot_ori,
                               useFixedBase=True,
                               physicsClientId=self.clientId,
                               flags=pb.URDF_USE_SELF_COLLISION|pb.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)

        # Disable panda base collision
        pairs = [("camera_rack", x) for x in ["panda_hand", "panda_leftfinger", "panda_rightfinger", "panda_link5", "panda_link6", "panda_link7"]]
        self.configure_self_collisions(robot_id, pairs)

        return robot_id

    def configure_ext_collisions(self, bodyExt, body, enabled_links):
        for j in range(pb.getNumJoints(body, physicsClientId=self.clientId)):
            info_j = pb.getJointInfo(body, j, physicsClientId=self.clientId)
            link_name_j = info_j[12].decode("utf-8")
            idx_j = info_j[0]

            enabled = link_name_j in enabled_links
            pb.setCollisionFilterPair(body, bodyExt, idx_j, -1, enabled,
                                      self.clientId)

    def configure_self_collisions(self, body, enabled_pairs):
        pairs = ["{}|{}".format(x, y) for x, y in enabled_pairs]
        for j in range(pb.getNumJoints(body, physicsClientId=self.clientId)):
            info_j = pb.getJointInfo(body, j, physicsClientId=self.clientId)
            link_name_j = info_j[12].decode("utf-8")
            idx_j = info_j[0]

            for k in range(pb.getNumJoints(body, physicsClientId=self.clientId)):
                info_k = pb.getJointInfo(body, k, physicsClientId=self.clientId)
                link_name_k = info_k[12].decode("utf-8")
                idx_k = info_k[0]

                s1 = "{}|{}".format(link_name_j, link_name_k)
                s2 = "{}|{}".format(link_name_k, link_name_j)
                enabled = s1 in pairs or s2 in pairs
                pb.setCollisionFilterPair(body, body, idx_j, idx_k, enabled,
                                          self.clientId)

    def set_collisionpair(self, bodyA, bodyB, linknameA, linknameB, collision):
        linkA = None
        linkB = None
        for j in range(pb.getNumJoints(bodyA,
                                       physicsClientId=self.clientId)):
            info = pb.getJointInfo(bodyA, j,
                                   physicsClientId=self.clientId)
            link_name = info[12].decode("utf-8")
            idx = info[0]
            print(idx, link_name)
            if link_name == linknameA:
                linkA = idx
        for j in range(pb.getNumJoints(bodyB,
                                       physicsClientId=self.clientId)):
            info = pb.getJointInfo(bodyB, j,
                                   physicsClientId=self.clientId)
            link_name = info[12].decode("utf-8")
            idx = info[0]
            if link_name == linknameB:
                linkB = idx

        if None not in [linkA, linkB]:
            pb.setCollisionFilterPair(bodyA, bodyB, linkA, linkB, collision,
                                      self.clientId)
            return True
        return False

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

        # Reset occupancy map
        om_res = self.p['world']['HPF']['res']
        self.occmap = OccupancyMap(0, corr_l, 0, corr_w, om_res)

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

            # Create room walls
            if self.np_random.uniform(0, 1) <= 0.5:
                he_wall_w_w = self.np_random.uniform(0, 3)
                he_wall_d_w = self.np_random.uniform(0, 3)
                he_w = [wall_w/2, he_wall_w_w, wall_h/2]
                he_d = [wall_w/2, he_wall_d_w, wall_h/2]
                colBoxId_w = pb.createCollisionShape(pb.GEOM_BOX,
                                                   halfExtents=he_w)
                visBoxId_w = pb.createCollisionShape(pb.GEOM_BOX,
                                                   halfExtents=he_w)
                colBoxId_d = pb.createCollisionShape(pb.GEOM_BOX,
                                                   halfExtents=he_d)
                visBoxId_d = pb.createCollisionShape(pb.GEOM_BOX,
                                                   halfExtents=he_d)
                pos_i = pos + np.array(he_w*np.array((1, -1, 1)))
                pos_i[0] += wall_l_i - wall_w
                id_w = pb.createMultiBody(0, colBoxId_w, visBoxId_w, pos_i)
                ids.append(id_w)
                pos_i = pos + np.array(he_d*np.array((1, -1, 1)))
                pos_i[0] += wall_l_i + door_l_i
                id_d = pb.createMultiBody(0, colBoxId_d, visBoxId_d, pos_i)
                ids.append(id_d)

            pos += np.array((wall_l_i + door_l_i, 0, 0))
            self.configure_ext_collisions(id, self.robotId, self.collision_links)

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

            # Create room walls
            if self.np_random.uniform(0, 1) <= 0.5:
                he_wall_w_w = self.np_random.uniform(0, 3)
                he_wall_d_w = self.np_random.uniform(0, 3)
                he_w = [wall_w/2, he_wall_w_w, wall_h/2]
                he_d = [wall_w/2, he_wall_d_w, wall_h/2]
                colBoxId_w = pb.createCollisionShape(pb.GEOM_BOX,
                                                   halfExtents=he_w)
                visBoxId_w = pb.createCollisionShape(pb.GEOM_BOX,
                                                   halfExtents=he_w)
                colBoxId_d = pb.createCollisionShape(pb.GEOM_BOX,
                                                   halfExtents=he_d)
                visBoxId_d = pb.createCollisionShape(pb.GEOM_BOX,
                                                   halfExtents=he_d)
                pos_i = pos + np.array(he_w*np.array((-1, 1, 1)))
                pos_i[0] -= wall_l_i
                id_w = pb.createMultiBody(0, colBoxId_w, visBoxId_w, pos_i)
                ids.append(id_w)
                pos_i = pos + np.array(he_d*np.array((-1, 1, 1)))
                pos_i[0] -= wall_l_i + door_l_i
                id_d = pb.createMultiBody(0, colBoxId_d, visBoxId_d, pos_i)
                ids.append(id_d)

            pos -= np.array((wall_l_i+door_l_i, 0, 0))
            self.configure_ext_collisions(id, self.robotId, self.collision_links)

        sg = SpawnGrid(corr_l*2, corr_w, res=0.01, min_dis=self.p["world"]["min_clearance"])
        sg.add_shelf(4+1.47/2, 1.47, 0.39, 0)
        sg.add_shelf(12+1.47/2, 1.47, 0.39, 0)
        self.occmap.add_rect([4+1.47/2, 0.39/2], 1.47, 0.39)
        self.occmap.add_rect([12+1.47/2, 0.39/2], 1.47, 0.39)
        # Spawn shelves, row 1
        pos = np.zeros(3)
        while pos[0] < corr_l:
            shlf_l_i = self.np_random.uniform(*self.p["world"]["shelf_length"])
            mw = sg.get_max_width(pos[0]+shlf_l_i/2, shlf_l_i, 0)
            width_lims = self.p["world"]["shelf_width"].copy()
            width_lims[1] = min(width_lims[1], mw)
            shlf_w_i = self.np_random.uniform(*width_lims)

            shlf_h_i = self.np_random.uniform(*self.p["world"]["shelf_height"])
            sgap_l_i = self.np_random.uniform(*self.p["world"]["shelf_gap"])

            halfExtents = [shlf_l_i/2, shlf_w_i/2, shlf_h_i/2]
            colBoxId = pb.createCollisionShape(pb.GEOM_BOX,
                                               halfExtents=halfExtents)
            visBoxId = pb.createCollisionShape(pb.GEOM_BOX,
                                               halfExtents=halfExtents)
            pos_i = pos + np.array(halfExtents*np.array((1, 1, 1)))
            margin = 0.3
            if abs(pos_i[0] - 4.735) >= 0.735 + halfExtents[0] + margin and \
                    abs(pos_i[0] - 12.735) >= 0.735 + halfExtents[0] + margin:
                sg.add_shelf(pos[0]+shlf_l_i/2, shlf_l_i, shlf_w_i, 0)
                id = pb.createMultiBody(0, colBoxId, visBoxId, pos_i)
                ids.append(id)

                self.occmap.add_rect([pos[0]+shlf_l_i/2.0, pos[1]+shlf_w_i/2],
                                     shlf_l_i, shlf_w_i)
                pos += np.array((shlf_l_i + sgap_l_i, 0, 0))
                self.configure_ext_collisions(id, self.robotId, self.collision_links)

            else:
                pos += np.array((0.05, 0, 0))

        # Spawn shelves, row 2
        pos += np.array((0, corr_w, 0))
        while pos[0] > 0:
            shlf_l_i = self.np_random.uniform(*self.p["world"]["shelf_length"])
            mw = sg.get_max_width(pos[0]-shlf_l_i/2, shlf_l_i, 1)
            width_lims = self.p["world"]["shelf_width"].copy()
            width_lims[1] = min(width_lims[1], mw)
            shlf_w_i = self.np_random.uniform(*width_lims)
            sg.add_shelf(pos[0]-shlf_l_i/2, shlf_l_i, shlf_w_i, 1)
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

            self.occmap.add_rect([pos[0]-shlf_l_i/2.0, pos[1]-shlf_w_i/2],
                                 shlf_l_i, shlf_w_i)
            pos -= np.array((shlf_l_i+sgap_l_i, 0, 0))
            self.configure_ext_collisions(id, self.robotId, self.collision_links)

        for id in ids:
            for human in self.humans:
                pb.setCollisionFilterPair(human.leg_l, id, -1, -1, False,
                                          self.clientId)
                pb.setCollisionFilterPair(human.leg_r, id, -1, -1, False,
                                          self.clientId)
        # print(sg.matrix1)
        # print(sg.matrix0)
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(sg.matrix0.astype(float))
        # plt.show()
        return ids

    def calculate_goal_distance(self):
        sp_pose_ee = self.get_ee_sp_transform()
        # Ignore x coord. if 2D locked
        if self.p['setpoint']['2D_locked']:
            eucl_dis = np.linalg.norm(sp_pose_ee[0:2])
        else:
            eucl_dis = np.linalg.norm(sp_pose_ee[0:3])
        eucl_ang = np.linalg.norm(sp_pose_ee[3:6])
        return eucl_dis, eucl_ang

    def calculate_base_sp_angle(self):
        base_pose_sp = self.get_base_sp_transform()
        return base_pose_sp[5]

    def spawn_kallax(self):
        '''
        Prepares the simulation by spawning n bookcases.

        Args:
            n (int): Number of bookcases.
        Returns:
            list: List of bookcase IDs.
        '''
        model_path = 'urdf/kallax/kallax_large_easy.urdf'

        kallax1 = pb.loadURDF(model_path, [1.47 / 2 + 4.0, 0, 0],
                              useFixedBase=True, physicsClientId=self.clientId)
        kallax2 = pb.loadURDF(model_path, [1.47 / 2 + 12.0, 0, 0],
                              useFixedBase=True, physicsClientId=self.clientId)
        self.configure_ext_collisions(kallax1, self.robotId, self.collision_links)
        self.configure_ext_collisions(kallax2, self.robotId, self.collision_links)
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

        # Sample possible setpoint positions
        sp_pos = []
        p_noise = self.p['setpoint']['noise']
        for l in sp_layers:
            z = 0.037 + (0.33 + 0.025) * l
            y = 0.195
            # for dx in [+0.1775, -0.1775, +0.5325, -0.5325]:
            dx = 0.0
            pos_i = pos + Rt.dot(np.array([dx, y, z]))
            nx = self.np_random.uniform(*p_noise['range_x'])
            ny = self.np_random.uniform(*p_noise['range_y'])
            nz = self.np_random.uniform(*p_noise['range_z'])

            pos_i += np.array((nx, ny, nz))
            sp_pos.append(pos_i)

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
        robot_ori = pb.getQuaternionFromEuler([0, 0, 0])
        pb.resetBasePositionAndOrientation(self.robotId, robot_pos, robot_ori,
                                           self.clientId)
        pb.stepSimulation(self.clientId)
        scan_ret = self.get_lidar_scan()
        self.scan_calib = scan_ret
        pb.removeBody(calib_id, self.clientId)
        return scan_ret

    def get_lidar_calibration(self):
        if self.scan_calib is None:
            return self.calibrate_lidar()
        else:
            return self.scan_calib

    def get_ee_pose(self):
        state_ee = pb.getLinkState(self.robotId, self.eeLinkId,
                                   False, False, self.clientId)
        ee_pos_w, ee_ori_w = state_ee[4:6]
        return ee_pos_w, ee_ori_w

    def generate_occmap_path(self):
        ee_pos_w, ee_ori_w = self.get_ee_pose()
        pos = [ee_pos_w[0], ee_pos_w[1]]
        self.path, path_idx = self.occmap.generate_path(pos, n_its=25000)
        self.path = np.array(self.path)
        # self.occmap.visualize_path(path_idx)
        dis_to_p, rem_dis = self.get_path_stats()
        self.last_dis_to_path = dis_to_p
        self.total_path_length = rem_dis
        self.last_remaining_dis = rem_dis

    def get_path_stats(self):
        pos_ee, _ = self.get_ee_pose()
        pos_ee = np.array(pos_ee[0:2])
        deltas = self.path - pos_ee
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        idx = np.argmin(dist_2)

        candidates = []
        if idx > 0:
            idx_next = idx
            candidates.append((self.path[(idx-1):idx+1, :], idx_next))
        if idx < self.path.shape[0]-1:
            idx_next = idx + 1
            candidates.append((self.path[idx:(idx+2), :], idx_next))

        results_p = []
        results_d = []
        results_idx_next = []

        for cand, idx_next in candidates:
            p, d = self.get_nearest_point_on_line(cand, pos_ee)
            results_p.append(p)
            results_d.append(d)
            results_idx_next.append(idx_next)
        idx_r = np.argmin(results_d)

        nearest_point = np.array(results_p[idx_r])
        distance = results_d[idx_r]
        idx_next = results_idx_next[idx_r]

        dis_to_next_p = np.linalg.norm(nearest_point - self.path[idx_next, :])
        total_path_dis = dis_to_next_p + self.path_length_from_index(idx_next)

        return distance, total_path_dis

    def get_nearest_point_on_line(self, pts_l, pt):
        x1, y1 = pts_l[0, :]
        x2, y2 = pts_l[1, :]
        x3, y3 = pt

        a1, a2 = x2-x1, y2-y1
        if a2 == 0:
            a2 = 1e-3
        anorm = np.sqrt(a1**2 + a2**2)
        a1, a2 = a1 / anorm, a2 / anorm

        n1, n2 = 1, -a1/a2

        divid = (n2/n1*a1-a2)
        if divid == 0:
            divid = 1e-3
        t = (n2/n1*(x3-x1)+y1-y3) / divid
        t = max(0, min(1, t))

        o1, o2 = x1+t*a1, y1+t*a2

        dis = np.sqrt((o1-x3)**2 + (o2-y3)**2)
        return [o1, o2], dis

    def path_length_from_index(self, idx):
        if idx >= self.path.shape[0] - 1:
            return 0.0

        vecs = self.path[(idx+1):, :] - self.path[idx:-1, :]
        diss = np.linalg.norm(vecs, axis=1)
        return np.sum(diss)


class SpawnGrid:
    def __init__(self, length, width, res=0.1, min_dis=1.0):
        self.length = length
        self.width = width
        self.res = res
        self.min_dis = min_dis

        self.clear()

    def clear(self):
        self.matrix1 = np.zeros((self.discretize(self.length),
                                 self.discretize(self.width)))
        self.matrix0 = np.zeros((self.discretize(self.length),
                                 self.discretize(self.width)))

    def discretize(self, x):
        return int(round(x/self.res))

    def undiscretize(self, idx):
        return self.res*(idx+0.5)

    def get_idx(self, x, y):
        return self.discretize(x), self.discretize(y)

    def get_max_width(self, x, length, wall):
        matrix = self.matrix1 if wall == 0 else self.matrix0

        b_l = max(0, min(matrix.shape[0], self.discretize(x-length/2-self.min_dis/2.0)))
        b_r = max(0, min(matrix.shape[0], self.discretize(x+length/2+self.min_dis/2.0)))
        min_idxs = self.discretize(self.min_dis)
        for ib in range(matrix.shape[1]-min_idxs):
            if (matrix[b_l:b_r, 0:ib+min_idxs] == 1).any():
                break

        return self.undiscretize(ib)

    def add_shelf(self, x, length, width, wall):
        b_l = self.discretize(x-length/2.0-self.min_dis/2.0)
        b_r = self.discretize(x+length/2.0+self.min_dis/2.0)
        n_w = self.discretize(width)
        if wall == 0:
            self.matrix0[b_l:b_r, -n_w:] = 1
        if wall == 1:
            self.matrix1[b_l:b_r, -n_w:] = 1


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
        resp = nansum / numnonnan
        return 0 if np.isnan(resp) else resp


class OccupancyMap:
    def __init__(self, x_l, x_u, y_l, y_u, res):
        self.x_l, self.x_u, self.y_l, self.y_u = x_l, x_u, y_l, y_u
        self.res = res
        self.w_obj = -3
        self.w_sp = 1
        self.original_map = None
        self.reset()

    def add_rect(self, pos, dx, dy):
        p_u = self.coord_to_idx([pos[0]+dx/2.0, pos[1]+dy/2.0])
        p_l = self.coord_to_idx([pos[0]-dx/2.0, pos[1]-dy/2.0])
        self.map[p_l[0]:p_u[0], p_l[1]:p_u[1]] = self.w_obj

    def set_sp(self, pos, tol_radius=0.2):
        if self.original_map is None:
            self.original_map = self.map.copy()

        self.map = self.original_map.copy()
        self.add_sp(pos, tol_radius)

    def add_sp(self, pos, tol_radius=0.2):
        pos_idx = self.coord_to_idx(pos)

        tol_l = self.coord_to_idx([pos[0]-tol_radius, pos[1]-tol_radius])
        tol_u = self.coord_to_idx([pos[0]+tol_radius, pos[1]+tol_radius])
        self.map[tol_l[0]:tol_u[0], tol_l[1]:tol_u[1]] = 0
        self.map[pos_idx[0], pos_idx[1]] = self.w_sp
        self.pos_sp = pos
        self.idx_sp = pos_idx

    def generate_path(self, pos, n_its=5000):
        harm = self.find_harmonic_field_fast(self.idx_sp, self.coord_to_idx(pos), n_its)
        path, path_idx = self.find_path(harm, pos)
        #self.visualize_path(path_idx, harm)
        return path, path_idx

    def find_harmonic_field(self, n_its=5000):
        harm = self.map.copy()
        obj_sel = self.map == self.w_obj
        sp_sel = self.map == self.w_sp

        kernel = np.ones((3, 3))/8.0
        kernel[1, 1] = 0
        harm_last = harm.copy()
        for i in range(n_its):
            harm = convolve2d(harm, kernel, mode='same')

            harm[obj_sel] = self.w_obj
            harm[sp_sel] = self.w_sp

            diff = np.linalg.norm(harm-harm_last)
            if diff < 1e-9:
                break
            harm_last = harm.copy()

        return harm

    def find_harmonic_field_fast(self, idx_init, idx_sp, n_its=5000):
        harm_original = self.map.copy()
        harm = self.map.copy()

        kernel = np.ones((3, 3))/8.0
        kernel[1, 1] = 0

        margin = 1.0
        left_cut_idx = min(idx_init[0], idx_sp[0])
        left_cut_idx = int(round(max(0, left_cut_idx-margin*self.res)))
        right_cut_idx = max(idx_init[0], idx_sp[0])
        right_cut_idx = int(round(min(harm.shape[0], right_cut_idx+margin*self.res)))

        harm = harm[left_cut_idx:right_cut_idx, :]
        harm[0, :] = self.w_obj
        harm[-1, :] = self.w_obj

        harm_last = harm.copy()

        obj_sel = harm_last == self.w_obj
        sp_sel = harm_last == self.w_sp
        for i in range(n_its):
            harm = convolve2d(harm, kernel, mode='same')

            harm[obj_sel] = self.w_obj
            harm[sp_sel] = self.w_sp

            diff = np.linalg.norm(harm-harm_last)
            if diff < 1e-9:
                break
            harm_last = harm.copy()

        harm_original[:, :] = self.w_obj
        harm_original[left_cut_idx:right_cut_idx, :] = harm
        return harm_original

    def find_path(self, harm, pos):
        x, y = self.coord_to_idx(pos)
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)]

        path = [pos]
        path_px = [[x, y]]
        for i in range(int(self.res*(self.x_u-self.x_l)*10)):
            values = []
            for dir in dirs:
                if (x+dir[0] < harm.shape[0]-1 and x-dir[0] > 0
                        and y+dir[1] < harm.shape[1]-1 and y-dir[1] > 0):
                    values.append(harm[x+dir[0], y+dir[1]])
                else:
                    values.append([-np.inf])
            best_dir = dirs[np.argmax(values)]
            x, y = x + best_dir[0], y + best_dir[1]
            path.append(self.idx_to_coord([x, y]))
            path_px.append([x, y])

            if self.idx_sp[0] == x and self.idx_sp[1] == y:
                break
        path[-1] = self.pos_sp[0:2]
        return path, path_px

    def visualize_path(self, path_idx, harm=None):
        if harm is None:
            map = self.map.copy()
        else:
            map = harm.copy()
        for idx in path_idx:
            map[idx[0], idx[1]] = self.w_sp
        map[self.idx_sp[0], self.idx_sp[1]] = 1
        self.visualize(map)

    def visualize(self, map):
        max_v = np.max(map)
        min_v = np.min(map)
        img_viz = ((map-min_v)/(max_v-min_v)*254.0).astype(np.uint8)

        scl = int(1500/img_viz.shape[0])
        width = int(img_viz.shape[1] * scl)
        img_viz = cv2.resize(img_viz, (width, 1500),
                             interpolation=cv2.INTER_NEAREST)
        img_viz = np.flip(img_viz.T, axis=0)
        img_viz = cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB)
        print(img_viz)
        cv2.imshow("Occupancy map", img_viz)
        cv2.waitKey()

    def reset(self):
        n_x = int(round(self.res*(self.x_u-self.x_l)))+2
        n_y = int(round(self.res*(self.y_u-self.y_l)))+2
        self.map = np.zeros((n_x, n_y), dtype=float)
        self.map[0, :] = self.w_obj
        self.map[-1, :] = self.w_obj
        self.map[:, 0] = self.w_obj
        self.map[:, -1] = self.w_obj

    def coord_to_idx(self, pos):
        idx_x = self.res*(pos[0]-self.x_l) + 1
        idx_y = self.res*(pos[1]-self.y_l) + 1

        idx_x = max(1, min(self.map.shape[0]-1, idx_x))
        idx_y = max(1, min(self.map.shape[1]-1, idx_y))

        return int(round(idx_x)), int(round(idx_y))

    def idx_to_coord(self, idx):
        coord_x = (idx[0]-1)/self.res+self.x_l
        coord_y = (idx[1]-1)/self.res+self.y_l

        return coord_x, coord_y
