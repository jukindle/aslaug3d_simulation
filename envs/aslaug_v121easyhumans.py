from gym import spaces
import numpy as np
import pybullet as pb
import random
from . import aslaug_base
from .base.objects import AslaugRobot, AslaugKallax


# Aslaug environment with automatic domain randomization, sensor noise,
# harmonic potential field path, fast HPT, adapted maximum velocity of base
# and improved GUI
class AslaugEnv(aslaug_base.AslaugBaseEnv):

    def __init__(self, folder_name="", gui=False, free_cam=False,
                 recording=False, params=None, randomized_env=True):
        # Common params
        version = "v200"
        self.folder_name = folder_name
        self.soft_reset = False
        self.recording = recording
        self.success_counter = 0
        self.episode_counter = 0
        self.cum_rew = 0.0
        self.randomized_env = randomized_env
        self.scan_calib = None
        self.last_ee_vel = None

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
        # Calibrate lidar scan
        self.robot.calibrate_lidar()
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
        if self.robot.isColliding():
            reward += self.p["reward"]["rew_collision"]
            info["done_reason"] = "collision"
            done = True

        # Reward: Safety margin
        scan_ret = self.robot.getLidarScan()
        scan_cal = np.concatenate([x for x in self.scan_calib if x is not None])
        scan = np.concatenate([x for x in scan_ret if x is not None])
        min_val = np.min(scan-scan_cal)
        start_dis = self.p["reward"]["dis_lidar"]
        rew_lidar = self.p["reward"]["rew_lidar_p_m"]
        if min_val <= start_dis:
            vel_norm = np.linalg.norm(self.robot.getBaseVelocity()[:2])
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

        # Reward: Goal-hold
        # Replaced by EE velocity
        ee_vels = self.robot.getEEVelocity()
        ee_speed = np.linalg.norm(ee_vels)
        if eucl_dis <= self.p["setpoint"]["tol_lin_mag"] and \
                eucl_ang <= self.p["setpoint"]["tol_ang_mag"]:
            ee_vels = self.getEEVelocity()
            ee_speed = np.linalg.norm(ee_vels)
            if self.last_ee_vel is None:
                self.last_ee_vel = ee_speed
                self.ee_vel_fac = self.p['reward']['ee_vel_target_reduction']/max(ee_speed, 0.01)

            delta_vel = self.last_ee_vel - ee_speed

            reward += delta_vel*self.ee_vel_fac

            if ee_speed <= self.p['reward']['ee_vel_limit']:
                done = True
                info["done_reason"] = "success"
                reward += self.p["reward"]["rew_goal_reached"]
                self.success_counter += 1
                self.soft_reset = True
                self.last_ee_vel = None
                self.step_no = 0
            else:
                self.last_ee_vel = ee_speed

        else:
            self.last_ee_vel = None

        # if eucl_dis <= self.p["setpoint"]["tol_lin_mag"] and \
        #         eucl_ang <= self.p["setpoint"]["tol_ang_mag"]:

        #     if self.sp_hold_time >= self.p["setpoint"]["hold_time"]:
        #         if self.p["setpoint"]["continious_mode"]:
        #             self.soft_reset = True
        #             self.sp_hold_time = 0.0
        #             self.step_no = 0
        #             self.integrated_hold_reward = 0.0
        #         if not self.recording:
        #             done = True
        #             info["done_reason"] = "success"
        #         else:
        #             self.success_counter += 1
        #             self.reset()
        #
        #         reward += self.p["reward"]["rew_goal_reached"]
        #
        #     self.sp_hold_time += self.tau
        #     dis_f = (1.0 - eucl_dis / self.p["setpoint"]["tol_lin_mag"])**2
        #     rew_hold = (self.tau * self.p["reward"]["fac_sp_hold"]
        #                 + self.tau
        #                 * self.p["reward"]["fac_sp_hold_near"] * dis_f)
        #     rew_hold = rew_hold / self.p["setpoint"]["hold_time"]
        #     self.integrated_hold_reward += rew_hold
        #     reward += rew_hold
        # else:
        #     reward -= self.integrated_hold_reward
        #     self.integrated_hold_reward = 0.0
        #     self.sp_hold_time = 0.0
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
        link_pose_r = self.robot.getLinkStates(self.link_mapping)
        j_pos, j_vel = self.robot.getJointStates(self.actuator_selection)

        # Observation: Base velocities
        mb_vel_w = self.robot.getBaseVelocity()

        # Add noise to base velocities
        std_lin = self.p["sensors"]["odometry"]["std_lin"]
        std_ang = self.p["sensors"]["odometry"]["std_ang"]
        mb_vel_w[0:2] *= self.np_random.normal(1, std_lin, size=2)
        mb_vel_w[2:3] *= self.np_random.normal(1, std_ang, size=1)

        # Observation: Lidar
        scan_ret = self.robot.getLidarScan()
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
        self.robot.resetFixedJoints()

        self.possible_sp_pos = self.randomize_environment()

        # Reset robot base
        self.robot.setBaseVelocity([0, 0, 0])

        # Reset setpoint
        sp_pos = self.reset_setpoint()

        # Reset robot arm
        collides = True
        n_tries = 0
        self.stepSimulation()
        err_c = 0
        while collides:
            n_tries += 1
            if n_tries >= 300:
                err_c += 1
                if err_c == 10:
                    print("Agent stuck...")
                self.randomize_environment(force_new_env=True)

                h_s_x = self.np_random.uniform(self.sp_init_pos[0]-7.5, self.sp_init_pos[0]+7.5)
                h_s_y = self.np_random.uniform(-0.5, self.corridor_width+0.5)
                h_e_x = self.np_random.uniform(self.sp_init_pos[0]-7.5, self.sp_init_pos[0]+7.5)
                h_e_y = self.np_random.uniform(-0.5, self.corridor_width+0.5)
                self.human.set_start_end([h_s_x, h_s_y], [h_e_x, h_e_y])


                n_tries = 0
            cl = self.corridor_length

            dis_sp = self.p['world']['spawn_range']

            j4 = self.np_random.uniform(self.joint_limits[3, 0],
                                       self.joint_limits[3, 1])
            j1 = self.np_random.uniform(self.joint_limits[0, 0],
                                       self.joint_limits[0, 1])

            rs = np.random.uniform(0, dis_sp)
            rsphi = np.random.uniform(np.pi/2.0, 3*np.pi/2.0)
            rx = rs*np.cos(rsphi)
            ry = rs*np.sin(rsphi)
            ths = np.random.uniform(-np.pi, np.pi)
            ths = 0

            c = lambda x: np.cos(x)
            s = lambda x: np.sin(x)
            T_se = np.array([[c(ths), -s(ths), rx],
                             [s(ths), c(ths), ry],
                             [0, 0, 1]])
            T_es = np.linalg.inv(T_se)

            T_2e = np.array([[1, 0, 0.594],
                             [0, 1, -0.083],
                             [0, 0, 1]])

            T_12 = np.array([[c(-j4), -s(-j4), 0.316],
                             [s(-j4), c(-j4), 0.082],
                             [0, 0, 1]])

            T_01 = np.array([[c(j1), -s(j1), 0.317],
                             [s(j1), c(j1), 0.0],
                             [0, 0, 1]])

            T_0s = T_01.dot(T_12.dot(T_2e.dot(T_es)))

            ang_base = -np.pi/2 - j1 + j4 - ths

            T_ws = np.array([[c(-np.pi/2), -s(-np.pi/2), sp_pos[0]],
                             [s(-np.pi/2), c(-np.pi/2), sp_pos[1]],
                             [0, 0, 1]])
            T_0si = T_0s.copy()
            T_0si[0:2, 0:2] = T_0s[0:2, 0:2].T
            T_0si[0:2, 2] = -T_0si[0:2, 0:2].dot(T_0s[0:2, 2])
            T_w0 = T_ws.dot(np.linalg.inv(T_0s))

            pose2d = [T_w0[0, 2], T_w0[1, 2], robot_init_yaw]
            self.robot.moveObject2D(pose2d, z=0.08)

            self.robot.setJointState(0, j1)
            self.robot.setJointState(3, j4)

            self.stepSimulation()

            collides = self.robot.isColliding()
            collides = (collides or robot_pos[0] < 0
                        or robot_pos[0] > self.corridor_length
                        or robot_pos[1] < 0
                        or robot_pos[1] > self.corridor_width)

        self.robot_init_pos = robot_pos
        self.sp_init_pos = sp_pos
        # Reset setpoint variables
        self.occmap.set_sp(sp_pos)
        self.generate_occmap_path()
        self.reset_setpoint_normalization()

        h_s_x = self.np_random.uniform(self.sp_init_pos[0]-7.5, self.sp_init_pos[0]+7.5)
        h_s_y = self.np_random.uniform(-0.5, self.corridor_width+0.5)
        h_e_x = self.np_random.uniform(self.sp_init_pos[0]-7.5, self.sp_init_pos[0]+7.5)
        h_e_y = self.np_random.uniform(-0.5, self.corridor_width+0.5)
        self.human.set_start_end([h_s_x, h_s_y], [h_e_x, h_e_y])
        # Calculate observation and return
        obs = self.calculate_observation()
        self.last_yaw = None

        if err_c >= 10:
            print("AGENT FREED! Whoow")
        return obs


    def stepSimulation(self):
        pb.stepSimulation(self.clientId)
        self.robot.valid_buffer_scan = False

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
            self.scl_eucl_dis = 1 / max(self.last_eucl_dis+1e-9, 3.0)
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
        self.robot = AslaugRobot(self.clientId)

        # Disable panda base collision
        pairs = [("camera_rack", x) for x in ["panda_hand", "panda_leftfinger", "panda_rightfinger", "panda_link5", "panda_link6", "panda_link7"]]
        self.robot.setupSelfCollisions(pairs)


    def configure_ext_collisions(self, bodyExt, body, enabled_links):
        for j in range(pb.getNumJoints(body, physicsClientId=self.clientId)):
            info_j = pb.getJointInfo(body, j, physicsClientId=self.clientId)
            link_name_j = info_j[12].decode("utf-8")
            idx_j = info_j[0]

            enabled = link_name_j in enabled_links

            pb.setCollisionFilterPair(body, bodyExt, idx_j, -1, enabled,
                                      self.clientId)
            for k in range(pb.getNumJoints(bodyExt, physicsClientId=self.clientId)):
                info_k = pb.getJointInfo(bodyExt, k, physicsClientId=self.clientId)
                idx_k = info_k[0]
                pb.setCollisionFilterPair(body, bodyExt, idx_j, idx_k, enabled,
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
            #print(idx, link_name)
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
        self.occmap = OccupancyMapGhost(0, corr_l, 0, corr_w, om_res)

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

        # print(sg.matrix1)
        # print(sg.matrix0)
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(sg.matrix0.astype(float))
        # plt.show()

        for id in ids:
            pb.setCollisionFilterPair(self.human.leg_l, id, -1, -1, False,
                                      self.clientId)
            pb.setCollisionFilterPair(self.human.leg_r, id, -1, -1, False,
                                      self.clientId)
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

        kallax1 = AslaugKallax([4.735, 0, 0], clientId=self.clientId)
        kallax2 = AslaugKallax([12.735, 0, 0], clientId=self.clientId)
        kallax1.setupCollisions(self.robot.id, self.collision_links)
        kallax2.setupCollisions(self.robot.id, self.collision_links)
        self.bookcases = [kallax1, kallax2]

    def move_object(self, id, pose2d):
        pos = [pose2d[0], pose2d[1], 0.0]
        ori = [0.0, 0.0] + [pose2d[2]]
        ori_quat = pb.getQuaternionFromEuler(ori)
        pb.resetBasePositionAndOrientation(id, pos, ori_quat,
                                           self.clientId)

        Rt = self.rotation_matrix(pose2d[2]).T
        return np.array(pos), Rt

    def randomize_environment(self, force_new_env=False):
        if force_new_env or \
                self.np_random.uniform() <= self.p["world"]["prob_new_env"]:
            for id in self.additionalIds:
                pb.removeBody(id, physicsClientId=self.clientId)
            self.additionalIds = self.spawn_additional_objects()

        # Randomize bookcases
        layers = self.p["setpoint"]["layers"]
        p_noise = self.p['setpoint']['noise']
        possible_sp_pos = []
        for bookcase in self.bookcases:
            poses = bookcase.sampleSetpointPoses(10, p_noise['range_x'],
                                                 p_noise['range_y'],
                                                 p_noise['range_z'],
                                                 layers=layers)
            possible_sp_pos.append(poses)

        return possible_sp_pos



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
