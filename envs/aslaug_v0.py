import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet as pb
import pybullet_data
import os
import random


class AslaugEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    # Overvation: [sp_pos_ee[3], mb_vel_r[3], link_pos[3*n_joints],
    #              joint_pos[3], joint_vel[3]]
    # Action: [mb_d_vel_r[3], joint_d_vel[3]]
    # State: [mb_pos_w[3], mb_vel_w[3], joint_pos[3], joint_vel[3]]
    def __init__(self, folder_name="", gui=False):
        # Common params
        self.version = "v0"
        self.folder_name = folder_name
        self.gui = gui

        self.params = {
            "joints": {
                "joint_names": ['panda_joint{}'.format(i+1) for i in range(7)],
                "init_states": [0.0, -0.75, 0.0, -1.75, 0.0, 2.5, np.pi/4],
                "link_names": (['top_link', 'front_laser']
                               + ['panda_link{}'.format(i+1) for i in range(7)]
                               + ['panda_hand']),
                "vel_mag": 1.5,
                "acc_mag": 1.5
            },
            "base": {
                "vel_mag": np.array([0.6, 0.6, 1.2]),
                "acc_mag": np.array([3.0, 3.0, 1.0])
            },
            "setpoint": {
                "hold_time": 2.0,
                "tol_lin_mag": 0.15,
                "tol_ang_mag": np.pi/4
            },
            "reward": {
                "fac_goal_dis_lin": 10.0,
                "fac_goal_dis_ang": 10.0,
                "fac_ang_vel": -2.0,
                "fac_sp_hold": 3.0,
                "rew_timeout": -5.0,
                "rew_joint_limits": -5.0,
                "rew_collision": -15.0,
                "rew_goal_reached": 25.0
            },
            "world": {
                "tau": 0.02,
                "timeout": 30.0,
                "size": 20.0,
                "action_discretization": 7,
                "n_bookcases": 12,
                "corridor_width": 3.0,
            },
            "sensors": {
                "lidar": {
                    "n_scans": 41,
                    "ang_mag": np.pi/2,
                    "range": 5.0
                }
            }
        }

        # # # Internal params # # #
        self.p = self.params
        self.tau = self.p["world"]["tau"]
        self.metadata["video.frames_per_second"] = int(round(1.0/self.tau))
        self.seed()
        self.n_joints = len(self.p["joints"]["joint_names"])
        self.n_links = len(self.p["joints"]["link_names"])
        self.timeout_steps = (self.p["world"]["timeout"]
                              / self.p["world"]["tau"])
        self.step_no = 0

        # Set up simulation
        self.setup_simulation(gui=gui)

        # Define action space
        # [delta_mb_vel_r[3], delta_joint_vel[n_joints]]
        accel_lims_mb = self.p["base"]["acc_mag"]
        acc_lim_joints = (self.n_joints*[self.p["joints"]["acc_mag"]])
        highs_a = (self.p["world"]["tau"]
                   * np.concatenate((accel_lims_mb, acc_lim_joints)))
        lows_a = -highs_a
        n_d = self.p["world"]["action_discretization"]
        self.action_space = spaces.MultiDiscrete(lows_a.shape[0]*[n_d])
        self.actions = np.linspace(lows_a, highs_a, n_d)

        # Define observation space
        # Overvation: [sp_pos_ee[6], mb_vel_r[3], link_pos[6*n_links+1],
        #              joint_pos[n_joints], joint_vel[n_joints],
        #              scan[n_scans]]
        high_sp = np.array([self.p["world"]["size"]]*2 + [1.5] + 3*[np.pi])
        low_sp = -high_sp
        high_mb = np.array(self.p["base"]["vel_mag"])
        low_mb = -high_mb
        # TODO: add proper scaling
        high_lp = np.array((self.n_joints+1)*[1.5, 1.5, 1.5, np.pi, np.pi, np.pi])
        low_lp = -high_lp
        high_j_p = self.joint_limits[:, 1]
        low_j_p = self.joint_limits[:, 0]
        high_j_v = np.array([self.p["joints"]["vel_mag"]]*self.n_joints)
        low_j_v = -high_j_v
        rng = self.p["sensors"]["lidar"]["range"]
        high_scan = rng * np.ones(self.p["sensors"]["lidar"]["n_scans"])
        low_scan = 0.1*high_scan
        high_o = np.concatenate((high_sp, high_mb, high_lp, high_j_p,
                                 high_j_v, high_scan))
        low_o = np.concatenate((low_sp, low_mb, low_lp, low_j_p,
                                low_j_v, low_scan))

        self.observation_space = spaces.Box(low_o, high_o)

    def step(self, action_d):
        self.step_no += 1

        # Extract current state
        state_c = self.state
        mb_vel_c_r = state_c["base_vel"]
        joint_vel_c = state_c["joint_vel"]

        # Obtain actions
        action = np.choose(action_d, self.actions)
        mb_actions = action[:3]
        joint_actions = action[3:]

        # Calculate new velocities and clip limits
        mb_vel_n_r = np.clip(mb_vel_c_r + mb_actions,
                             -self.p["base"]["vel_mag"],
                             +self.p["base"]["vel_mag"])
        joint_vel_n = np.clip(joint_vel_c + joint_actions,
                              -self.p["joints"]["vel_mag"],
                              +self.p["joints"]["vel_mag"])

        # Apply new velocity commands to robot
        self.set_velocities(mb_vel_n_r, joint_vel_n)

        # Execute one step in simulation
        pb.stepSimulation(self.clientId)

        # Update internal state
        self.state = {"base_vel": mb_vel_n_r, "joint_vel": joint_vel_n}

        # Calculate reward
        reward, done, info = self.calculate_reward()

        # Obtain observation
        obs = self.calculate_observation()

        return obs, reward, done, info

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

        # Penalize velocity in move base rotation
        mb_ang_vel = self.get_base_vels()[2]
        reward += np.abs(mb_ang_vel)*self.tau*self.p["reward"]["fac_ang_vel"]

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
                done = True
                info["done_reason"] = "success"
                reward += self.p["reward"]["rew_goal_reached"]

            self.sp_hold_time += self.tau
            reward += self.tau*self.p["reward"]["fac_sp_hold"]
        else:
            reward -= self.sp_hold_time*self.p["reward"]["fac_sp_hold"]
            self.sp_hold_time = 0.0

        return reward, done, info

    def calculate_observation(self):
        sp_pose_ee = self.get_ee_sp_transform()
        link_pose_r = self.get_link_states(self.link_mapping[2:])
        j_pos, j_vel = self.get_joint_states()
        mb_vel_w = self.get_base_vels()
        scan = self.get_lidar_scan()
        obs = np.concatenate((sp_pose_ee, mb_vel_w, link_pose_r.flatten(),
                              j_pos, j_vel, scan))
        return obs

    def reset(self, init_state=None, init_setpoint_state=None,
              init_obstacle_grid=None, init_obstacle_locations=None):

        # Reset internal parameters
        self.step_no = 0
        self.state = {"base_vel": np.array([0.0, 0.0, 0.0]),
                      "joint_vel": np.array(7*[0.0])}

        # Reset environment
        for i in range(len(self.joint_mapping)):
            pb.resetJointState(self.robotId, self.joint_mapping[i],
                               self.p["joints"]["init_states"][i],
                               0.0, self.clientId)

        pb.resetBaseVelocity(self.robotId, [0, 0, 0], [0, 0, 0], self.clientId)

        robot_pos = (0, 0, 0.02)
        robot_ori = pb.getQuaternionFromEuler([np.pi/2, 0, np.pi/2])
        pb.resetBasePositionAndOrientation(self.robotId, robot_pos, robot_ori,
                                           self.clientId)

        # Reorder bookcases
        possible_sp_pos = []
        pos = np.array([1.0, -self.p["world"]["corridor_width"]/2, np.pi/2])
        for i in range(int(len(self.bookcaseIds)/2.0)):
            bookcaseId = self.bookcaseIds[i]
            possible_sp_pos += self.move_bookcase(bookcaseId, pos)
            pos[0] += 1.1 + self.np_random.uniform(0, 0.2)

        pos = np.array([1.0, self.p["world"]["corridor_width"]/2, np.pi/2])
        for i in range(int(len(self.bookcaseIds)/2.0), len(self.bookcaseIds)):
            bookcaseId = self.bookcaseIds[i]
            possible_sp_pos += self.move_bookcase(bookcaseId, pos)
            pos[0] += 1.2 + self.np_random.uniform(0, 0.2)

        # Spawn random setpoint
        sp_pos = random.sample(possible_sp_pos, 1)[0]
        self.move_sp(sp_pos)

        # Initialize reward state variables
        self.last_eucl_dis, self.last_eucl_ang = self.calculate_goal_distance()
        self.scl_eucl_dis = 1/self.last_eucl_dis
        self.scl_eucl_ang = 1/self.last_eucl_ang
        self.sp_hold_time = 0.0

        # Calculate observation and return
        obs = self.calculate_observation()
        return obs

    def render(self, mode='human', w=1600, h=1600):
        assert self.gui, "GUI mode needs to be enabled for rendering!"
        # if mode == "human":
        #     self.isRender = True
        # if mode != "rgb_array":
        #     return np.array([])

    def setup_simulation(self, gui=False):
        # Setup simulation parameters
        mode = pb.GUI if gui else pb.DIRECT
        self.clientId = pb.connect(mode)
        pb.setGravity(0.0, 0.0, 9.81, self.clientId)
        pb.setPhysicsEngineParameter(fixedTimeStep=self.p["world"]["tau"])
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.loadURDF("plane.urdf", physicsClientId=self.clientId)

        # Spawn robot
        robot_pos = [0, 0, 0.02]
        robot_ori = pb.getQuaternionFromEuler([0, 0, 0])
        dirname = os.path.dirname(__file__)
        model_path = os.path.join(dirname, '../urdf/robot/aslaug.urdf')
        self.robotId = pb.loadURDF(model_path, robot_pos, robot_ori,
                                   useFixedBase=True,
                                   physicsClientId=self.clientId)

        # Spawn mug
        mug_pos = [5, 2, 0.0]
        mug_ori = pb.getQuaternionFromEuler([0, 0, 0])
        dirname = os.path.dirname(__file__)
        model_path = os.path.join(dirname, '../urdf/beer_rothaus/beer_rothaus.urdf')
        self.spId = pb.loadURDF(model_path, mug_pos, mug_ori,
                                useFixedBase=True,
                                physicsClientId=self.clientId)

        # Figure out joint mapping: self.joint_mapping maps as in
        # desired_mapping list.
        self.joint_mapping = np.zeros(self.n_joints, dtype=int)
        self.link_mapping = np.zeros(self.n_links, dtype=int)
        self.joint_limits = np.zeros((self.n_joints, 2), dtype=float)
        joint_names = self.p["joints"]["joint_names"]
        link_names = self.p["joints"]["link_names"]
        for j in range(pb.getNumJoints(self.robotId,
                                       physicsClientId=self.clientId)):
            info = pb.getJointInfo(self.robotId, j,
                                   physicsClientId=self.clientId)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            idx = info[0]
            if joint_name in joint_names:
                map_idx = joint_names.index(joint_name)
                self.joint_mapping[map_idx] = idx
                self.joint_limits[map_idx, :] = info[8:10]
            if link_name in link_names:
                self.link_mapping[link_names.index(link_name)] = idx

        # Prepare lidar
        n_scans = self.p["sensors"]["lidar"]["n_scans"]
        mag_ang = self.p["sensors"]["lidar"]["ang_mag"]
        scan_range = self.p["sensors"]["lidar"]["range"]
        angs = ((np.array(range(n_scans))
                 - (n_scans-1)/2.0)*2.0/n_scans*mag_ang)
        r_uv = np.vstack((np.cos(angs), np.sin(angs),
                          np.zeros(angs.shape[0])))
        r_from = r_uv * 0.1
        r_to = r_uv * scan_range

        self.rays = (r_from, r_to)
        # Spawn bookcases
        self.bookcaseIds = self.spawn_bookcases(self.p["world"]["n_bookcases"])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def rotation_matrix(self, ang):
        return np.array([[np.cos(ang),  -np.sin(ang),   0],
                         [np.sin(ang),  +np.cos(ang),   0],
                         [0,            0,              1]])

    def get_ee_sp_transform(self):
        state_ee = pb.getLinkState(self.robotId, self.link_mapping[-1],
                                   False, False, self.clientId)
        ee_pos_w, ee_ori_w = state_ee[4:6]
        w_pos_ee, w_ori_ee = pb.invertTransform(ee_pos_w, ee_ori_w,
                                                self.clientId)
        sp_pos_w, sp_ori_w = pb.getBasePositionAndOrientation(self.spId,
                                                              self.clientId)

        sp_pos_ee, sp_ori_ee = pb.multiplyTransforms(w_pos_ee, w_ori_ee,
                                                     sp_pos_w, sp_ori_w,
                                                     self.clientId)

        sp_eul_ee = pb.getEulerFromQuaternion(sp_ori_ee, self.clientId)

        return np.array(sp_pos_ee + sp_eul_ee)

    def get_link_states(self, links_idx):
        # NOTE: Using euler angles, might this be a problem?
        link_poses = np.zeros((len(links_idx), 6))
        states = pb.getLinkStates(self.robotId, links_idx, False, False,
                                  self.clientId)
        states_mb = pb.getLinkState(self.robotId, self.link_mapping[0], False,
                                    False, self.clientId)
        mb_pos_w, mb_ori_w = states_mb[4:6]
        w_pos_mb, w_ori_mb = pb.invertTransform(mb_pos_w, mb_ori_w,
                                                self.clientId)
        for i, state in enumerate(states):
            link_pos_w, link_ori_w = state[4:6]
            link_pos_r, link_ori_r = pb.multiplyTransforms(w_pos_mb, w_ori_mb,
                                                           link_pos_w,
                                                           link_ori_w,
                                                           self.clientId)
            link_eul_r = pb.getEulerFromQuaternion(link_ori_r, self.clientId)
            link_poses[i, :] = link_pos_r + link_eul_r

        return link_poses

    def get_joint_states(self):
        states = pb.getJointStates(self.robotId, self.joint_mapping,
                                   self.clientId)
        j_pos = [x[0] for x in states]
        j_vel = [x[1] for x in states]

        return np.array(j_pos), np.array(j_vel)

    def get_base_vels(self):
        state = pb.getLinkState(self.robotId, self.link_mapping[0], True,
                                False, self.clientId)
        v_lin, v_ang = state[6:8]

        return np.array(v_lin[0:2] + v_ang[2:3])

    def get_lidar_scan(self):
        # Get pose of lidar
        states = pb.getLinkState(self.robotId, self.link_mapping[1],
                                  False, False, self.clientId)
        lidar_pos, lidar_ori = states[4:6]
        lidar_pos = np.array(lidar_pos)
        R = np.array(pb.getMatrixFromQuaternion(lidar_ori))
        R = np.reshape(R, (3, 3))
        scan_l = R.dot(self.rays[0]).T + lidar_pos
        scan_h = R.dot(self.rays[1]).T + lidar_pos
        scan_r = pb.rayTestBatch(scan_l.tolist(), scan_h.tolist(), self.clientId)

        scan = [x[2]*self.p["sensors"]["lidar"]["range"] for x in scan_r]
        return scan

    def set_velocities(self, mb_vel_r, joint_vel):
        # Obtain robot orientation and transform mb vels to world frame
        mb_link_state = pb.getLinkState(self.robotId, self.link_mapping[0],
                                        False, False, self.clientId)
        mb_ang_w = pb.getEulerFromQuaternion(mb_link_state[5])[2]

        mb_vel_w = self.rotation_matrix(mb_ang_w).dot(mb_vel_r)

        # Apply velocities to simulation
        vel_lin = np.append(mb_vel_w[0:2], 0.0)
        vel_ang = np.append([0.0, 0.0], mb_vel_w[2])
        pb.resetBaseVelocity(self.robotId, vel_lin, vel_ang, self.clientId)

        pb.setJointMotorControlArray(self.robotId, self.joint_mapping,
                                     pb.VELOCITY_CONTROL,
                                     targetVelocities=joint_vel,
                                     forces=len(joint_vel)*[1e10],
                                     physicsClientId=self.clientId)

    def check_joint_limits_reached(self):
        j_pos, _ = self.get_joint_states()
        max_reached = ((self.joint_limits[:, 1] - j_pos) <= 1e-3).any()
        min_reached = ((j_pos - self.joint_limits[:, 0]) <= 1e-3).any()

        return min_reached or max_reached

    def check_collision(self):
        return len(pb.getContactPoints(self.robotId, self.clientId)) > 0

    def calculate_goal_distance(self):
        sp_pose_ee = self.get_ee_sp_transform()
        eucl_dis = np.linalg.norm(sp_pose_ee[0:3])
        eucl_ang = np.linalg.norm(sp_pose_ee[3:6])
        return eucl_dis, eucl_ang

    def spawn_bookcases(self, n):
        pose2d = [5.0, 0.0, 0.0]
        dirname = os.path.dirname(__file__)
        model_path = os.path.join(dirname, '../urdf/bookcase/bookcase.urdf')
        pos = pose2d[0:2] + [0.0]
        ori = [0.0, 0.0] + [pose2d[2]]
        ori_quat = pb.getQuaternionFromEuler(ori)

        ids = []
        for i in range(n):
            bookcaseId = pb.loadURDF(model_path, pos, ori_quat,
                                     useFixedBase=True,
                                     physicsClientId=self.clientId)
            ids.append(bookcaseId)
        return ids

    def move_bookcase(self, bookcaseId, pose2d):
        pos = [pose2d[0], pose2d[1], 0.0]
        ori = [0.0, 0.0] + [pose2d[2]]
        ori_quat = pb.getQuaternionFromEuler(ori)
        pb.resetBasePositionAndOrientation(bookcaseId, pos, ori_quat,
                                           self.clientId)

        # Calculate possible setpoint positions
        sp_pos = []
        Rt = self.rotation_matrix(pose2d[2]).T
        sp_pos.append(np.array(pos) + Rt.dot(np.array([0.0, 0.18, 1.1])))
        sp_pos.append(np.array(pos) + Rt.dot(np.array([0.0, -0.18, 1.1])))
        sp_pos.append(np.array(pos) + Rt.dot(np.array([0.0, 0.18, 0.75])))
        sp_pos.append(np.array(pos) + Rt.dot(np.array([0.0, -0.18, 0.75])))
        sp_pos.append(np.array(pos) + Rt.dot(np.array([0.0, 0.18, 0.4])))
        sp_pos.append(np.array(pos) + Rt.dot(np.array([0.0, -0.18, 0.4])))
        sp_pos.append(np.array(pos) + Rt.dot(np.array([0.0, 0.18, 0.05])))
        sp_pos.append(np.array(pos) + Rt.dot(np.array([0.0, -0.18, 0.05])))

        return sp_pos

    def close(self):
        pb.disconnect(self.clientId)

    def move_sp(self, pos):
        ori_quat = pb.getQuaternionFromEuler([0, 0, 0])
        pb.resetBasePositionAndOrientation(self.spId, pos, ori_quat,
                                           self.clientId)
