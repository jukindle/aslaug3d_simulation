import gym
from gym.utils import seeding
import numpy as np
import pybullet as pb
import pybullet_data
import os
import json

class AslaugBaseEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, version, params, gui=False, init_seed=None,
                 free_cam=False):
        self.free_cam = free_cam
        self.version = version
        self.gui = gui
        self.params = params
        self.viewer = None

        self.p = self.params
        self.tau = self.p["world"]["tau"]
        self.metadata["video.frames_per_second"] = int(round(1.0/self.tau))
        self.seed(init_seed)
        self.n_joints = len(self.p["joints"]["joint_names"])
        self.n_links = len(self.p["joints"]["link_names"])
        self.timeout_steps = (self.p["world"]["timeout"]
                              / self.p["world"]["tau"])
        self.step_no = 0

        # Set up simulation
        self.setup_simulation(gui=gui)

        self.setup_action_observation_spaces()

    def step(self, action_d):
        '''
        Executes one step.
        '''
        self.step_no += 1

        # Extract current state
        state_c = self.state
        mb_vel_c_r = state_c["base_vel"]
        joint_vel_c = state_c["joint_vel"]
        # Obtain actions
        mb_actions = np.choose(action_d[:3], self.actions[:, :3])
        joint_actions = np.zeros(7)
        act_joint_actions = np.choose(action_d[3:], self.actions[:, 3:])
        joint_actions[self.actuator_selection] = act_joint_actions

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

    def render(self, mode='human', w=1280, h=720):
        '''
        Renders the environment. Currently does nothing.
        '''
        if mode == 'rgb_array' or mode == 'human_fast' or not self.free_cam:
            camDistance = 4
            nearPlane = 0.01
            farPlane = 15
            fov = 60

            cam_pos, rpy = self.get_camera_pose()

            viewMatrix = pb.computeViewMatrixFromYawPitchRoll(cam_pos,
                                                              camDistance,
                                                              rpy[2], rpy[1],
                                                              rpy[0], 2,
                                                              self.clientId)
        if not self.free_cam:
            pb.resetDebugVisualizerCamera(camDistance, rpy[2], rpy[1], cam_pos,
                                          self.clientId)
        if mode == 'rgb_array' or mode == 'human_fast':
            aspect = w / h
            projectionMatrix = pb.computeProjectionMatrixFOV(fov, aspect,
                                                             nearPlane,
                                                             farPlane,
                                                             self.clientId)
            img_arr = pb.getCameraImage(w,
                                        h,
                                        viewMatrix,
                                        projectionMatrix,
                                        shadow=1,
                                        lightDirection=[0.5, 0.3, 1],
                                        renderer=pb.ER_BULLET_HARDWARE_OPENGL,
                                        physicsClientId=self.clientId)

            img = np.array(img_arr[2])[:, :, 0:3]
        if mode == 'rgb_array':
            return img
        if mode == 'human_fast':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return img
        # elif mode == 'human':
        #     assert self.gui, "Must use GUI for render mode human!"

    def setup_simulation(self, gui=False):
        '''
        Initializes the simulation by setting up the environment and spawning
        all objects used later.

        Params:
            gui (bool): Specifies if a GUI should be spawned.
        '''
        # Setup simulation parameters
        mode = pb.GUI if gui else pb.DIRECT
        self.clientId = pb.connect(mode)
        pb.setGravity(0.0, 0.0, 0.0, self.clientId)
        pb.setPhysicsEngineParameter(fixedTimeStep=self.p["world"]["tau"])
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        dirname = os.path.dirname(__file__)
        model_path = os.path.join(dirname, '../urdf/floor/plane.urdf')
        pb.loadURDF(model_path, useFixedBase=True,
                    physicsClientId=self.clientId)
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

        # Spawn robot
        self.robotId = self.spawn_robot()

        # Spawn setpoint
        self.spId = self.spawn_setpoint()

        # Spawn all objects in the environment
        self.spawn_additional_objects()

        # Spawn bookcases
        self.bookcaseIds = self.spawn_bookcases(self.p["world"]["n_bookcases"])

        # Figure out joint mapping: self.joint_mapping maps as in
        # desired_mapping list.
        self.joint_mapping = np.zeros(7, dtype=int)
        self.link_mapping = np.zeros(self.n_links, dtype=int)
        self.joint_limits = np.zeros((7, 2), dtype=float)
        self.eeLinkId = None
        self.baseLinkId = None
        self.lidarLinkId1 = None
        self.lidarLinkId2 = None

        joint_names = ["panda_joint{}".format(x) for x in range(1, 8)]
        link_names = self.p["joints"]["link_names"]

        for j in range(pb.getNumJoints(self.robotId,
                                       physicsClientId=self.clientId)):
            info = pb.getJointInfo(self.robotId, j,
                                   physicsClientId=self.clientId)
            j_name, l_name = info[1].decode("utf-8"), info[12].decode("utf-8")
            idx = info[0]
            if j_name in joint_names:
                map_idx = joint_names.index(j_name)
                self.joint_mapping[map_idx] = idx
                self.joint_limits[map_idx, :] = info[8:10]
            if l_name in link_names:
                self.link_mapping[link_names.index(l_name)] = idx
            if l_name == self.p["joints"]["ee_link_name"]:
                self.eeLinkId = idx
            if l_name == self.p["joints"]["base_link_name"]:
                self.baseLinkId = idx
            if l_name == self.p["sensors"]["lidar"]["link_id1"]:
                self.lidarLinkId1 = idx
            if l_name == self.p["sensors"]["lidar"]["link_id2"]:
                self.lidarLinkId2 = idx

        for j in range(pb.getNumJoints(self.spId,
                                       physicsClientId=self.clientId)):
            info = pb.getJointInfo(self.spId, j,
                                   physicsClientId=self.clientId)
            link_name = info[12].decode("utf-8")
            idx = info[0]
            if link_name == "grasp_loc":
                self.spGraspLinkId = idx

        self.actuator_selection = np.zeros(7, bool)
        for i, name in enumerate(joint_names):
            if name in self.p["joints"]["joint_names"]:
                self.actuator_selection[i] = 1

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

    def seed(self, seed=None):
        '''
        Initializes numpy's random package with a given seed.

        Params:
            seed (int): Seed to use. None means a random seed.
        Returns:
            list: The seed packed in a list.
        '''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def rotation_matrix(self, ang):
        '''
        Calculates a rotation matrix around z-axis.

        Params:
            ang (float): Angle to rotate.
        Returns:
            numpy.array: 3x3 rotation matrix
        '''
        return np.array([[np.cos(ang),  -np.sin(ang),   0],
                         [np.sin(ang),  +np.cos(ang),   0],
                         [0,            0,              1]])

    def get_ee_sp_transform(self):
        '''
        Calculates pose of setpoint w.r.t. end effector frame.

        Returns:
            numpy.array: 6D pose of setpoint in end effector frame.
        '''
        state_ee = pb.getLinkState(self.robotId, self.eeLinkId,
                                   False, False, self.clientId)
        ee_pos_w, ee_ori_w = state_ee[4:6]
        w_pos_ee, w_ori_ee = pb.invertTransform(ee_pos_w, ee_ori_w,
                                                self.clientId)

        state_sp = pb.getLinkState(self.spId, self.spGraspLinkId,
                                   False, False, self.clientId)
        sp_pos_w, sp_ori_w = state_sp[4:6]

        sp_pos_ee, sp_ori_ee = pb.multiplyTransforms(w_pos_ee, w_ori_ee,
                                                     sp_pos_w, sp_ori_w,
                                                     self.clientId)

        sp_eul_ee = pb.getEulerFromQuaternion(sp_ori_ee, self.clientId)

        return np.array(sp_pos_ee + sp_eul_ee)

    def get_link_states(self, links_idx):
        '''
        Obtain matrix with 6D poses of links specified.

        Args:
            links_idx (list): Indices of links from link_names list in params.
        Returns:
            numpy.array: 3D poses for all link indices.
                Shape [len(links_idx, 6)] where second dim. (x,y,z,r,p,y)
        '''
        # NOTE: Using euler angles, might this be a problem?
        link_poses = np.zeros((len(links_idx), 6))
        states = pb.getLinkStates(self.robotId, links_idx, False, False,
                                  self.clientId)
        states_mb = pb.getLinkState(self.robotId, self.baseLinkId, False,
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

    def get_joint_states(self, sel=None):
        '''
        Obtain joint positions and velocities.

        Returns:
            numpy.array: Joint positions in radians.
            numpy.array: Joint velocities in radians/s.
        '''
        states = pb.getJointStates(self.robotId, self.joint_mapping,
                                   self.clientId)
        j_pos = [x[0] for x in states]
        j_vel = [x[1] for x in states]

        if sel is None:
            return np.array(j_pos), np.array(j_vel)
        else:
            return np.array(j_pos)[sel], np.array(j_vel)[sel]

    def get_base_vels(self):
        '''
        Obtain base velocities.

        Returns:
            numpy.array: Velocities of movebase (x, y, theta).
        '''
        state = pb.getLinkState(self.robotId, self.baseLinkId, True,
                                False, self.clientId)

        mb_ang_w = pb.getEulerFromQuaternion(state[5])[2]
        v_lin, v_ang = state[6:8]
        mb_vel_w = np.array(v_lin[0:2] + v_ang[2:3])
        return self.rotation_matrix(mb_ang_w).T.dot(mb_vel_w)

    def get_lidar_scan(self):
        '''
        Obtain lidar scan values for current state.

        Returns:
            list: Scan values for range and resolution specified in
                params dict.
        '''
        scan_front = None
        scan_rear = None

        if self.lidarLinkId1 is not None:
            # Get pose of lidar
            states = pb.getLinkState(self.robotId, self.lidarLinkId1,
                                     False, False, self.clientId)
            lidar_pos, lidar_ori = states[4:6]
            lidar_pos = np.array(lidar_pos)
            R = np.array(pb.getMatrixFromQuaternion(lidar_ori))
            R = np.reshape(R, (3, 3))
            scan_l = R.dot(self.rays[0]).T + lidar_pos
            scan_h = R.dot(self.rays[1]).T + lidar_pos
            scan_r = pb.rayTestBatch(scan_l.tolist(), scan_h.tolist(),
                                     self.clientId)

            scan = [x[2]*self.p["sensors"]["lidar"]["range"] for x in scan_r]
            scan_front = scan

        if self.lidarLinkId2 is not None:
            # Get pose of lidar
            states = pb.getLinkState(self.robotId, self.lidarLinkId2,
                                     False, False, self.clientId)
            lidar_pos, lidar_ori = states[4:6]
            lidar_pos = np.array(lidar_pos)
            R = np.array(pb.getMatrixFromQuaternion(lidar_ori))
            R = np.reshape(R, (3, 3))
            scan_l = R.dot(self.rays[0]).T + lidar_pos
            scan_h = R.dot(self.rays[1]).T + lidar_pos
            scan_r = pb.rayTestBatch(scan_l.tolist(), scan_h.tolist(),
                                     self.clientId)

            scan = [x[2]*self.p["sensors"]["lidar"]["range"] for x in scan_r]
            scan_rear = scan
        return [scan_front, scan_rear]

    def set_velocities(self, mb_vel_r, joint_vel):
        '''
        Applies velocities of move base and joints to the simulation.

        Args:
            mb_vel_r (numpy.array): Base velocities to apply (x, y, theta).
            joint_vel (numpy.array): Joint velocities to apply. Length: 7.
        '''
        # Obtain robot orientation and transform mb vels to world frame
        mb_link_state = pb.getLinkState(self.robotId, self.baseLinkId,
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
        '''
        Checks if any joint limit is reached.

        Returns:
            bool: Any joint has reached its limit.
        '''
        j_pos, _ = self.get_joint_states()
        max_reached = ((self.joint_limits[:, 1] - j_pos) <= 1e-3).any()
        min_reached = ((j_pos - self.joint_limits[:, 0]) <= 1e-3).any()

        return min_reached or max_reached

    def check_collision(self):
        '''
        Checks if robot collides with any body.

        Returns:
            bool: Whether robot is in collision or not.
        '''
        return len(pb.getContactPoints(bodyA=self.robotId,
                                       physicsClientId=self.clientId)) > 0

    def spawn_bookcases(self, n):
        '''
        Prepares the simulation by spawning n bookcases.

        Args:
            n (int): Number of bookcases.
        Returns:
            list: List of bookcase IDs.
        '''
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
        pos = [pose2d[0], pose2d[1], 0.0]
        ori = [0.0, 0.0] + [pose2d[2]]
        ori_quat = pb.getQuaternionFromEuler(ori)
        pb.resetBasePositionAndOrientation(bookcaseId, pos, ori_quat,
                                           self.clientId)

        # Calculate possible setpoint positions
        sp_pos = []
        Rt = self.rotation_matrix(pose2d[2]).T
        pos = np.array(pos)
        for l in sp_layers:
            z = 0.05 + 0.35*l
            sp_pos.append(pos + Rt.dot(np.array([-0.15, 0.18, z])))
            sp_pos.append(pos + Rt.dot(np.array([-0.15, -0.18, z])))

        return sp_pos

    def close(self):
        '''
        Closes the environment.
        '''
        try:
            pb.disconnect(self.clientId)
        except:
            pass

    def move_sp(self, pos):
        '''
        Function which moves setpoint to a new location,
        together with its marker

        Args:
            pos (numpy.array): 3D position where to move setpoint to
        '''
        pos_sp = list(pos)
        pos_mk = list(pos)
        pos_mk[2] = 1.6
        ori_quat = pb.getQuaternionFromEuler([0, 0, 0])
        pb.resetBasePositionAndOrientation(self.spId, pos_sp, ori_quat,
                                           self.clientId)
        pb.resetBasePositionAndOrientation(self.markerId, pos_mk, ori_quat,
                                           self.clientId)

    def get_camera_pose(self):
        state_mb = pb.getLinkState(self.robotId, self.baseLinkId,
                                   False, False, self.clientId)
        mb_pos_w, mb_ori_w = state_mb[4:6]
        state_sp = pb.getLinkState(self.spId, self.spGraspLinkId,
                                   False, False, self.clientId)
        sp_pos_w, sp_ori_w = state_sp[4:6]

        spmb_uvec = (np.array(sp_pos_w) - np.array(mb_pos_w))
        spmb_uvec = spmb_uvec / np.linalg.norm(spmb_uvec)
        cam_pos = np.array(mb_pos_w)

        yaw = (np.arctan2(spmb_uvec[1], spmb_uvec[0]) / (2.0 * np.pi) * 360.0
               - 90.0)
        if self.last_yaw is None:
            self.last_yaw = yaw

        dy_mag = 55.0
        yaw = self.last_yaw + 0.05*max(-dy_mag, min(dy_mag, yaw-self.last_yaw))
        self.last_yaw = yaw
        rpy = [0, -45, yaw]
        return cam_pos, rpy

    def calculate_success_rate(self):
        if self.episode_counter <= 1:
            return 0
        else:
            return self.success_counter / (self.episode_counter - 1)

    def save_world(self, dir, pre_f, inf_f, ep):
        wn = "{}.video.{}.video{:06}.world.world".format(pre_f, inf_f, ep)
        sn = "{}.video.{}.video{:06}.setpoints.json".format(pre_f, inf_f, ep)
        world_path = os.path.join(dir, wn)
        sp_path = os.path.join(dir, sn)
        pb.saveWorld(world_path, self.clientId)
        with open(sp_path, 'w') as f:
            json.dump(self.sp_history, f)
