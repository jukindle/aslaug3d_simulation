import gym
from gym.utils import seeding
import numpy as np
import pybullet as pb
import pybullet_data
import os
import yaml


class AslaugBaseEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, version, params, gui=False, init_seed=None,
                 free_cam=False, easy_bookcases=False):
        self.free_cam = free_cam
        self.version = version
        self.gui = gui
        self.done_info = None

        if params is None:
            print("No env params specified, using default.")
            with open("params.yaml") as f:
                params_all = yaml.load(f)
            params = params_all["environment_params"]
        params = self.numpyfy_dict(params)
        self.p = params
        self.viewer = None

        self.fixed_joint_states = self.p["joints"]["init_states"]
        self.tau = self.p["world"]["tau"]
        self.metadata["video.frames_per_second"] = int(round(1.0/self.tau))
        self.seed(init_seed)
        self.n_joints = len(self.p["joints"]["joint_names"])
        self.n_links = len(self.p["joints"]["link_names"])
        self.timeout_steps = (self.p["world"]["timeout"]
                              / self.p["world"]["tau"])
        self.step_no = 0
        self.valid_buffer_scan = False

        # Set up simulation
        self.setup_simulation(gui=gui, easy_bookcases=easy_bookcases)

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
        self.action_d = action_d

        joint_actions = np.zeros(7)
        if self.p["world"]["action_discretization"] > 0:
            mb_actions = np.choose(action_d[:3], self.actions[:, :3])
            act_joint_actions = np.choose(action_d[3:], self.actions[:, 3:])
            joint_actions[self.actuator_selection] = act_joint_actions
        else:
            mb_actions = action_d[:3]
            lim_up = self.n_joints + 3
            joint_actions[self.actuator_selection] = action_d[3:lim_up]
            if self.p["world"]["use_stop_action"]:
                stop_base = action_d[lim_up] > 0.5
                stop_arm = action_d[lim_up+1] > 0.5
                if stop_base:
                    mb_actions[:3] = np.clip(-mb_vel_c_r,
                                             self.action_space.low[:3],
                                             self.action_space.high[:3])
                if stop_arm:
                    joint_actions[self.actuator_selection] = (
                            np.clip(-joint_vel_c[self.actuator_selection],
                                    self.action_space.low[3:lim_up],
                                    self.action_space.high[3:lim_up])
                            )

        # Add noise to base accelerations
        std_lin = self.p["base"]["std_acc_lin"]
        std_ang = self.p["base"]["std_acc_ang"]
        mb_noise_fac_lin = self.np_random.normal(1, std_lin, 2)
        mb_noise_fac_ang = self.np_random.normal(1, std_ang, 1)
        mb_actions[0:2] *= mb_noise_fac_lin
        mb_actions[2:3] *= mb_noise_fac_ang

        # Add noise to joint accelerations
        j_std = self.p["joints"]["std_acc"]
        joint_noise_fac = self.np_random.normal(1, j_std, joint_actions.shape)
        joint_actions *= joint_noise_fac
        # Calculate new velocities and clip limits
        mb_vel_n_r = mb_vel_c_r + mb_actions
        mb_vel_abs_lin = np.linalg.norm(mb_vel_n_r[0:2])
        mb_vel_abs_ang = np.linalg.norm(mb_vel_n_r[2])
        if mb_vel_abs_lin > 0.0:
            cut_vel = min(mb_vel_abs_lin, self.p['base']['vel_mag_lin'])
            mb_vel_n_r[0:2] = mb_vel_n_r[0:2] / mb_vel_abs_lin * cut_vel
        if mb_vel_abs_ang > 0.0:
            cut_vel = min(mb_vel_abs_ang, self.p['base']['vel_mag_ang'])
            mb_vel_n_r[2] = mb_vel_n_r[2] / mb_vel_abs_ang * cut_vel

        joint_vel_n = np.clip(joint_vel_c + joint_actions,
                              -self.p["joints"]["vel_mag"],
                              +self.p["joints"]["vel_mag"])

        # Apply new velocity commands to robot
        self.set_velocities(mb_vel_n_r, joint_vel_n)

        # Ensure that fixed joints do not move at all
        for i in range(len(self.actuator_selection)):
            if not self.actuator_selection[i]:
                pb.resetJointState(self.robotId, self.joint_mapping[i],
                                   self.fixed_joint_states[i], 0.0,
                                   self.clientId)


        for human in self.humans:
            human_done = human.step()
            if human_done:
                h_s_x = self.np_random.uniform(self.sp_init_pos[0]-7.5, self.sp_init_pos[0]+7.5)
                h_s_y = self.np_random.uniform(-0.5, self.corridor_width+0.5)
                h_e_x = self.np_random.uniform(self.sp_init_pos[0]-7.5, self.sp_init_pos[0]+7.5)
                h_e_y = self.np_random.uniform(-0.5, self.corridor_width+0.5)
                human.set_start_end([h_s_x, h_s_y], [h_e_x, h_e_y])
                human.setEnabled(self.np_random.uniform() <= self.p['world']['p_spawn_human'])
        # Execute one step in simulation
        pb.stepSimulation(self.clientId)
        self.valid_buffer_scan = False

        # Update internal state
        self.state = {"base_vel": mb_vel_n_r, "joint_vel": joint_vel_n}

        # Calculate reward
        reward, done, info = self.calculate_reward()
        if done:
            self.done_info = info
        else:
            self.done_info = None

        # Obtain observation
        obs = self.calculate_observation()

        return obs, reward, done, info

    def render(self, mode='human', w=1280, h=720):
        '''
        Renders the environment. Currently does nothing.
        '''
        if mode == 'rgb_array' or mode == 'human_fast' or not self.free_cam:
            camDistance = 4
            dis, _ = self.calculate_goal_distance()
            x1, y1 = 0.5, 1.5
            x2, y2 = 2.0, 4.0
            f = lambda x: min(y2, max(y1, (y2-y1)/(x2-x1)*x+y1-(y2-y1)/(x2-x1)*x1))
            camDistance = f(dis)

            x1, y1 = 0.5, -55
            x2, y2 = 2.0, -80
            f = lambda x: min(max(y1, y2), max(min(y1, y2), (y2-y1)/(x2-x1)*x+y1-(y2-y1)/(x2-x1)*x1))
            pitch = f(dis)
            nearPlane = 0.01
            farPlane = 15
            fov = 60

            cam_pos, rpy = self.get_camera_pose()


            viewMatrix = pb.computeViewMatrixFromYawPitchRoll(cam_pos,
                                                              camDistance,
                                                              rpy[2], pitch,
                                                              rpy[0], 2,
                                                              self.clientId)
        if not self.free_cam:
            pb.resetDebugVisualizerCamera(camDistance, rpy[2], pitch, cam_pos,
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

    def setup_simulation(self, gui=False, easy_bookcases=False, clientId=None):
        '''
        Initializes the simulation by setting up the environment and spawning
        all objects used later.

        Params:
            gui (bool): Specifies if a GUI should be spawned.
        '''
        # Setup simulation parameters
        if clientId is None:
            mode = pb.GUI if gui else pb.DIRECT
            self.clientId = pb.connect(mode)
        else:
            self.clientId = clientId
        pb.setGravity(0.0, 0.0, 0.0, self.clientId)
        pb.setPhysicsEngineParameter(fixedTimeStep=self.p["world"]["tau"],
                                     physicsClientId=self.clientId)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())

        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        pb.setPhysicsEngineParameter(enableFileCaching=0)

        # Setup humans
        self.humans = [Human(self.clientId, self.tau) for _ in range(self.p['world']['n_humans'])]

        # Spawn robot
        self.robotId = self.spawn_robot()

        # Spawn setpoint
        self.spId = self.spawn_setpoint()

        # Spawn all objects in the environment
        self.additionalIds = self.spawn_additional_objects()

        # Enable collision of base and all objects
        # for id in self.additionalIds:
        #     pb.setCollisionFilterPair(self.robotId, id, -1, -1, True,
        #                               self.clientId)

        # Spawn bookcases
        self.spawn_kallax()

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

        for human in self.humans:
            self.configure_ext_collisions(human.leg_l, self.robotId, self.collision_links)
            self.configure_ext_collisions(human.leg_r, self.robotId, self.collision_links)

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


    def get_ee_velocity(self):
        state_ee = pb.getLinkState(self.robotId, self.eeLinkId, True, False,
                                   self.clientId)

        return np.array(state_ee[6])
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

    def get_base_sp_transform(self):
        '''
        Calculates pose of setpoint w.r.t. base frame.

        Returns:
            numpy.array: 6D pose of setpoint in base frame.
        '''
        state_ee = pb.getLinkState(self.robotId, self.baseLinkId,
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
        if self.valid_buffer_scan:
            return self.last_scan
        '''
        Obtain lidar scan values for current state.

        Returns:
            list: Scan values for range and resolution specified in
                params dict.
        '''
        scan_front = None
        scan_rear = None


        # Get pose of lidar
        states = pb.getLinkState(self.robotId, self.lidarLinkId1,
                                 False, False, self.clientId)
        lidar_pos, lidar_ori = states[4:6]
        lidar_pos = np.array(lidar_pos)
        R = np.array(pb.getMatrixFromQuaternion(lidar_ori))
        R = np.reshape(R, (3, 3))
        scan_l1 = R.dot(self.rays[0]).T + lidar_pos
        scan_h1 = R.dot(self.rays[1]).T + lidar_pos

        # Get pose of lidar
        states = pb.getLinkState(self.robotId, self.lidarLinkId2,
                                 False, False, self.clientId)
        lidar_pos, lidar_ori = states[4:6]
        lidar_pos = np.array(lidar_pos)
        R = np.array(pb.getMatrixFromQuaternion(lidar_ori))
        R = np.reshape(R, (3, 3))
        scan_l2 = R.dot(self.rays[0]).T + lidar_pos
        scan_h2 = R.dot(self.rays[1]).T + lidar_pos

        scan_l = np.concatenate((scan_l1, scan_l2))
        scan_h = np.concatenate((scan_h1, scan_h2))

        scan_r = pb.rayTestBatch(scan_l.tolist(), scan_h.tolist(),
                                 self.clientId)

        scan = [x[2]*self.p["sensors"]["lidar"]["range"] for x in scan_r]
        scan_front = scan[:len(scan_l1)]
        scan_rear = scan[len(scan_l1):]
        self.last_scan = [scan_front, scan_rear]
        self.valid_buffer_scan = True
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
        filt = self.actuator_selection
        max_reached = (filt*((self.joint_limits[:, 1] - j_pos) <= 1e-3)).any()
        min_reached = (filt*((j_pos - self.joint_limits[:, 0]) <= 1e-3)).any()

        return min_reached or max_reached

    def check_collision(self):
        '''
        Checks if robot collides with any body.

        Returns:
            bool: Whether robot is in collision or not.
        '''
        return len(pb.getContactPoints(bodyA=self.robotId,
                                       physicsClientId=self.clientId)) > 0

    def close(self):
        '''
        Closes the environment.
        '''
        try:
            pb.disconnect(self.clientId)
        except Exception:
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
        ang_noise = 0.4
        ang = -1.5708 + self.np_random.uniform(-ang_noise, ang_noise)
        ori_quat = pb.getQuaternionFromEuler([0, 0, ang])
        pb.resetBasePositionAndOrientation(self.spId, pos_sp, ori_quat,
                                           self.clientId)
        pb.resetBasePositionAndOrientation(self.markerId, pos_mk, ori_quat,
                                           self.clientId)
        pb.stepSimulation(self.clientId)
        self.valid_buffer_scan = False

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
        rpy = [0, -75, yaw]
        return cam_pos, rpy

    def calculate_success_rate(self):
        if self.episode_counter <= 1:
            return 0
        else:
            return self.success_counter / (self.episode_counter - 1)

    def save_world(self, dir, pre_f, inf_f, ep):
        wn = "{}.video.{}.video{:06}.world.world".format(pre_f, inf_f, ep)
        sn = "{}.video.{}.video{:06}.setpoints.yaml".format(pre_f, inf_f, ep)
        world_path = os.path.join(dir, wn)
        sp_path = os.path.join(dir, sn)
        pb.saveWorld(world_path, self.clientId)
        with open(sp_path, 'w') as f:
            yaml.dump(self.sp_history, f)

    def numpyfy_dict(self, input):
        if isinstance(input, list):
            sol_n = not (False in [isinstance(x, (float, int)) for x in input])
            sol_s = not (False in [isinstance(x, str) for x in input])
            if sol_n:
                return np.array(input)
            elif sol_s:
                return input
            else:
                for i in range(len(input)):
                    input[i] = self.numpyfy_dict(input[i])
                return input
        if isinstance(input, dict):
            for key in input:
                input[key] = self.numpyfy_dict(input[key])

            return input

        return input

    def set_param(self, param, value):
        key_list = param.split(".")
        obj = self.p
        for key in key_list[:-1]:
            if isinstance(obj, (dict,)):
                obj = obj[key]
            elif isinstance(obj, (list,)):
                obj = obj[int(key)]
            else:
                print("ERROR: curriculum learning has wrong param path.")
                return False
        if isinstance(obj, (dict,)):
            obj[key_list[-1]] = value
        elif isinstance(obj, (list, np.ndarray)):
            obj[int(key_list[-1])] = value
        else:
            print("ERROR: curriculum learning has wrong param path.")
            return False
        return True

    def get_param(self, param):
        key_list = param.split(".")
        obj = self.p
        for key in key_list:
            if isinstance(obj, (dict,)):
                obj = obj[key]
            elif isinstance(obj, (list,)):
                obj = obj[int(key)]
            else:
                print("ERROR: curriculum learning has wrong param path.")
                return False
        return obj


class Human:
    def __init__(self, clientId, tau, vel_range=[0.05, 0.15],
                 leg_dis_range=[0.1, 0.4], step_length_range=[0.3, 0.7]):
        self.clientId = clientId
        self.vel_range = vel_range
        self.leg_dis_range = leg_dis_range
        self.step_len_range = step_length_range
        self.tau = tau
        self.T = 0.0
        self.enabled = True

        self.v_l, self.v_r = lambda x: 0, lambda x: 0

        # Load legs
        model_path = 'urdf/human/leg1.urdf'
        self.leg_l = pb.loadURDF(model_path,
                                 useFixedBase=True,
                                 physicsClientId=self.clientId)
        self.leg_r = pb.loadURDF(model_path,
                                 useFixedBase=True,
                                 physicsClientId=self.clientId)

    def set_start_end(self, start_pos, end_pos):
        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)
        self.vel = np.random.uniform(*self.vel_range)
        self.leg_dis = np.random.uniform(*self.leg_dis_range)
        self.step_len = np.random.uniform(*self.step_len_range)

        self.n_dir = ((self.end_pos - self.start_pos)
                      / (np.linalg.norm(self.end_pos - self.start_pos)))
        n_orth = np.array((-self.n_dir[1], self.n_dir[0]))

        # Reset leg pos
        p_l = self.start_pos + n_orth*self.leg_dis/2.0
        p_r = self.start_pos - n_orth*self.leg_dis/2.0
        pb.resetBasePositionAndOrientation(self.leg_l,
                                           [p_l[0], p_l[1], 0], [0, 0, 0, 1],
                                           physicsClientId=self.clientId)
        pb.resetBasePositionAndOrientation(self.leg_r,
                                           [p_r[0], p_r[1], 0], [0, 0, 0, 1],
                                           physicsClientId=self.clientId)

        # Calculate feet velocity functions
        a = np.pi*self.vel
        b = 2*a/self.step_len
        self.v_l = lambda x: max(0.0, a*np.sin(b*x + np.pi/2.0))
        self.v_r = lambda x: max(0.0, a*np.sin(b*x + 3*np.pi/2.0))

        # Calculate time to travel
        self.T = 0.0
        self.T_max = np.linalg.norm(self.end_pos - self.start_pos)/self.vel

    def step(self):
        if not self.enabled:
            if self.T >= self.T_max:
                return True
            else:
                return False
        v_l = self.v_l(self.T)*self.n_dir
        v_r = self.v_r(self.T)*self.n_dir

        pb.resetBaseVelocity(self.leg_l, [v_l[0], v_l[1], 0], [0, 0, 0],
                             self.clientId)
        pb.resetBaseVelocity(self.leg_r, [v_r[0], v_r[1], 0], [0, 0, 0],
                             self.clientId)

        self.T += self.tau

        if self.T >= self.T_max:
            return True
        else:
            return False

    def setEnabled(self, enabled):
        self.enabled = enabled
        if not self.enabled:
            pb.resetBasePositionAndOrientation(self.leg_r,
                                               [0, 0, 100], [0, 0, 0, 1],
                                               physicsClientId=self.clientId)
            pb.resetBasePositionAndOrientation(self.leg_l,
                                               [0, 0, 100], [0, 0, 0, 1],
                                               physicsClientId=self.clientId)
