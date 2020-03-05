import pybullet as pb
import numpy as np


class AslaugObject:
    def __init__(self, objectId, clientId=0):
        self.id = objectId
        self.clientId = clientId

        self.jointIds = range(pb.getNumJoints(self.id,
                                              physicsClientId=self.clientId))

        self.linkIds = [-1] + [pb.getJointInfo(self.id, x,
                                               physicsClientId=self.clientId)
                               [0] for x in self.jointIds]

    def setId(self, id):
        self.id = id

    def setupCollisions(self, extBodyId, extLinkIds=-1, enabled=True):
        if isinstance(extLinkIds, int):
            extLinkIds = [extLinkIds]
        for extLinkId in extLinkIds:
            for intLinkId in self.linkIds:
                pb.setCollisionFilterPair(self.id, extBodyId, intLinkId,
                                          extLinkId, enabled, self.clientId)

    def setupSelfCollisions(self, enabledPairNames):
        pairs = ["{}|{}".format(x, y) for x, y in enabledPairNames]
        for j in range(pb.getNumJoints(self.id, physicsClientId=self.clientId)):
            info_j = pb.getJointInfo(self.id, j, physicsClientId=self.clientId)
            link_name_j = info_j[12].decode("utf-8")
            idx_j = info_j[0]

            for k in range(pb.getNumJoints(self.id, physicsClientId=self.clientId)):
                info_k = pb.getJointInfo(self.id, k, physicsClientId=self.clientId)
                link_name_k = info_k[12].decode("utf-8")
                idx_k = info_k[0]

                s1 = "{}|{}".format(link_name_j, link_name_k)
                s2 = "{}|{}".format(link_name_k, link_name_j)
                enabled = s1 in pairs or s2 in pairs
                pb.setCollisionFilterPair(self.id, self.id, idx_j, idx_k, enabled,
                                          self.clientId)

    def moveObject(self, xyz=[0, 0, 0], rpy=[0, 0, 0]):
        quat = pb.getQuaternionFromEuler(rpy)
        pb.resetBasePositionAndOrientation(self.id, xyz, quat, self.clientId)

    def moveObject2D(self, pose2d, z=0.0):
        xyz = [pose2d[0], pose2d[1], z]
        rpy = [0.0, 0.0] + [pose2d[2]]
        self.moveObject(xyz=xyz, rpy=rpy)

    def getPose(self):
        state_ee = pb.getLinkState(self.id, -1,
                                   False, False, self.clientId)
        xyz, quat = state_ee[4:6]
        rpy = pb.getEulerFromQuaternion(quat, self.clientId)

        return np.array(xyz), np.array(rpy)

    def remove(self):
        pb.removeBody(self.id, physicsClientId=self.clientId)

    def setVelocity2D(self, twist2d):
        vlin, vang = twist2d[0:2] + [0], [0, 0] + twist2d[2:3]
        pb.resetBaseVelocity(self.id, vlin, vang, self.clientId)


class AslaugCuboid(AslaugObject):
    def __init__(self, halfExtents, xyz, clientId=0):
        colBoxId = pb.createCollisionShape(pb.GEOM_BOX,
                                           halfExtents=halfExtents)
        visBoxId = pb.createCollisionShape(pb.GEOM_BOX,
                                           halfExtents=halfExtents)
        id = pb.createMultiBody(0, colBoxId, visBoxId, xyz)
        super(AslaugCuboid, self).__init__(id, clientId)


class AslaugURDFObject(AslaugObject):
    def __init__(self, urdfPath, xyz=[0, 0, 0], rpy=[0, 0, 0], dynamic=False,
                 clientId=0, flags=0):
        quat = pb.getQuaternionFromEuler(rpy)
        id = pb.loadURDF(urdfPath, xyz, quat, useFixedBase=(not dynamic),
                         physicsClientId=clientId, flags=flags)
        super(AslaugURDFObject, self).__init__(id, clientId)


class AslaugKallax(AslaugURDFObject):
    def __init__(self, pose2d=[0, 0, 0], clientId=0):
        urdfPath = 'urdf/kallax/kallax_large_easy.urdf'
        super(AslaugKallax, self).__init__(urdfPath, clientId=clientId)
        self.moveObject2D(pose2d)

    def sampleSetpointPoses(self, N, rx, ry, rz, layers=[0, 1, 2, 3]):
        xyz, rpy = self.getPose()
        ang = rpy[2]
        R = np.array([[np.cos(ang),  -np.sin(ang),   0],
                      [np.sin(ang),  +np.cos(ang),   0],
                      [0,            0,              1]])
        poses = []
        for _ in range(N):
            layer = np.random.choice(layers)
            x, y, z = 0, 0.195, 0.037 + (0.33 + 0.025) * layer
            pos = xyz + R.T.dot(np.array([x, y, z]))
            nx = self.np_random.uniform(*rx)
            ny = self.np_random.uniform(*ry)
            nz = self.np_random.uniform(*rz)
            pos += np.array((nx, ny, nz))
            poses.append(pos)

        return poses


class AslaugRobot(AslaugURDFObject):
    def __init__(self, params, clientId=0):
        urdfPath = 'urdf/robot/mopa/mopa.urdf'
        flags = (pb.URDF_USE_SELF_COLLISION
                 | pb.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
        super(AslaugRobot, self).__init__(urdfPath, xyz=[0, 0, 0],
                                          clientId=clientId, flags=flags)
        self.valid_buffer_scan = False
        self.p = params
        self.scanCalib = None

        # Setup internal variables
        self.setupInternalVariables()

    def setupInternalVariables(self):
        # Setup ray variable for faster lidar
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

        for j in range(pb.getNumJoints(self.id,
                                       physicsClientId=self.clientId)):
            info = pb.getJointInfo(self.id, j,
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

        self.actuator_selection = np.zeros(7, bool)
        for i, name in enumerate(joint_names):
            if name in self.p["joints"]["joint_names"]:
                self.actuator_selection[i] = 1

    def calibrateLidar(self):
        self.moveObject(xyz=[0, 0, 10])
        urdfPathLidar = 'urdf/calibration/ridgeback_lidar_calib.urdf'
        lidarBox = AslaugURDFObject(urdfPathLidar, xyz=[0, 0, 10],
                                    clientId=self.clientId)
        pb.stepSimulation(self.clientId)
        scanRet = self.getLidarScan()
        self.scanCalib = scanRet
        lidarBox.remove()
        return scanRet

    def getLidarCalibration(self):
        if self.scanCalib is None:
            return self.calibrateLidar()
        else:
            return self.scanCalib

    def getLidarScan(self):
        if self.valid_buffer_scan:
            return self.last_scan

        # Get pose of lidar
        states = pb.getLinkState(self.id, self.lidarLinkId1,
                                 False, False, self.clientId)
        lidar_pos, lidar_ori = states[4:6]
        lidar_pos = np.array(lidar_pos)
        R = np.array(pb.getMatrixFromQuaternion(lidar_ori))
        R = np.reshape(R, (3, 3))
        scan_l1 = R.dot(self.rays[0]).T + lidar_pos
        scan_h1 = R.dot(self.rays[1]).T + lidar_pos

        # Get pose of lidar
        states = pb.getLinkState(self.id, self.lidarLinkId2,
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

    def resetFixedJoints(self):
        up = self.p['joints']['static_act_noise_mag']
        self.fixed_joint_states = (np.array(self.p["joints"]["init_states"])
                                   + self.np_random.uniform(-up, up))
        for i in range(len(self.joint_mapping)):
            pb.resetJointState(self.id, self.joint_mapping[i],
                               self.fixed_joint_states[i],
                               0.0, self.clientId)

    def setJointState(self, idx, val):
        pb.resetJointState(self.id, self.joint_mapping[idx],
                           val, 0.0, self.clientId)

    def getEEPose(self):
        state_ee = pb.getLinkState(self.id, self.eeLinkId,
                                   False, False, self.clientId)
        ee_pos_w, ee_ori_w = state_ee[4:6]
        return ee_pos_w, ee_ori_w

    def getEEVelocity(self):
        state_ee = pb.getLinkState(self.id, self.eeLinkId, True, False,
                                   self.clientId)

        return np.array(state_ee[6])

    def setVelocities(self, mb_vel_r, joint_vel):
        '''
        Applies velocities of move base and joints to the simulation.

        Args:
            mb_vel_r (numpy.array): Base velocities to apply (x, y, theta).
            joint_vel (numpy.array): Joint velocities to apply. Length: 7.
        '''
        # Obtain robot orientation and transform mb vels to world frame
        xyz, rpy = self.getPose()
        mb_vel_w = self.rotation_matrix(rpy[2]).dot(mb_vel_r)

        # Apply velocities to simulation
        self.setVelocity2D(mb_vel_w)

        pb.setJointMotorControlArray(self.id, self.joint_mapping,
                                     pb.VELOCITY_CONTROL,
                                     targetVelocities=joint_vel,
                                     forces=len(joint_vel)*[1e10],
                                     physicsClientId=self.clientId)

    def getEESPTransform(self, sp_pos_w, sp_ori_w):
        '''
        Calculates pose of setpoint w.r.t. end effector frame.

        Returns:
            numpy.array: 6D pose of setpoint in end effector frame.
        '''
        ee_pos_w, ee_ori_w = self.getEEPose()
        w_pos_ee, w_ori_ee = pb.invertTransform(ee_pos_w, ee_ori_w,
                                                self.clientId)

        sp_pos_ee, sp_ori_ee = pb.multiplyTransforms(w_pos_ee, w_ori_ee,
                                                     sp_pos_w, sp_ori_w,
                                                     self.clientId)

        sp_eul_ee = pb.getEulerFromQuaternion(sp_ori_ee, self.clientId)

        return np.array(sp_pos_ee + sp_eul_ee)

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

    def getBaseVelocity(self):
        '''
        Obtain base velocities.

        Returns:
            numpy.array: Velocities of movebase (x, y, theta).
        '''
        state = pb.getLinkState(self.id, self.baseLinkId, True,
                                False, self.clientId)

        mb_ang_w = pb.getEulerFromQuaternion(state[5])[2]
        v_lin, v_ang = state[6:8]
        mb_vel_w = np.array(v_lin[0:2] + v_ang[2:3])
        return self.rotation_matrix(mb_ang_w).T.dot(mb_vel_w)

    def getBaseSPTransform(self, sp_pos_w, sp_ori_w):
        '''
        Calculates pose of setpoint w.r.t. base frame.

        Returns:
            numpy.array: 6D pose of setpoint in base frame.
        '''
        state_ee = pb.getLinkState(self.id, self.baseLinkId,
                                   False, False, self.clientId)
        ee_pos_w, ee_ori_w = state_ee[4:6]
        w_pos_ee, w_ori_ee = pb.invertTransform(ee_pos_w, ee_ori_w,
                                                self.clientId)

        sp_pos_ee, sp_ori_ee = pb.multiplyTransforms(w_pos_ee, w_ori_ee,
                                                     sp_pos_w, sp_ori_w,
                                                     self.clientId)

        sp_eul_ee = pb.getEulerFromQuaternion(sp_ori_ee, self.clientId)

        return np.array(sp_pos_ee + sp_eul_ee)

    def getLinkStates(self, links_idx):
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
        states = pb.getLinkStates(self.id, links_idx, False, False,
                                  self.clientId)
        states_mb = pb.getLinkState(self.id, self.baseLinkId, False,
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

    def getJointStates(self, sel=None):
        '''
        Obtain joint positions and velocities.

        Returns:
            numpy.array: Joint positions in radians.
            numpy.array: Joint velocities in radians/s.
        '''
        states = pb.getJointStates(self.id, self.joint_mapping,
                                   self.clientId)
        j_pos = [x[0] for x in states]
        j_vel = [x[1] for x in states]

        if sel is None:
            return np.array(j_pos), np.array(j_vel)
        else:
            return np.array(j_pos)[sel], np.array(j_vel)[sel]

    def isColliding(self):
        '''
        Checks if robot collides with any body.

        Returns:
            bool: Whether robot is in collision or not.
        '''
        return len(pb.getContactPoints(bodyA=self.id,
                                       physicsClientId=self.clientId)) > 0
