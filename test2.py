import time
from envs.aslaug_v121easyhumans import AslaugEnv
import pybullet as pb
import numpy as np

N = 100000000
env = AslaugEnv(gui=True)
env.reset()
env.human.set_start_end([-2, -11], [2, -11])

robot_ori = pb.getQuaternionFromEuler([0, 0,
                                       -np.pi/2])
pb.resetBasePositionAndOrientation(env.robotId, [0, -10, 0.08],
                                   robot_ori, env.clientId)

pb.resetJointState(env.robotId, env.joint_mapping[0],
                   0, 0.0, env.clientId)
pb.resetJointState(env.robotId, env.joint_mapping[3],
                   -0.5, 0.0, env.clientId)


ts = time.time()
for i in range(N):
    a = env.action_space.sample()
    a = a*0+2
    # a = a * 0 + 3
    # a[3] = 2
    o,r,d,i = env.step(a)
    # time.sleep(0.025)
    # print(env.joint_limits)
    if d:
        env.reset()
        env.human.set_start_end([-2, -11], [2, -11])

        robot_ori = pb.getQuaternionFromEuler([0, 0,
                                               -np.pi/2])
        pb.resetBasePositionAndOrientation(env.robotId, [0, -10, 0.08],
                                           robot_ori, env.clientId)

        pb.resetJointState(env.robotId, env.joint_mapping[0],
                           0, 0.0, env.clientId)
        pb.resetJointState(env.robotId, env.joint_mapping[3],
                           -0.5, 0.0, env.clientId)
te = time.time()
print("Took {}s".format(te-ts))
print("Runs at {}Hz".format(N/(te-ts)))
print("Corresponds to RTF of {}".format(N/(te-ts)/50.0))
