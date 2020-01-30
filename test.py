import time
from envs.aslaug_v12 import AslaugEnv


N = 100000
env = AslaugEnv(gui=True)
env.reset()
ts = time.time()
for i in range(N):
    a = env.action_space.sample()
    a = a * 0 + 3
    a[3] = 2
    o,r,d,i = env.step(a)
    time.sleep(0.025)
    # print(env.joint_limits)
    if d:
        env.reset()
te = time.time()
print("Took {}s".format(te-ts))
print("Runs at {}Hz".format(N/(te-ts)))
print("Corresponds to RTF of {}".format(N/(te-ts)/50.0))
