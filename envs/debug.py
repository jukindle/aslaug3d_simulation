from aslaug_v0 import AslaugEnv
import numpy as np
import time

env = AslaugEnv(gui=True)

env.reset()

nsteps = 100000
ts = time.time()
for i in range(nsteps):
    action = np.ones(10, dtype=int)*3
    action[2] = 4
    action[1] = 4
    action[3] = 3
    obs, rew, done, info = env.step(action)
    print(obs.shape)
    print(rew)
    print(done)
    time.sleep(1/50.0)
te = time.time()
print("RTF: {}".format(nsteps/(te-ts)/50))
